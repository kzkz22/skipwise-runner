"""The desktop app's moving parts: the worker thread, pairing, the test run (round 7c)."""
from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Any

from .. import activity, config, gpuswap, steps, trial, worker
from ..protocol import Central, CentralError, Outdated, Unauthorized
from ..i18n import gettext as _

log = logging.getLogger("dwyt-runner")


class Runtime:
    def __init__(self, start_worker: bool = True) -> None:
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.error = ""
        self.lock = threading.Lock()
        self.pairing: dict[str, Any] = {}
        self.trial: dict[str, Any] = {"running": False, "stage": "", "error": ""}
        self.closing = False
        self.machine: dict[str, Any] = {"m": None, "said": [], "at": 0.0, "scanning": False, "local_models": []}
        self.prepare: dict[str, Any] = {"running": False, "label": "", "file": "", "got": 0, "total": 0,
                                        "error": "", "ready": False}
        self.pull: dict[str, Any] = {"running": False, "name": "", "fraction": None, "status": "", "error": ""}
        self.starting: dict[str, Any] = {"running": False, "kind": "", "since": 0.0, "error": ""}
        self.refresh_machine()  # the machine page then opens at once
        if start_worker:
            self.start_worker()

    # -- the worker --
    def start_worker(self) -> None:
        cfg = config.load()
        if self.closing or not cfg.paired or (self.thread and self.thread.is_alive()):
            return
        self.stop = threading.Event()
        self.error = ""
        self.thread = threading.Thread(target=self._work, args=(cfg, self.stop), daemon=True, name="worker")
        self.thread.start()

    def _work(self, cfg: config.Config, stop: threading.Event) -> None:
        try:
            worker.loop(cfg, stop)
        except Unauthorized:
            self.error = _("A Skipwise nem fogadja el a párosítást (visszavonták?). Párosítsd újra.")
        except Outdated as exc:
            self.error = _("{exc} Frissítsd a futtatót.").format(exc=exc)
        except Exception as exc:  # noqa: BLE001 - the window must say it, not die with it
            log.exception("worker stopped")
            self.error = _("A futtató leállt: {exc}").format(exc=exc)

    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def settings_changed(self) -> None:
        """New settings take effect before the next job (never in the middle of one)."""
        worker.RELOAD.set()
        self.start_worker()

    def shutdown(self) -> None:
        self.closing = True
        self.stop.set()

    def reset(self, models: bool = False) -> list[str]:
        """Alaphelyzet: stop the worker (a job it was asking for goes back), then delete settings and keys.
        The window then starts on the setup steps; a new pairing starts a new worker."""
        from .. import reset
        self.stop.set()
        if self.thread is not None:
            # no waiting for a long poll (up to ~90 s): a job that arrives after this is handed back unread,
            # and the worker never writes the settings
            self.thread.join(timeout=3)
        said = reset.run(models)
        self.error = ""
        with self.lock:
            self.pairing = {}
        self.trial = {"running": False, "stage": "", "error": ""}
        self.machine["at"] = 0.0  # look again (what is installed did not change, the snapshot did)
        return said

    # -- pairing --
    def start_pairing(self, server: str, name: str) -> None:
        server = server.strip().rstrip("/")
        if not server.startswith(("http://", "https://")):
            server = "https://" + server
        central = Central(server)
        start = central.pair_start()
        with self.lock:
            self.pairing = {"server": server, "name": name.strip() or socket.gethostname(), "code": start["code"],
                            "expires": time.time() + int(start.get("expires_in", 600)), "state": "waiting",
                            "error": ""}
        threading.Thread(target=self._poll_pairing, args=(central, start["poll_id"]), daemon=True).start()

    def _poll_pairing(self, central: Central, poll_id: str) -> None:
        while time.time() < self.pairing.get("expires", 0):
            time.sleep(3)
            try:
                done = central.pair_poll(poll_id)
            except CentralError as exc:
                with self.lock:
                    self.pairing.update(state="error", error=str(exc))
                return
            if done:
                cfg = config.load()
                cfg.server, cfg.token, cfg.runner_id = self.pairing["server"], done["token"], done["runner_id"]
                cfg.name = self.pairing["name"] or done.get("name") or socket.gethostname()
                config.save(cfg)
                self.start_worker()
                with self.lock:
                    self.pairing.update(state="done", code="")
                return
        with self.lock:
            self.pairing.update(state="expired", code="")

    # -- the machine check, kept for the machine page --
    MACHINE_FRESH_S = 300

    def refresh_machine(self) -> bool:
        if self.machine["scanning"]:
            return False
        self.machine["scanning"] = True

        def scan() -> None:
            from .. import localllm, machine
            try:
                cfg = config.load()
                m = machine.scan(cfg)
                said = machine.changes(machine.load_snapshot(), m)
                machine.save_snapshot(m)
                models: list[str] = []
                server = localllm.server(cfg)
                if server and server.running():
                    try:
                        models = [x.name for x in server.models()]
                    except Exception:  # noqa: BLE001
                        models = []
                self.machine.update(m=m, said=said, at=time.time(), local_models=models)
            except Exception:  # noqa: BLE001
                log.exception("machine check")
            finally:
                self.machine["scanning"] = False
        threading.Thread(target=scan, daemon=True, name="machine-scan").start()
        return True

    def rescan_and_wait(self, timeout: float = 30) -> None:
        deadline = time.monotonic() + timeout
        while self.machine["scanning"] and time.monotonic() < deadline:
            time.sleep(0.1)
        self.refresh_machine()
        while self.machine["scanning"] and time.monotonic() < deadline:
            time.sleep(0.1)

    def machine_view(self) -> dict[str, Any]:
        """The last scan at once; a fresh one starts in the background when it is old."""
        if self.machine["m"] is not None and time.time() - self.machine["at"] > self.MACHINE_FRESH_S:
            self.refresh_machine()
        return dict(self.machine)

    # -- the built-in Whisper's files (setup: Hozzávalók) --
    def start_prepare(self, model: str) -> bool:
        from .. import whispercpp
        if self.prepare["running"]:
            return False
        self.prepare.update(running=True, label="", file="", got=0, total=0, error="", ready=False)

        def progress(name: str, got: int, total: int) -> None:
            self.prepare.update(file=name, got=got, total=total)

        def run() -> None:
            whispercpp.ON_DOWNLOAD = progress
            try:
                whispercpp.prepare(config.load(), model, lambda label: self.prepare.update(
                    label=label, file="", got=0, total=0))
                self.prepare["ready"] = True
            except Exception as exc:  # noqa: BLE001 - shown in the window
                log.exception("whisper prepare")
                self.prepare["error"] = str(exc)
            finally:
                whispercpp.ON_DOWNLOAD = None
                self.prepare["running"] = False
        threading.Thread(target=run, daemon=True, name="prepare").start()
        return True

    # -- starting a local model server (setup: Hozzávalók) --
    def start_server(self, kind: str) -> bool:
        from dataclasses import replace

        from .. import localllm
        if self.starting["running"]:
            return False
        self.starting = {"running": True, "kind": kind, "since": time.time(), "error": ""}

        def run() -> None:
            try:
                localllm.server(replace(config.load(), local_llm=kind)).ensure_running()
            except Exception as exc:  # noqa: BLE001 - shown in the window
                log.warning("starting %s: %s", kind, exc)
                self.starting["error"] = str(exc)
            finally:
                self.rescan_and_wait()
                self.starting["running"] = False
        threading.Thread(target=run, daemon=True, name="server-start").start()
        return True

    # -- a local model (setup: Hozzávalók, Saját gép) --
    def start_pull(self, kind: str, name: str) -> bool:
        from .. import localllm
        if self.pull["running"]:
            return False
        self.pull = {"running": True, "name": name, "fraction": None, "status": "indul", "error": ""}

        def run() -> None:
            try:
                cfg = config.load()
                cfg.local_llm = kind
                server = localllm.server(cfg)
                server.ensure_running()
                server.pull(name, lambda fraction, status: self.pull.update(fraction=fraction, status=status))
                self.rescan_and_wait()  # the model list changed: the page shows the new one
            except Exception as exc:  # noqa: BLE001
                log.exception("model pull")
                self.pull["error"] = str(exc)
            finally:
                self.pull["running"] = False
        threading.Thread(target=run, daemon=True, name="pull").start()
        return True

    # -- the test run --
    def start_trial(self, llm_model: str = "", server_kind: str = "") -> bool:
        if self.trial["running"]:
            return False
        self.trial = {"running": True, "stage": "indul", "error": ""}

        def run() -> None:
            try:
                cfg = config.load()
                if server_kind and not cfg.local_llm:  # setup: a detected server, before it is chosen
                    cfg.local_llm = server_kind
                trial.run(cfg, "large-v3", lambda s: self.trial.update(
                    stage=activity.STAGE_LABELS.get(s, s)), llm_model)
            except Exception as exc:  # noqa: BLE001
                log.exception("test run")
                self.trial["error"] = str(exc)
            finally:
                self.trial["running"] = False
        threading.Thread(target=run, daemon=True, name="trial").start()
        return True

    # -- what the Status tab shows --
    def status(self) -> dict[str, Any]:
        cfg = config.load()
        cur = activity.current()
        if cur:
            cur["kind_label"] = activity.KIND_LABELS.get(cur.get("kind"), cur.get("kind") or "")
            cur["stage_label"] = activity.STAGE_LABELS.get(cur.get("stage"), cur.get("stage") or "")
            p = cur.get("progress") or {}
            cur["part"] = _("{get}/{get2}. rész").format(
                get=p.get('part'), get2=p.get('parts')) if p.get("parts") else ""
            cur["minutes"] = int((time.time() - cur.get("started", time.time())) // 60)
            tracker = cur.pop("tracker", None)  # the step list with each step's place (19.34)
            cur["steps"] = [dict(s, label=activity.STAGE_LABELS.get(s["id"], s["id"]),
                                 place=steps.place_label(s), part=cur["part"] if s["state"] == "now" else "")
                            for s in (tracker.payload() if tracker is not None else [])]
        recent = [dict(e, kind_label=activity.KIND_LABELS.get(e.get("kind"), e.get("kind") or ""))
                  for e in reversed(activity.history()[-8:])]
        swap = dict(gpuswap.STATE)
        with self.lock:
            pairing = dict(self.pairing)
        return {"paired": cfg.paired, "server": cfg.server, "name": cfg.name, "running": self.running(),
                "error": self.error, "current": cur, "recent": recent, "today": activity.today(),
                "swap": swap, "pairing": pairing}
