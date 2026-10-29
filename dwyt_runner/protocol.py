"""Client side of the central–runner protocol (dwyt-docs/docs/protokoll.md)."""
from __future__ import annotations

import os
import time
from typing import Any

import httpx

from . import __version__
from .i18n import gettext as _

PROTOCOL = 1


PARTIAL_CHARS = 20000  # the central keeps at most this much of an answer being written


class CentralError(Exception):
    """The central service could not be reached or answered unexpectedly: try again later."""


class Unauthorized(Exception):
    """The token is invalid or revoked: pair again."""


class Outdated(Exception):
    """The central service does not speak this protocol version: update the runner."""


class JobGone(Exception):
    """The job is no longer ours (cancelled, deleted or reassigned)."""


class Central:
    def __init__(self, server: str, token: str = "", transport: httpx.BaseTransport | None = None,
                 timeout: float = 60) -> None:
        self.server = server.rstrip("/")
        self.token = token
        headers = {"Dwyt-Protocol": str(PROTOCOL), "User-Agent": f"dwyt-runner/{__version__}"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.http = httpx.Client(base_url=self.server, headers=headers, timeout=timeout, transport=transport)
        self.settings_pending = False  # the last claim said settings changes wait (Dwyt-Settings: pending)

    def close(self) -> None:
        self.http.close()

    def _call(self, method: str, path: str, **kw) -> httpx.Response:
        try:
            resp = self.http.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise CentralError(_("A központ nem érhető el ({server}): {exc}").format(
                server=self.server, exc=exc)) from exc
        if resp.status_code == 401:
            raise Unauthorized(_message(resp))
        if resp.status_code == 426:
            raise Outdated(_message(resp))
        if resp.status_code in (404, 409, 410) and "/jobs/" in path:
            raise JobGone(_message(resp))
        if resp.status_code >= 500 or resp.status_code in (408, 429):
            raise CentralError(_("A központ hibát adott (HTTP {status_code}).").format(status_code=resp.status_code))
        return resp

    # pairing
    def pair_start(self) -> dict[str, Any]:
        resp = self._call("POST", "/runner/v1/pair/start")
        if resp.status_code != 200:
            raise CentralError(_("A párosítás nem indult el (HTTP {status_code}): {_message}").format(
                status_code=resp.status_code, _message=_message(resp)))
        return resp.json()

    def pair_poll(self, poll_id: str) -> dict[str, Any] | None:
        resp = self._call("POST", "/runner/v1/pair/poll", json={"poll_id": poll_id})
        if resp.status_code == 202:
            return None
        if resp.status_code != 200:
            raise CentralError(_message(resp))
        return resp.json()

    # jobs
    def claim(self, capabilities: dict[str, Any], tool_versions: dict[str, str]) -> dict[str, Any] | None:
        resp = self._call("POST", "/runner/v1/claim", timeout=90, json={
            "capabilities": capabilities, "runner_version": __version__, "tool_versions": tool_versions,
            "holding": []})  # one job at a time: asking for work = no job in hand (a restart left none)
        self.settings_pending = resp.headers.get("dwyt-settings") == "pending"
        self.languages = resp.headers.get("dwyt-languages")  # the owner's languages (round 9)
        if resp.status_code == 204:
            return None
        if resp.status_code != 200:
            raise CentralError(_("Munkakérés: HTTP {status_code}: {_message}").format(
                status_code=resp.status_code, _message=_message(resp)))
        return resp.json()

    def heartbeat(self, job_id: int, stage: str | None = None, progress: dict | None = None) -> dict[str, Any]:
        resp = self._call("POST", f"/runner/v1/jobs/{job_id}/heartbeat", json={"stage": stage, "progress": progress})
        if resp.status_code != 200:
            raise CentralError(_message(resp))
        return resp.json()

    def partial(self, job_id: int, text: str, thinking: bool = False) -> dict[str, Any] | None:
        """The answer of a question as written so far (19a item 8). None: the central does not take it (an older
        version, or the job is no longer there; the heartbeat finds that out); {"cancel": true}: stop."""
        try:
            resp = self.http.request("PUT", f"/runner/v1/jobs/{job_id}/partial",
                                     json={"text": text[:PARTIAL_CHARS], "thinking": thinking}, timeout=15)
        except httpx.HTTPError as exc:
            raise CentralError(_("A központ nem érhető el ({server}): {exc}").format(
                server=self.server, exc=exc)) from exc
        if resp.status_code == 401:
            raise Unauthorized(_message(resp))
        if resp.status_code != 200:
            return None
        return resp.json()

    def originals(self, body: dict[str, Any]) -> bool:
        """The kept originals (plan 19c); False: an older central that does not know them."""
        resp = self._call("PUT", "/runner/v1/originals", json=body, timeout=30)
        if resp.status_code == 404:
            return False
        if resp.status_code != 200:
            raise CentralError(_message(resp))
        return True

    def result(self, job_id: int, result: dict[str, Any]) -> dict[str, Any]:
        resp = self._call("PUT", f"/runner/v1/jobs/{job_id}/result", json=result, timeout=300)
        if resp.status_code != 200:
            raise CentralError(_("Az eredményt a központ nem fogadta el (HTTP {status_code}): "
                                  "{_message}").format(
                                 status_code=resp.status_code, _message=_message(resp)))
        return resp.json()

    def fail(self, job_id: int, error: str, retry: str) -> None:
        resp = self._call("POST", f"/runner/v1/jobs/{job_id}/fail", json={"error": error[:4000], "retry": retry})
        if resp.status_code != 200:
            raise CentralError(_message(resp))


    # remote settings (round 7d)
    def report_settings(self, body: dict[str, Any]) -> int:
        """How many changes wait; 0 when the central does not know remote settings yet (older version)."""
        resp = self._call("PUT", "/runner/v1/settings", json=body)
        if resp.status_code in (404, 405):
            return 0
        if resp.status_code != 200:
            raise CentralError(_("A beállítások jelentése: HTTP {status_code}: {_message}").format(
                status_code=resp.status_code, _message=_message(resp)))
        return int(resp.json().get("pending") or 0)

    def setting_changes(self) -> list[dict[str, Any]]:
        resp = self._call("GET", "/runner/v1/settings/changes")
        if resp.status_code != 200:
            raise CentralError(_("A beállítás-változások lekérése: HTTP {status_code}: {_message}").format(
                status_code=resp.status_code, _message=_message(resp)))
        return list(resp.json().get("changes") or [])

    def setting_result(self, change_id: int, result: dict[str, Any]) -> None:
        resp = self._call("POST", f"/runner/v1/settings/changes/{change_id}/result", json=result)
        if resp.status_code not in (200, 404):
            raise CentralError(_("A beállítás eredménye: HTTP {status_code}: {_message}").format(
                status_code=resp.status_code, _message=_message(resp)))

    # pairing by install code (the one-line installer redeems it; this is for tests and other installers)
    def redeem(self, code: str, name: str) -> dict[str, Any]:
        resp = self._call("POST", "/runner/v1/pair/redeem", json={"code": code, "name": name})
        if resp.status_code != 200:
            raise CentralError(_message(resp))
        return resp.json()

    # own recordings
    def download(self, path: str, dest: str, max_bytes: int, attempts: int = 4) -> int:
        """Stream a job's uploaded file to dest; a broken transfer resumes with an HTTP Range request."""
        done = os.path.getsize(dest) if os.path.exists(dest) else 0
        for attempt in range(attempts):
            headers = {"Range": f"bytes={done}-"} if done else {}
            try:
                with self.http.stream("GET", path, headers=headers, timeout=httpx.Timeout(120, connect=15)) as resp:
                    if resp.status_code == 401:
                        raise Unauthorized(_message(resp))
                    if resp.status_code in (404, 409, 410):
                        resp.read()
                        raise JobGone(_message(resp))
                    if resp.status_code not in (200, 206):
                        raise CentralError(_("A fájl letöltése: HTTP {status_code}").format(
                            status_code=resp.status_code))
                    mode = "ab" if resp.status_code == 206 else "wb"
                    if mode == "wb":
                        done = 0
                    with open(dest, mode) as fh:
                        for chunk in resp.iter_bytes(1024 * 1024):
                            done += len(chunk)
                            if done > max_bytes:
                                raise CentralError(_("A fájl nagyobb, mint amennyit ez a futtató elfogad "
                                                     "(max_file_mb)."))
                            fh.write(chunk)
                return done
            except httpx.HTTPError as exc:
                if attempt == attempts - 1:
                    raise CentralError(_("A fájl letöltése megszakadt: {exc}").format(exc=exc)) from exc
                time.sleep(2 * (attempt + 1))
        return done

    def local_item(self, sha256: str, name: str, size: int, options: dict[str, Any], summarize: bool) -> dict[str, Any]:
        resp = self._call("POST", "/runner/v1/local-items", json={"sha256": sha256, "name": name, "size": size,
                                                                  "options": options, "summarize": summarize})
        if resp.status_code != 200:
            raise RefusedItem(f"HTTP {resp.status_code}: {_message(resp)}")
        return resp.json()


class RefusedItem(Exception):
    """The central service did not accept a watched-folder file (bad options, quota...)."""


def _message(resp: httpx.Response) -> str:
    try:
        return str(resp.json().get("error") or resp.text)[:500]
    except ValueError:
        return resp.text[:500]
