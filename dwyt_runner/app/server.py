"""The desktop window's local web page (plan 15b, round 7c): Állapot, Működés, Gépem, Kulcsok.

Only for this machine's user: it listens on 127.0.0.1, answers only to the Host it was started with
(no DNS rebinding), and lets in only the window that opened it: the start URL carries a one-time code that
becomes an HttpOnly, SameSite=Strict session cookie. Forms carry a CSRF token. No inline script or
style (strict CSP), like the central's pages. Keys can be written and tried here, never read back.
"""
from __future__ import annotations

import secrets
import socket
import threading
import time
import webbrowser
from dataclasses import replace
from urllib.parse import urlencode
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .. import (__version__, autostart, config, gateway, i18n, keystore, localllm, machine, modes, release, remote,
               settings, steps, trial, updater, whispercpp)
from . import wizard
from .runtime import Runtime
from ..i18n import gettext as _
from ..i18n import N_, labels

HERE = Path(__file__).parent
COOKIE = "dwyt_runner_session"
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
       "frame-ancestors 'none'; form-action 'self'; base-uri 'none'")
SECRET_LABELS = labels({"assemblyai_key": N_("AssemblyAI-kulcs (felhős átírás)"),
                        "llm_key": N_("LLM-kulcs (felhős összefoglaló)"), "lmstudio_token": N_("LM Studio API-token")})


class App:
    def __init__(self, runtime: Runtime, host: str) -> None:
        self.runtime = runtime
        self.host = host                       # "127.0.0.1:PORT": the only Host header answered
        self.launch_code = secrets.token_urlsafe(24)
        self.instance_secret = secrets.token_urlsafe(24)  # a second start of the app shows this window
        self.on_show = lambda: None
        self.on_quit = lambda: None
        self.reset_said: list[str] = []   # what the last Alaphelyzet did, shown on the first setup step
        self.sessions: dict[str, str] = {}     # session id -> CSRF token
        self.env = Environment(loader=FileSystemLoader(HERE / "templates"), autoescape=select_autoescape(),
                               extensions=["jinja2.ext.i18n"])
        # the window's texts in the chosen language (round 9); newstyle: the message is markup, %-formatted
        self.env.install_gettext_callables(i18n.gettext, lambda s, p, n: i18n.gettext(s if n == 1 else p),
                                           newstyle=True)
        self.env.globals.update(ui_lang=i18n.current)
        self.env.globals.update(version=__version__, lamp_text=machine.LAMP_TEXT,
                                card_swap=machine.card_swap_servers, seconds=steps.seconds_text)

    @property
    def start_url(self) -> str:
        return f"http://{self.host}/belepes?t={self.launch_code}"

    def new_start_url(self) -> str:
        """A fresh one-time entry (the browser mode's tray "open" opens a new tab with it)."""
        self.launch_code = secrets.token_urlsafe(24)
        return self.start_url

    async def form(self, request: Request):
        """The posted form, after checking its CSRF token against the session's."""
        form = await request.form()
        expected = self.sessions.get(request.cookies.get(COOKIE, ""), "")
        if not expected or not secrets.compare_digest(str(form.get("csrf", "")), expected):
            raise HTTPException(403, _("Lejárt űrlap; töltsd újra az oldalt."))
        return form

    def render(self, request: Request, name: str, **ctx: Any) -> HTMLResponse:
        try:
            i18n.apply(config.load())  # the setting may have just changed (round 9)
        except Exception:  # noqa: BLE001 - an unreadable settings file: keep the language as it is
            pass
        sid = request.cookies.get(COOKIE, "")
        ctx.update(csrf=self.sessions.get(sid, ""), tab=name.split(".")[0], cfg=config.load())
        return HTMLResponse(self.env.get_template(name).render(**ctx))


def create(runtime: Runtime, host: str) -> tuple[FastAPI, App]:
    remote.APP = True  # the desktop app: remote settings only when switched on here (decision 18.11)
    updater.APP = True  # and it updates itself (auto_update auto = on)
    state = App(runtime, host)
    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    api.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @api.middleware("http")
    async def guard(request: Request, call_next):
        if request.headers.get("host") != state.host:
            return Response(_("Ismeretlen cím."), status_code=403)
        path = request.url.path
        if not (path in ("/belepes", "/ablak", "/ablak/kilepes") or path.startswith("/static/")):
            sid = request.cookies.get(COOKIE, "")
            if sid not in state.sessions:
                return Response(_("Ezt az oldalt a Skipwise futtató ablaka nyitja meg."), status_code=403)
        response = await call_next(request)
        response.headers.update({"Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff",
                                 "Referrer-Policy": "no-referrer", "Cache-Control": "no-store"})
        return response

    @api.get("/belepes")
    def enter(t: str = ""):
        if not state.launch_code or not secrets.compare_digest(t, state.launch_code):
            return Response(_("Ez a belépő már nem érvényes; nyisd meg az ablakot újra."), status_code=403)
        state.launch_code = ""  # one time only
        sid = secrets.token_urlsafe(24)
        state.sessions[sid] = secrets.token_urlsafe(24)
        r = RedirectResponse("/telepites/1" if wizard.needed(config.load()) else "/allapot", status_code=303)
        r.set_cookie(COOKIE, sid, httponly=True, samesite="strict")
        return r

    @api.post("/ablak")
    def show_window(request: Request):
        """Another start of the app on this machine: show this window instead of a second app."""
        given = request.headers.get("x-dwyt-instance", "")
        if not secrets.compare_digest(given, state.instance_secret):
            return Response(status_code=403)
        state.on_show()
        return Response(status_code=204)

    @api.post("/ablak/kilepes")
    def quit_from_outside(request: Request):
        """The uninstaller (or `skipwise-runner app-quit`) closes the running app: same secret as /ablak."""
        given = request.headers.get("x-dwyt-instance", "")
        if not secrets.compare_digest(given, state.instance_secret):
            return Response(status_code=403)
        threading.Timer(0.2, state.on_quit).start()
        return Response(status_code=204)

    @api.get("/")
    def root():
        return RedirectResponse("/allapot", status_code=303)

    @api.post("/kilepes")
    async def quit_app(request: Request):
        await state.form(request)
        threading.Timer(0.5, state.on_quit).start()  # the goodbye page gets out first
        return HTMLResponse(_("<!doctype html><html lang=\"hu\"><head><meta charset=\"utf-8\">"
                            "<title>Skipwise futtató</title>"
                            "<link rel=\"stylesheet\" href=\"/static/style.css\"></head><body><main>"
                            "<h1>A futtató leállt</h1><p class=\"muted\">Nem vesz fel munkát, amíg újra nem indítod."
                            "</p></main></body></html>"))

    # -- Állapot --
    @api.get("/allapot")
    def status_page(request: Request):
        return state.render(request, "allapot.html", s=runtime.status())

    @api.get("/api/allapot")
    def status_json():
        return JSONResponse(runtime.status())

    @api.post("/parositas")
    async def pair(request: Request):
        form = await state.form(request)
        try:
            runtime.start_pairing(str(form.get("server", "")), str(form.get("name", "")))
        except Exception as exc:  # noqa: BLE001 - shown on the page
            runtime.pairing = {"state": "error", "error": str(exc)}
        return RedirectResponse("/allapot", status_code=303)

    # -- Működés --
    @api.get("/mukodes")
    def mode_page(request: Request, hiany: str = ""):
        cfg = config.load()
        models_gb = machine.folder_gb(machine.models_dir())
        return state.render(request, "mukodes.html", modes=modes.MODES, mode=modes.current(cfg),
                            flows=modes.preview(cfg), missing=[m for m in hiany.split("|") if m],
                            groups=settings.describe()["groups"], values=config.masked(cfg),
                            login_start=autostart.enabled(), models_gb=models_gb,
                            remote_on=remote.enabled(cfg), paired=cfg.paired, server=cfg.server,
                            models_dir=str(machine.models_dir()),
                            trial_llm=gateway.is_trial(cfg.llm_url, cfg.llm_public_model)
                            and not localllm.uses_local(cfg),
                            weak_language=i18n.weak_local_language(config.home()) if localllm.uses_local(cfg)
                            else None)

    @api.post("/mukodes/osszefoglalo")
    async def summary_provider(request: Request):
        """Who writes the summaries: AssemblyAI's trial model or OpenRouter (19.9). Keys stay as they are."""
        form = await state.form(request)
        cfg = config.load()
        if form.get("provider") == "openrouter":
            cfg.llm_url = wizard.CLOUD_LLM_URL
            cfg.llm_public_model = wizard.cloud_model_default(cfg)
        elif form.get("provider") == "gateway":
            gateway.use(cfg)
        config.save(cfg)
        runtime.settings_changed()
        missing = modes.apply(config.load(), modes.current(cfg)) if modes.current(cfg) != "egyeni" else []
        return RedirectResponse("/mukodes" + ("?hiany=" + "|".join(missing) if missing else ""), status_code=303)

    @api.post("/mukodes")
    async def mode_save(request: Request):
        form = await state.form(request)
        cfg = config.load()
        missing: list[str] = []
        mode = str(form.get("mode", ""))
        if mode in ("felho", "vegyes", "sajat"):
            missing = modes.apply(cfg, mode)
        cfg.polite_mode = form.get("polite_mode") == "on"
        config.save(cfg)
        runtime.settings_changed()
        return RedirectResponse("/mukodes" + ("?hiany=" + "|".join(missing) if missing else ""), status_code=303)

    @api.post("/mukodes/inditas")
    async def login_start_save(request: Request):
        form = await state.form(request)
        try:
            autostart.set_enabled(form.get("on") == "on")
        except OSError as exc:
            return RedirectResponse("/mukodes?hiany=" + _("Az indítást nem sikerült beállítani: "
                                                           "{exc}").format(
                                                          exc=exc), status_code=303)
        return RedirectResponse("/mukodes", status_code=303)

    @api.post("/mukodes/tavoli")
    async def remote_save(request: Request):
        """A Skipwise-ból is állítható (round 7d): on/off here; the change reaches dwyt with the next report."""
        form = await state.form(request)
        cfg = config.load()
        cfg.remote_settings = "on" if form.get("on") == "on" else "off"
        config.save(cfg)
        runtime.settings_changed()
        return RedirectResponse("/mukodes", status_code=303)

    @api.post("/alaphelyzet")
    async def reset_all(request: Request):
        form = await state.form(request)
        if form.get("ertem") != "on":
            return RedirectResponse("/mukodes?hiany=" + _("Az alaphelyzethez pipáld be, hogy érted, mi törlődik."),
                                    status_code=303)
        server = config.load().server
        from starlette.concurrency import run_in_threadpool
        said = await run_in_threadpool(runtime.reset, form.get("models") == "on")
        if server:
            said.append(_("A Skipwise-ban ({server}/futtatok) a régi futtatót vond vissza: onnan ez "
                           "a gép nem tudja törölni magát.").format(
                          server=server))
        state.reset_said = said
        return RedirectResponse("/telepites/1", status_code=303)

    @api.post("/mukodes/reszletek")
    async def details_save(request: Request):
        form = await state.form(request)
        cfg = config.load()
        errors = []
        for s in settings.SETTINGS:
            if s.readonly or s.kind == "secret":
                continue
            if s.kind == "bool":
                raw = "igen" if form.get(s.key) == "on" else "nem"
            elif s.key in form:
                raw = str(form.get(s.key))
                if s.kind == "map":
                    raw = raw.replace("\n", ";")
            else:
                continue
            try:
                setattr(cfg, s.key, settings.parse(s.key, raw))
            except ValueError as exc:
                errors.append(str(exc))
        if errors:
            return RedirectResponse("/mukodes?hiany=" + "|".join(errors), status_code=303)
        config.save(cfg)
        runtime.settings_changed()
        return RedirectResponse("/mukodes", status_code=303)

    # -- Gépem --
    @api.get("/gepem")
    def machine_page(request: Request):
        view = runtime.machine_view()
        m = view["m"]
        if m is None:  # the very first scan is still running
            return state.render(request, "gepem_wait.html")
        last = trial.load()
        card = m.best_gpu.name if m.best_gpu else ""
        last_when = time.strftime("%Y-%m-%d %H:%M", time.localtime(last.at)) if last else ""
        scanned = time.strftime("%H:%M", time.localtime(view["at"]))
        return state.render(request, "gepem.html", m=m, a=machine.assess(m), said=view["said"], last=last,
                            last_when=last_when, same_card=last is not None and last.gpu == card,
                            trial_state=runtime.trial, local_models=view["local_models"], scanned=scanned,
                            scanning=view["scanning"], ytdlp=_ytdlp_view(), upd=_update_view(config.load()))

    def _update_view(cfg) -> dict[str, Any]:
        st = dict(updater.STATE)
        st.update(version=__version__, signed=updater.signed(), kind=updater.kind(), auto=updater.auto(cfg),
                  newer=release.newer(st["latest"]), previous=updater.history().get("previous") or "",
                  checked_text=time.strftime("%Y-%m-%d %H:%M", time.localtime(st["checked"])) if st["checked"] else "")
        st["can_go_back"] = bool(st["previous"]) and st["previous"] != __version__
        return st

    def _ytdlp_view() -> dict[str, Any]:
        from .. import ytdlp_update
        st = dict(ytdlp_update.STATE)
        st["installed"] = st["installed"] or ytdlp_update.installed()
        st["checked_text"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(st["checked"])) if st["checked"] else ""
        st["newer"] = ytdlp_update.newer(st["latest"], st["installed"])
        return st

    @api.post("/gepem/ytdlp")
    async def ytdlp_now(request: Request):
        """Look for a newer yt-dlp now; install it at once when no job runs, else between jobs."""
        from .. import activity, ytdlp_update
        await state.form(request)
        busy = bool(activity.current())
        if busy:
            ytdlp_update._wanted_soon.set()
        threading.Thread(target=ytdlp_update.check, kwargs={"update": not busy}, daemon=True,
                         name="ytdlp").start()
        time.sleep(0.1)  # the page then already shows the search
        return RedirectResponse("/gepem", status_code=303)

    @api.post("/gepem/frissites")
    async def update_self(request: Request):
        """Look for a signed newer version and install it (or go back to the previous one); only when idle."""
        from .. import activity
        form = await state.form(request)
        target = updater.history().get("previous") if form.get("action") == "vissza" else None
        if activity.current():
            updater.STATE["error"] = _("Most munka fut; a frissítés utána indítható (vagy magától megtörténik).")
            return RedirectResponse("/gepem", status_code=303)

        def work() -> None:
            if form.get("action") == "keres":
                updater.check()
                return
            ok, _said, action = updater.update_now(target=target)
            if ok and action:
                updater.finish(action)
        threading.Thread(target=work, daemon=True, name="update").start()
        time.sleep(0.1)
        return RedirectResponse("/gepem", status_code=303)

    @api.post("/gepem/ujra")
    async def rescan(request: Request):
        await state.form(request)
        runtime.refresh_machine()
        return RedirectResponse("/gepem", status_code=303)

    @api.get("/api/gep")
    def machine_json():
        view = runtime.machine_view()
        return JSONResponse({"scanning": view["scanning"], "at": view["at"]})

    @api.post("/gepem/proba")
    async def start_trial(request: Request):
        form = await state.form(request)
        runtime.start_trial(str(form.get("llm", "")))
        return RedirectResponse("/gepem", status_code=303)

    @api.get("/api/proba")
    def trial_json():
        return JSONResponse(runtime.trial)

    # -- Telepítés (the setup steps) --
    def step_page(request: Request, n: int, **ctx: Any) -> HTMLResponse:
        return state.render(request, f"telepites_{n}.html", step=n, steps=wizard.STEPS, **ctx)

    @api.get("/telepites")
    def setup_start():
        return RedirectResponse("/telepites/1", status_code=303)

    @api.get("/telepites/{n}")
    def setup_page(request: Request, n: int, hiba: str = ""):
        cfg = config.load()
        mode = wizard.chosen_mode(cfg)
        if n == 1:
            said, state.reset_said = state.reset_said, []
            return step_page(request, 1, reset_said=said)
        if n == 2:
            view = runtime.machine_view()
            m = view["m"]
            servers = [s for s in m.servers if s.running and s.models] if m else []
            return step_page(request, 2, m=m, a=machine.assess(m) if m else None, last=trial.load(),
                             trial_state=runtime.trial, servers=servers, scanning=view["scanning"])
        if n == 3:
            view = runtime.machine_view()
            last = trial.load()
            recommended = (last.mode if last and last.mode else
                           machine.assess(view["m"]).mode if view["m"] else "vegyes")
            return step_page(request, 3, modes=modes.MODES, mode=wizard.load().get("mode") or recommended,
                             recommended=recommended,
                             texts=wizard.MODE_TEXT,
                             flows=modes.preview(replace(cfg, llm_url=cfg.llm_url or wizard.CLOUD_LLM_URL)))
        if n == 4:
            view = runtime.machine_view()
            m = view["m"]
            items = wizard.ingredients(cfg, mode, runtime.prepare,
                                       wizard.needs_whisper(cfg, mode) and
                                       whispercpp.prepared(cfg, wizard.whisper_model(cfg)))
            servers = m.servers if m else []
            kind = cfg.local_llm or next((s.kind for s in servers if s.running), "") or \
                next((s.kind for s in servers if s.installed), "ollama")
            server_models = next((s.models for s in servers if s.kind == kind), [])
            return step_page(request, 4, mode=mode, mode_label=modes.MODES.get(mode, mode), items=items,
                             go_on=wizard.can_go_on(items), prepare=wizard.prepare_view(runtime.prepare),
                             pull=wizard.pull_view(runtime.pull), checks=wizard.load().get("checks", {}),
                             cloud_model=wizard.cloud_model_default(cfg), servers=servers, kind=kind,
                             provider=wizard.provider(cfg), trial_model=gateway.TRIAL_MODEL,
                             server_models=server_models, error=hiba,
                             starting=runtime.starting,
                             suggested=wizard.suggested_local_model(m.vram_gb if m else 0, kind,
                                                                    (m.ram_gb or 0) if m else 0, server_models))
        if n == 5:
            with runtime.lock:
                pairing = dict(runtime.pairing)
            return step_page(request, 5, pairing=pairing, paired=cfg.paired, error=hiba,
                             default_name=cfg.name or socket.gethostname(),
                             approve_url=_approve_url(pairing) if pairing.get("code") else "")
        if n == 6:
            return step_page(request, 6, login_start=autostart.enabled() or not wizard.load().get("done"))
        return RedirectResponse("/telepites/1", status_code=303)

    def _approve_url(pairing: dict[str, Any]) -> str:
        return f"{pairing['server']}/futtatok?" + urlencode({"kod": pairing["code"], "nev": pairing.get("name", "")})

    @api.get("/api/telepites")
    def setup_json():
        with runtime.lock:
            pairing = {k: runtime.pairing.get(k) for k in ("state", "code")}
        from .. import ytdlp_update
        return JSONResponse({"prepare": wizard.prepare_view(runtime.prepare), "pull": wizard.pull_view(runtime.pull),
                             "ytdlp": {"running": ytdlp_update.STATE["running"], "text": ""},
                             "update": {"running": updater.STATE["running"], "text": ""},
                             "starting": {"running": runtime.starting["running"],
                                          "text": f"{int(time.time() - runtime.starting['since'])} mp"},
                             "trial": runtime.trial, "pairing": pairing})

    @api.post("/telepites/kihagyas")
    async def setup_skip(request: Request):
        await state.form(request)
        wizard.save(done=True, skipped=True)
        return RedirectResponse("/allapot", status_code=303)

    @api.post("/telepites/2")
    async def setup_machine(request: Request):
        form = await state.form(request)
        if form.get("action") == "ujra":
            runtime.refresh_machine()
        else:
            runtime.start_trial(str(form.get("llm", "")), str(form.get("server", "")))
        return RedirectResponse("/telepites/2", status_code=303)

    @api.post("/telepites/3")
    async def setup_mode(request: Request):
        form = await state.form(request)
        mode = str(form.get("mode", ""))
        if mode not in modes.MODES:
            return RedirectResponse("/telepites/3", status_code=303)
        cfg = config.load()
        if mode != "egyeni":
            modes.apply(cfg, mode)
            if mode == "felho" and not cfg.llm_url and not cfg.llm_key:
                gateway.use(cfg)  # a new user: one AssemblyAI registration is enough to try (19.9)
            elif mode in ("felho", "vegyes") and not cfg.llm_url:
                cfg.llm_url = wizard.CLOUD_LLM_URL
            config.save(cfg)
            runtime.settings_changed()
        wizard.save(mode=mode)
        if wizard.needs_whisper(cfg, mode) and not whispercpp.prepared(cfg, wizard.whisper_model(cfg)):
            runtime.start_prepare(wizard.whisper_model(cfg))
        return RedirectResponse("/telepites/4", status_code=303)

    @api.post("/telepites/4")
    async def setup_ingredients(request: Request):
        form = await state.form(request)
        action = str(form.get("action", ""))
        cfg = config.load()
        checks = wizard.load().get("checks", {})
        if action == "llm":  # the OpenRouter form
            key = str(form.get("key", "")).strip()
            if key:
                cfg.llm_key = key
            if not cfg.llm_url or gateway.is_gateway(cfg.llm_url):
                cfg.llm_url = wizard.CLOUD_LLM_URL
            cfg.llm_public_model = str(form.get("model", "")).strip() or wizard.CLOUD_MODEL
            config.save(cfg)
            checks["llm"] = keystore.check_llm(config.load())
        elif action == "assemblyai":
            key = str(form.get("key", "")).strip()
            if key:
                cfg.assemblyai_key = key
                config.save(cfg)
            checks["assemblyai"] = keystore.check_assemblyai(config.load())
            if gateway.is_gateway(cfg.llm_url):  # the same key writes the summaries: try that too
                checks["llm"] = keystore.check_llm(config.load())
        elif action == "provider":
            chosen = str(form.get("provider", ""))
            if chosen == "gateway":
                gateway.use(cfg)
            elif chosen == "openrouter":
                cfg.llm_url = wizard.CLOUD_LLM_URL
                cfg.llm_public_model = wizard.cloud_model_default(cfg)
            config.save(cfg)
            checks.pop("llm", None)
            if chosen == "gateway" and cfg.assemblyai_key:
                checks["llm"] = keystore.check_llm(config.load())
        elif action == "local":
            kind = str(form.get("server", ""))
            name = str(form.get("model", "")).strip()
            if kind in ("ollama", "lmstudio") and name:
                cfg.local_llm = kind
                cfg.llm_private_model = name
                if wizard.chosen_mode(cfg) == "sajat":
                    cfg.llm_public_model = name  # Saját gép: public sources also use the local server
                config.save(cfg)
        elif action == "pull":
            kind = str(form.get("server", ""))
            name = str(form.get("name", "")).strip()
            if kind in ("ollama", "lmstudio") and name:
                runtime.start_pull(kind, name)
        elif action == "start":  # in the background: the page shows that it is starting and follows it
            kind = str(form.get("server", ""))
            if kind in ("ollama", "lmstudio"):
                runtime.start_server(kind)
        elif action == "rescan":
            from starlette.concurrency import run_in_threadpool
            await run_in_threadpool(runtime.rescan_and_wait)
        elif action == "prepare":
            runtime.start_prepare(wizard.whisper_model(cfg))
        wizard.save(checks=checks)
        runtime.settings_changed()
        return RedirectResponse("/telepites/4", status_code=303)

    @api.post("/telepites/5")
    async def setup_pair(request: Request):
        form = await state.form(request)
        try:
            runtime.start_pairing(str(form.get("server", "")), str(form.get("name", "")))
        except Exception as exc:  # noqa: BLE001 - shown on the page
            return RedirectResponse("/telepites/5?" + urlencode({"hiba": str(exc)}), status_code=303)
        with runtime.lock:
            pairing = dict(runtime.pairing)
        try:
            webbrowser.open(_approve_url(pairing))
        except Exception:  # noqa: BLE001 - the page has the link too
            pass
        return RedirectResponse("/telepites/5", status_code=303)

    @api.post("/telepites/6")
    async def setup_done(request: Request):
        form = await state.form(request)
        try:
            autostart.set_enabled(form.get("login_start") == "on")
        except OSError:
            pass
        wizard.save(done=True)
        cfg = config.load()
        if form.get("open_dwyt") == "on" and cfg.server:
            try:
                webbrowser.open(cfg.server.rstrip("/") + "/uj")
            except Exception:  # noqa: BLE001
                pass
        return RedirectResponse("/allapot", status_code=303)

    # -- Kulcsok --
    @api.get("/kulcsok")
    def keys_page(request: Request, proba: str = ""):
        cfg = config.load()
        rows = [{"key": k, "label": label, "set": bool(getattr(cfg, k))} for k, label in SECRET_LABELS.items()]
        checks = {}
        if proba:
            checks = {"assemblyai_key": keystore.check_assemblyai(cfg), "llm_key": keystore.check_llm(cfg)}
        return state.render(request, "kulcsok.html", rows=rows, store=keystore.name(), checks=checks)

    @api.post("/kulcsok")
    async def keys_save(request: Request):
        form = await state.form(request)
        key = str(form.get("key", ""))
        if key not in SECRET_LABELS:
            return Response(_("Ismeretlen kulcs."), status_code=400)
        cfg = config.load()
        setattr(cfg, key, str(form.get("value", "")).strip())
        config.save(cfg)
        runtime.settings_changed()
        return RedirectResponse("/kulcsok?proba=1", status_code=303)

    return api, state
