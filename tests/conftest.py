"""Fakes for the runner tests: the central service, yt-dlp (tests/fake_modules), AssemblyAI and an
OpenAI-compatible LLM server. Everything runs on localhost; no network, any OS."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

HERE = Path(__file__).resolve().parent
os.environ.setdefault("DWYT_RUNNER_LANG", "hu")  # the tests read the Hungarian texts (round 9)
sys.path.insert(0, str(HERE / "fake_modules"))  # the fake yt_dlp shadows the real one
sys.path.insert(0, str(HERE.parent))

import yt_dlp  # noqa: E402  (the fake)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, app: FastAPI) -> None:
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self._server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning"))
        threading.Thread(target=self._server.run, daemon=True).start()
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.1).close()
                break
            except OSError:
                time.sleep(0.05)

    def stop(self) -> None:
        self._server.should_exit = True


class FakeCentral:
    def __init__(self) -> None:
        self.reset()
        app = FastAPI()
        fake = self

        def check(request: Request, auth: bool = True):
            if request.headers.get("dwyt-protocol") != "1":
                return JSONResponse({"error": "frissítsd"}, status_code=426)
            if auth and request.headers.get("authorization") != "Bearer tok-1":
                return JSONResponse({"error": "bad token"}, status_code=401)
            return None

        @app.post("/runner/v1/pair/start")
        async def pair_start(request: Request):
            return check(request, auth=False) or {"code": "ABC-234", "poll_id": "poll-123456789", "expires_in": 600}

        @app.post("/runner/v1/pair/poll")
        async def pair_poll(request: Request):
            if fake.approved:
                return {"runner_id": 7, "name": "teszt gép", "token": "tok-1"}
            return JSONResponse({"status": "pending"}, status_code=202)

        @app.post("/runner/v1/claim")
        async def claim(request: Request):
            bad = check(request)
            if bad:
                return bad
            body = await request.json()
            fake.claims.append(body)
            headers = {"Dwyt-Settings": "pending"} if fake.changes else {}
            if fake.languages:
                headers["Dwyt-Languages"] = fake.languages
            if not fake.queue:
                return Response(status_code=204, headers=headers)
            return JSONResponse(fake.queue.pop(0), headers=headers)

        @app.put("/runner/v1/settings")
        async def settings_report(request: Request):
            bad = check(request)
            if bad:
                return bad
            if fake.no_settings:
                return JSONResponse({"detail": "Not Found"}, status_code=404)
            fake.reports.append(await request.json())
            return {"pending": len(fake.changes)}

        @app.get("/runner/v1/settings/changes")
        async def settings_changes(request: Request):
            bad = check(request)
            if bad:
                return bad
            out, fake.changes = fake.changes, []
            return {"changes": out}

        @app.post("/runner/v1/settings/changes/{change_id}/result")
        async def settings_result(change_id: int, request: Request):
            bad = check(request)
            if bad:
                return bad
            fake.change_results[change_id] = await request.json()
            return {"ok": True}

        @app.post("/runner/v1/jobs/{job_id}/heartbeat")
        async def heartbeat(job_id: int, request: Request):
            bad = check(request)
            if bad:
                return bad
            body = await request.json()
            fake.beats.append((job_id, body.get("stage")))
            if body.get("progress"):
                fake.progress.append((job_id, body["progress"]))
            if job_id not in fake.known:
                return JSONResponse({"error": "no such job"}, status_code=404)
            return {"lease_s": 900, "cancel": job_id in fake.cancel}

        @app.put("/runner/v1/jobs/{job_id}/partial")
        async def partial(job_id: int, request: Request):
            bad = check(request)
            if bad:
                return bad
            if fake.no_partial:  # an older central: no such route
                return JSONResponse({"detail": "Not Found"}, status_code=404)
            fake.partials.append((job_id, await request.json()))
            return {"ok": True, "cancel": job_id in fake.stop_on_partial}

        @app.put("/runner/v1/jobs/{job_id}/result")
        async def result(job_id: int, request: Request):
            bad = check(request)
            if bad:
                return bad
            if fake.result_failures:
                fake.result_failures -= 1
                return JSONResponse({"error": "busy"}, status_code=503)
            fake.results[job_id] = await request.json()
            return {"ok": True}

        @app.get("/runner/v1/jobs/{job_id}/media")
        async def media(job_id: int, request: Request):
            bad = check(request)
            if bad:
                return bad
            data = fake.files.get(job_id)
            if data is None:
                return JSONResponse({"error": "gone"}, status_code=410)
            rng = request.headers.get("range")
            if rng:
                start = int(rng.split("=")[1].split("-")[0])
                return Response(data[start:], status_code=206, media_type="application/octet-stream")
            if fake.cut_once:  # a broken transfer: half the file, then the runner resumes with Range
                fake.cut_once = False
                from fastapi.responses import StreamingResponse

                async def half():
                    yield data[: len(data) // 2]
                    raise RuntimeError("connection lost")
                return StreamingResponse(half(), media_type="application/octet-stream",
                                         headers={"content-length": str(len(data))})
            return Response(data, media_type="application/octet-stream")

        @app.post("/runner/v1/local-items")
        async def local_items(request: Request):
            bad = check(request)
            if bad:
                return bad
            body = await request.json()
            fake.registered.append(body)
            if body["options"].get("mode") not in ("local", "cloud"):
                return JSONResponse({"error": "rossz mód"}, status_code=422)
            job_id = 100 + len(fake.registered)
            fake.add_job(job_id, "transcribe", {"source": {"kind": "local", "sha256": body["sha256"],
                                                           "name": body["name"], "size": body["size"]},
                                                "options": body["options"]})
            return {"item_id": 50, "job_id": job_id, "existing": False}

        @app.post("/runner/v1/jobs/{job_id}/fail")
        async def fail(job_id: int, request: Request):
            bad = check(request)
            if bad:
                return bad
            fake.failures[job_id] = await request.json()
            return {"ok": True}

        self.server = Server(app)
        self.url = self.server.url

    def reset(self) -> None:
        self.queue, self.claims, self.beats, self.progress = [], [], [], []
        self.results, self.failures, self.known, self.cancel = {}, {}, set(), set()
        self.approved, self.result_failures = False, 0
        self.files, self.registered, self.cut_once = {}, [], False
        self.reports, self.changes, self.change_results, self.no_settings = [], [], {}, False
        self.partials, self.no_partial, self.stop_on_partial = [], False, set()
        self.languages = None  # the Dwyt-Languages claim header (round 9)

    def add_job(self, job_id: int, kind: str, data: dict) -> None:
        self.known.add(job_id)
        self.queue.append({"job_id": job_id, "kind": kind, "lease_s": 900, "input": data})


class FakeAssembly:
    def __init__(self) -> None:
        self.reset()
        app = FastAPI()
        fake = self

        @app.post("/v2/upload")
        async def upload(request: Request):
            if request.headers.get("authorization") != "aai-key":
                return JSONResponse({"error": "Invalid API key"}, status_code=401)
            fake.uploaded = len(await request.body())
            return {"upload_url": "https://cdn.example/u1"}

        @app.get("/v2/transcript")
        async def listing(request: Request):  # the key check
            if request.headers.get("authorization") != "aai-key":
                return JSONResponse({"error": "Invalid API key"}, status_code=401)
            return {"transcripts": []}

        @app.post("/v2/transcript")
        async def create(request: Request):
            fake.requests.append(await request.json())
            return {"id": "t1", "status": "queued"}

        @app.get("/v2/transcript/{tid}")
        async def get(tid: str):
            return {"id": tid, "status": "completed", "language_code": "hu", "audio_duration": 12,
                    "speech_model_used": "universal-2", "utterances": [
                        {"speaker": "A", "words": [{"text": "Szia", "start": 0, "end": 400},
                                                   {"text": "mindenkinek.", "start": 400, "end": 1200}]},
                        {"speaker": "B", "words": [{"text": "Helló!", "start": 1500, "end": 2000}]}]}

        @app.delete("/v2/transcript/{tid}")
        async def delete(tid: str):
            fake.deleted.append(tid)
            return {}

        self.server = Server(app)
        self.url = self.server.url

    def reset(self) -> None:
        self.requests, self.deleted, self.uploaded = [], [], 0


class FakeLlm:
    def __init__(self) -> None:
        self.reset()
        app = FastAPI()
        fake = self

        @app.get("/v1/model/info")
        async def model_info(request: Request):
            fake.info_calls += 1
            if fake.info is None:
                return JSONResponse({"detail": "Not Found"}, status_code=404)
            return {"data": fake.info}

        @app.get("/v1/models")
        async def models(request: Request):  # the key check
            if request.headers.get("authorization") != "Bearer llm-key":
                return JSONResponse({"error": {"message": "invalid key"}}, status_code=401)
            return {"data": [{"id": "osszefoglalo"}]}

        @app.post("/v1/chat/completions")
        async def chat(request: Request):
            body = await request.json()
            fake.calls.append(body)
            fake.auths.append(request.headers.get("authorization"))
            if fake.limited:  # "too many requests" this many more times
                fake.limited -= 1
                return JSONResponse({"message": "too many requests for this action", "code": 429}, status_code=429,
                                    headers={"Retry-After": "1"})
            if fake.access_denied:
                return JSONResponse({"metadata": {"errors": ["Your account does not have access to this LLM Gateway "
                                                             "model"]}, "message": "invalid request body"},
                                    status_code=400)
            if fake.status != 200:
                return JSONResponse({"error": {"message": fake.message}}, status_code=fake.status)
            result = answer_for(body)
            if not (body.get("stream") and fake.stream):
                return result
            return StreamingResponse(sse(result), media_type="text/event-stream")

        def answer_for(body):
            system = body["messages"][0]["content"]
            if system.startswith(("Fordító vagy", "You are a translator")):
                import re as _re
                user = body["messages"][1]["content"]
                marker = "Fordítandó sorok:\n" if system.startswith("Fordító") else "Lines to translate:\n"
                rows = _re.findall(r"^(\d+)\| (.*)$", user.split(marker, 1)[1], _re.M)
                if fake.drop_once and len(rows) > 1:  # leave one line out: the runner must ask again
                    rows, fake.drop_once = rows[:-1], False
                answer = "\n".join(f"{i}| HU {t}" for i, t in rows) + "\nMAGYARÁZOTT: GPU (grafikus processzor)"
                return {"model": body["model"], "choices": [{"message": {"content": answer}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
            if system.startswith("Egy videó- és hangfelvétel-könyvtár"):
                answer = f"<think>hmm</think>A kertről szól [00:00]. ({len(body['messages']) - 1} üzenet)"
                return {"model": body["model"], "choices": [{"message": {"content": answer}}],
                        "usage": {"prompt_tokens": 70, "completion_tokens": 9,
                                  "completion_tokens_details": {"reasoning_tokens": 30}}}
            if system.startswith("Videókat sorolsz"):  # categories (19.13): scripted answers, else the first id
                import re as _re
                first = _re.search(r"^(\d+): ", body["messages"][1]["content"], _re.M).group(1)
                answer = (fake.category_answers.pop(0) if fake.category_answers
                          else f'```json\n{{"main": {first}, "side": [], "suggestion": null}}\n```')
                return {"model": body["model"], "choices": [{"message": {"content": answer}}],
                        "usage": {"prompt_tokens": 20, "completion_tokens": 4}}
            if system.startswith("Fejezetekre bontasz"):  # chapters (19.21): scripted, else every 4th listed line
                import re as _re
                stamps = _re.findall(r"^\[([\d:]+)\] ", body["messages"][1]["content"], _re.M)
                answer = (fake.chapter_answers.pop(0) if fake.chapter_answers else '{"chapters": [' + ", ".join(
                    f'{{"start": "{st}", "title": "Rész {n + 1}"}}' for n, st in enumerate(stamps[::4])) + "]}")
                return {"model": body["model"], "choices": [{"message": {"content": answer}}],
                        "usage": {"prompt_tokens": 30, "completion_tokens": 6}}
            if system.startswith(("Megbeszélésről vagy előadásról", "You write the minutes")):  # minutes (19c)
                import re as _re
                user = body["messages"][-1]["content"]
                stamps = _re.findall(r"^\[([\d:]+)\] ", user, _re.M) or ["00:00"]
                answer = (fake.minutes_answers.pop(0) if fake.minutes_answers else json.dumps({
                    "takeaways": [{"text": f"Tanulság {stamps[0]}", "at": stamps[0]}],
                    "decisions": [{"text": "Marad a régi rend.", "by": "nincs megadva", "at": stamps[-1]}],
                    "tasks": [{"task": "Ajánlatot kér.", "owner": "Szereplő 2", "due": "péntek", "at": stamps[0]}],
                    "next_steps": [], "open_questions": [{"text": "Mennyibe kerül?", "to": None, "at": None}]},
                    ensure_ascii=False))
                return {"model": body["model"], "choices": [{"message": {"content": answer}}],
                        "usage": {"prompt_tokens": 40, "completion_tokens": 8}}
            answer = "<think>hmm</think>## Összefoglaló\nKert."
            return {"model": body["model"], "choices": [{"message": {"content": answer}}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 5}}

        async def sse(result):
            """The answer as an OpenAI-style stream, a few characters per piece."""
            import asyncio
            import json as _json
            yield ": OPENROUTER PROCESSING\n\n"
            if fake.stall_s:
                await asyncio.sleep(fake.stall_s)
            text = result["choices"][0]["message"]["content"]
            pieces = [text[i:i + 7] for i in range(0, len(text), 7)]
            for n, piece in enumerate(pieces):
                if fake.cut and n == len(pieces) // 2:
                    return  # the connection ends in the middle: no finish_reason, no [DONE]
                chunk = {"model": result["model"], "choices": [{"index": 0, "delta": {"content": piece}}]}
                yield f"data: {_json.dumps(chunk, ensure_ascii=False)}\n\n"
                if fake.piece_delay:
                    await asyncio.sleep(fake.piece_delay)
            done = {"model": result["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {_json.dumps(done)}\n\n"
            yield f"data: {_json.dumps({'model': result['model'], 'choices': [], 'usage': result['usage']})}\n\n"
            yield "data: [DONE]\n\n"

        self.server = Server(app)
        self.url = self.server.url

    def reset(self) -> None:
        self.calls, self.status, self.message, self.drop_once = [], 200, "down", False
        self.auths, self.limited, self.access_denied = [], 0, False
        self.stream, self.stall_s, self.cut, self.piece_delay = True, 0.0, False, 0.0
        self.info, self.info_calls = None, 0  # /model/info rows (LiteLLM); None = not a LiteLLM
        self.category_answers: list[str] = []
        self.minutes_answers: list[str] = []
        self.chapter_answers: list[str] = []


class FakeWhisper:
    """The Whisper worker behind llama-swap: NDJSON progress lines, then a result or an error."""

    def __init__(self) -> None:
        self.reset()
        app = FastAPI()
        fake = self

        @app.post("/upstream/whisper/transcribe")
        async def transcribe(request: Request):
            body = await request.body()
            fake.requests.append({"query": dict(request.query_params), "size": len(body)})
            if fake.status != 200:
                return JSONResponse({"detail": fake.detail}, status_code=fake.status)

            async def lines():
                import asyncio
                import json as _json
                for stage in ("waiting-for-gpu", "transcribe", "diarize"):
                    yield _json.dumps({"stage": stage, "device": "cuda"}) + "\n"
                    await asyncio.sleep(fake.delay)
                if fake.error:
                    yield _json.dumps({"error": fake.error}) + "\n"
                elif not fake.cut:
                    yield _json.dumps({"result": {"language": "hu", "device": "cuda", "diarized": True,
                                                  "warnings": [], "segments": [
                        {"start": 0.5, "end": 2.0, "text": " Jó napot. ", "speaker": "SPEAKER_01"},
                        {"start": 2.2, "end": 4.0, "text": "Köszönöm.", "speaker": "SPEAKER_00"}]}}) + "\n"

            return StreamingResponse(lines(), media_type="application/x-ndjson")

        self.server = Server(app)
        self.url = self.server.url + "/upstream/whisper"

    def reset(self) -> None:
        self.requests, self.status, self.detail, self.error, self.cut, self.delay = [], 200, "", "", False, 0.0


@pytest.fixture(scope="session")
def whisper():
    w = FakeWhisper()
    yield w
    w.server.stop()


@pytest.fixture(scope="session")
def central():
    c = FakeCentral()
    yield c
    c.server.stop()


@pytest.fixture(scope="session")
def assembly():
    a = FakeAssembly()
    yield a
    a.server.stop()


@pytest.fixture(scope="session")
def llm():
    m = FakeLlm()
    yield m
    m.server.stop()


@pytest.fixture(scope="session")
def audio_file(tmp_path_factory) -> str:
    from dwyt_runner import tools
    path = tmp_path_factory.mktemp("audio") / "tone.wav"
    subprocess.run([tools.ffmpeg(), "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=1", str(path)], check=True)
    return str(path)


class MemoryKeyring:
    """A key store in memory: the tests never touch the machine's own (Windows Credential Manager on CI)."""

    def __init__(self) -> None:
        import keyring.backend

        class Backend(keyring.backend.KeyringBackend):
            priority = 10
            data: dict = {}

            def get_password(self, service, username):
                return self.data.get((service, username))

            def set_password(self, service, username, password):
                self.data[(service, username)] = password

            def delete_password(self, service, username):
                self.data.pop((service, username), None)
        self.backend = Backend()


KEYRING = MemoryKeyring()


@pytest.fixture(scope="session")
def remote_key_pem() -> bytes:
    """One RSA key for all tests (3072-bit generation is slow)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch, central, assembly, llm, whisper, audio_file, remote_key_pem):
    import keyring

    from dwyt_runner import keystore
    KEYRING.backend.data.clear()
    keyring.set_keyring(KEYRING.backend)
    keystore.reset()
    monkeypatch.setenv("DWYT_RUNNER_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(exist_ok=True)
    (tmp_path / "home" / "remote_key.pem").write_bytes(remote_key_pem)
    from dwyt_runner import remote, updater
    remote.APP = False
    updater.APP, updater.ON_RESTART = False, None
    updater.STATE.update(latest="", checked=time.time(), running=False, error="", message="")  # no GitHub
    remote._latest.update(version="", at=time.time())  # no GitHub in tests
    central.reset()
    whisper.reset()
    assembly.reset()
    llm.reset()
    yt_dlp.reset()
    yt_dlp.REGISTRY["audio"] = audio_file
    from dwyt_runner import modelinfo, worker, ytdlp_update
    ytdlp_update.STATE.update(checked=time.time(), running=False, error="", updated="")  # no PyPI in tests
    ytdlp_update._wanted_soon.clear()
    monkeypatch.setattr(worker, "DELIVERY_DELAYS", (0.05, 0.05, 0.05))
    modelinfo.reset()
    yield


@pytest.fixture
def cfg(central, assembly, llm):
    from dwyt_runner.config import Config
    return Config(server=central.url, token="tok-1", runner_id=7, name="teszt",
                  assemblyai_key="aai-key", assemblyai_url=assembly.url,
                  llm_url=llm.url + "/v1", llm_key="k", llm_public_model="osszefoglalo",
                  llm_private_model="osszefoglalo-helyi")


# ---- the desktop window (used by test_setup and test_gateway) -------------------------------------------------------

@pytest.fixture
def opened(monkeypatch):
    import webbrowser
    urls = []
    monkeypatch.setattr(webbrowser, "open", lambda url, *a, **k: urls.append(url) or True)
    return urls


@pytest.fixture
def setup_window(opened):
    import time

    from fastapi.testclient import TestClient

    from dwyt_runner.app.runtime import Runtime
    from dwyt_runner.app.server import COOKIE, create
    runtime = Runtime(start_worker=False)
    api, state = create(runtime, "testserver")
    client = TestClient(api)
    r = client.get(f"/belepes?t={state.launch_code}", follow_redirects=False)
    assert COOKIE in r.cookies
    for _ in range(100):  # the first machine check
        if not runtime.machine["scanning"]:
            break
        time.sleep(0.05)
    return client, runtime, r.headers["location"]


@pytest.fixture(autouse=True)
def _hungarian_again():
    """A test that renders another language must not leave it set for the next one."""
    yield
    from dwyt_runner import i18n
    i18n.set_language("hu")
