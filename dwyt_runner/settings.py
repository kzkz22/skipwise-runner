"""The runner's settings described once (plan 15b, decision 18.3): field, kind, allowed values, plain
Hungarian label and explanation. The desktop window (7c) and the dwyt Settings page (7d) draw from this,
and `config set` checks values against it. Every field of config.Config is here exactly once (a test
makes sure); `remote: False` fields are never set from the dwyt website (the pairing, local paths of the
machine's own programs), secrets are write-only everywhere.
"""
from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields
from typing import Any

from .config import Config
from .i18n import N_
from .i18n import gettext as _

GROUPS = [
    ("connection", N_("Kapcsolat a Skipwise-zal")),
    ("cloud", N_("Felhős átírás")),
    ("local", N_("Helyi átírás ezen a gépen")),
    ("llm", N_("Összefoglaló, fordítás, kérdések")),
    ("sources", N_("Források és saját felvételek")),
    ("advanced", N_("Haladó")),
]
TRUE = ("1", "true", "yes", "on", "igen", "be")
FALSE = ("0", "false", "no", "off", "nem", "ki", "")


@dataclass(frozen=True)
class Setting:
    key: str
    kind: str              # text | secret | bool | int | float | choice | list | map | path | url
    group: str
    label: str
    help: str
    choices: tuple[tuple[str, str], ...] = ()   # (value, label)
    min: float | None = None
    max: float | None = None
    unit: str = ""
    advanced: bool = False
    show_if: tuple[tuple[str, Any], ...] = ()   # shown only when these settings have these values
    remote: bool = True    # may be set from the dwyt website (7d)
    readonly: bool = False  # set by pairing, not by hand


S = Setting
SETTINGS: tuple[Setting, ...] = (
    # -- connection --
    S("server", "url", "connection", N_("A Skipwise címe"), N_("Ahová a futtató munkáért fordul. A párosítás tölti ki."),
      remote=False, readonly=True),
    S("token", "secret", "connection", N_("Párosítási token"), N_("A párosításkor kapott belépő; a Skipwise Futtatók oldalán "
      "visszavonható."), remote=False, readonly=True),
    S("runner_id", "int", "connection", N_("Futtató azonosító"), N_("A párosítás tölti ki."), remote=False, readonly=True),
    S("name", "text", "connection", N_("A futtató neve"), N_("Így látszik a Skipwise Futtatók oldalán, "
      "például „otthoni gép”.")),
    S("remote_settings", "choice", "connection", N_("A Skipwise-ból is állítható"),
      N_("A beállítások a Skipwise Futtatók oldaláról, a Beállítás gombbal is módosíthatók; a kulcsokat a böngésző "
      "a futtató saját kulcsával titkosítja. Minden "
      "változás naplóba kerül. Automatikusan: szerveren be, az asztali alkalmazásban ki."),
      choices=(("auto", N_("automatikus")), ("on", N_("be")), ("off", N_("ki"))), remote=False),
    S("ui_language", "choice", "connection", N_("A futtató nyelve"),
      N_("Az ablak, az üzenetek és a hibák nyelve. Automatikusan: a párosított fiók felületi nyelve, párosítás előtt "
         "a rendszeré."),
      choices=(("auto", N_("automatikus")), ("hu", "Magyar"), ("en", "English"), ("de", "Deutsch"),
               ("it", "Italiano"), ("es", "Español"), ("fr", "Français"))),
    # -- cloud transcription --
    S("assemblyai_key", "secret", "cloud", N_("AssemblyAI-kulcs"), N_("A saját AssemblyAI-fiókod kulcsa. Felhős átírás "
      "csak akkor készül, ha valaki kifejezetten azt kéri; a költség a te fiókodat terheli. Javasolt egy külön kulcs "
      "keretkorláttal.")),
    S("assemblyai_url", "url", "cloud", N_("AssemblyAI címe"), N_("Csak ha az EU-s végpontot használod "
      "(https://api.eu.assemblyai.com)."), advanced=True),
    S("cloud_delete_after", "bool", "cloud", N_("Törlés az AssemblyAI-nál"), N_("Az átírás után a hangot és a szöveget "
      "törli az AssemblyAI-nál.")),
    # -- local transcription --
    S("whisper_builtin", "bool", "local", N_("Helyi átírás ezen a gépen"), N_("A whisper.cpp és a beszélők szétválasztása "
      "ezen a gépen fut, a modelleket egyszer tölti le (kb. 3,5 GB). Videókártyával gyors, nélküle lassú.")),
    S("whisper_models", "list", "local", N_("Választható Whisper-modellek"), N_("large-v3: pontosabb; large-v3-turbo: "
      "gyorsabb, kicsit pontatlanabb. Az első az alapértelmezett. Külön GPU-gépnél az ott telepített nevek.")),
    S("diarize_threshold", "float", "local", N_("Beszélők szétválasztása"), N_("Ha a beküldéskor nincs megadva a beszélők "
      "száma, ebből becsli: nagyobb érték kevesebb beszélőt ad. A 0,8 egy 8 beszélős interjún mérve a legjobb."),
      min=0.3, max=1.5, advanced=True, show_if=(("whisper_builtin", True),)),
    S("polite_mode", "bool", "local", N_("Udvarias mód"), N_("Ha valaki épp használja a helyi nyelvi modellt, az átírás "
      "előtt megvárja, hogy egy percig ne használja (legfeljebb a megadott ideig), és csak utána üríti ki a "
      "videókártyáról."), show_if=(("whisper_builtin", True),)),
    S("polite_wait_min", "int", "local", N_("Udvarias mód: leghosszabb várakozás"), N_("Ennyi perc után akkor is kiüríti "
      "a modellt, ha még használják."), min=0, max=120, unit=N_("perc"), show_if=(("polite_mode", True),)),
    S("whisper_cli", "path", "local", N_("A whisper.cpp helye"), N_("Üresen a futtató maga intézi (Windowson letölti). "
      "Megadható a whisper-cli program vagy egy kicsomagolt kiadás mappája."), advanced=True, remote=False),
    S("whisper_threads", "int", "local", N_("Processzorszálak"), N_("0: a whisper.cpp dönt."), min=0, max=256,
      advanced=True),
    S("whisper_url", "url", "local", N_("Külön GPU-gép (AI-router)"), N_("Ha a Whisper egy másik gépen fut (dwyt-airouter "
      "és dwyt-gpu), annak címe; ilyenkor ez a gép nem ír át maga."), advanced=True),
    S("whisper_timeout_min", "int", "local", N_("Egy átírás leghosszabb ideje"), N_("Hosszú felvételnél a hossz négyszerese "
      "számít, ha az több."), min=5, max=1440, unit=N_("perc"), advanced=True),
    # -- language models --
    S("local_llm", "choice", "llm", N_("Helyi modellszerver"), N_("Az összefoglalók ezen a gépen készülnek, a választott "
      "programmal. A futtató elindítja, letölti és betölti a modelleket."),
      choices=(("", N_("nincs")), ("ollama", N_("Ollama")), ("lmstudio", N_("LM Studio")))),
    S("local_llm_url", "url", "llm", N_("A helyi modellszerver címe"), N_("Üresen a szokásos cím ezen a gépen."),
      advanced=True, show_if=(("local_llm", "ollama"), ("local_llm", "lmstudio"))),
    S("local_llm_context", "int", "llm", N_("A helyi modell kontextusa"), N_("Ennyi tokennyi szöveget lát egyszerre a "
      "helyi modell. Az LM Studióban a futtató ezzel tölti be a modellt (ha a memória nem bírja, kevesebbel); "
      "az Ollama ezzel indul, ha a futtató indítja. A hosszabb szöveget a futtató részekre bontja."), min=4096,
      max=262144, unit=N_("token"), advanced=True, show_if=(("local_llm", "ollama"), ("local_llm", "lmstudio"))),
    S("lmstudio_token", "secret", "llm", N_("LM Studio API-token"), N_("Csak ha az LM Studióban bekapcsoltad az "
      "API-tokent."), show_if=(("local_llm", "lmstudio"),)),
    S("llm_prefer_local", "bool", "llm", N_("A helyi modellszerver az elsődleges"), N_("Akkor is a helyi modellszerver "
      "dolgozik, ha felhős cím is meg van adva (a Saját gép mód ezt állítja)."), advanced=True,
      show_if=(("local_llm", "ollama"), ("local_llm", "lmstudio"))),
    S("llm_url", "url", "llm", N_("Felhős vagy saját LLM-szerver címe"), N_("Például https://openrouter.ai/api/v1, vagy a "
      "saját LiteLLM-ed. Az AssemblyAI-kulccsal: https://llm-gateway.assemblyai.com/v1 (ingyenes fiókkal csak a "
      "qwen3.5-4b-32k-fast próba-modell). Ha meg van adva, ez az elsődleges a helyi modellszerverrel szemben.")),
    S("llm_key", "secret", "llm", N_("Az LLM-szerver kulcsa"), N_("Javasolt egy külön kulcs keretkorláttal. Az AssemblyAI "
      "LLM Gateway-nél nem kell: ott az AssemblyAI-kulcs írja az összefoglalót is.")),
    S("llm_reasoning", "choice", "llm", N_("Gondolkodás (felhős modellnél)"), N_("OpenRouter-modelleknél: a modell a "
      "válasz előtt végiggondolja a feladatot. Alaposabb, de lassabb és drágább (a gondolkodás is fizetős token). "
      "Ahol a modell mindig gondolkodik, ott nem kapcsolható ki."),
      choices=(("auto", N_("a modell alapértelmezése")), ("on", N_("be")), ("off", N_("ki")))),
    S("llm_public_model", "text", "llm", N_("Modell a nyilvános videókhoz"), N_("Ezzel készül a YouTube-videók "
      "összefoglalója. Az AssemblyAI-nál feltöltött egyenleggel nagyobb modell is választható (a nevek az AssemblyAI "
      "LLM Gateway modell-listáján).")),
    S("llm_private_model", "text", "llm", N_("Modell a saját felvételekhez"), N_("A saját felvételek szövege csak ehhez a "
      "modellhez kerül; helyi modell javasolt.")),
    S("llm_translate_model", "text", "llm", N_("Modell a fordításhoz"), N_("Üresen a nyilvános videók modellje."),
      advanced=True),
    S("llm_ask_model", "text", "llm", N_("Modell a kérdésekhez"), N_("Üresen a nyilvános videók modellje; egy gondolkodó "
      "modell itt jobb válaszokat ad."), advanced=True),
    S("llm_models", "list", "llm", N_("További választható modellek (nyilvános)"), N_("Ezek közül is választhatsz az elem "
      "oldalán."), advanced=True),
    S("llm_private_models", "list", "llm", N_("További választható modellek (saját felvételek)"), N_("Csak amit ide "
      "felveszel, ahhoz kerülhet saját felvétel szövege."), advanced=True),
    S("llm_model_notes", "map", "llm", N_("Modellek leírása"), N_("Ha az LLM-szerver nem mondja meg, mi fut egy név "
      "mögött: név=leírás."), advanced=True),
    # -- sources --
    S("allowed_domains", "list", "sources", N_("Engedélyezett oldalak"), N_("Honnan tölthet le a futtató (* = bárhonnan).")),
    S("auto_update", "choice", "sources", N_("A futtató frissítése magától"), N_("Naponta megnézi, van-e aláírt újabb "
      "változat, és két munka között telepíti. Automatikusan: az asztali alkalmazásban be; szerveren ki, "
      "ott a Skipwise Futtatók oldala szól, és a Frissítés most gombbal telepíthető."),
      choices=(("auto", N_("automatikus")), ("on", N_("be")), ("off", N_("ki")))),
    S("ytdlp_auto_update", "bool", "sources", N_("A yt-dlp frissítése magától"), N_("A YouTube gyakran változik; a "
      "futtató naponta (és egy elutasított letöltés után) megnézi, van-e újabb yt-dlp, és két munka között "
      "telepíti.")),
    S("cookies_file", "path", "sources", N_("Sütifájl"), N_("Csak ha egy oldal bejelentkezést kér (Netscape-formátumú "
      "cookies.txt)."), advanced=True, remote=False),
    S("watch_dirs", "list", "sources", N_("Figyelt mappák"), N_("Az ide kerülő hang- és videófájlokat a futtató magától "
      "átírja; a fájlok nem hagyják el a gépet.")),
    S("watch_mode", "choice", "sources", N_("Figyelt mappák: átírás módja"), N_("Helyben vagy felhőben."),
      choices=(("local", N_("helyben")), ("cloud", N_("felhőben")))),
    S("watch_model", "text", "sources", N_("Figyelt mappák: modell"), N_("Üresen az alapértelmezett."), advanced=True),
    S("watch_language", "text", "sources", N_("Figyelt mappák: nyelv"), N_("Például hu; üresen felismeri."),
      advanced=True),
    S("watch_summarize", "bool", "sources", N_("Figyelt mappák: összefoglaló"), N_("Az átírás után összefoglaló is "
      "készül.")),
    S("keep_originals", "bool", "sources", N_("Feltöltött felvételek megőrzése"), N_("A böngészőből feltöltött saját "
      "felvételt a futtató az Eredetik mappában megtartja, így újraíráshoz nem kell újra feltölteni. A Skipwise-ban az "
      "elem oldalán vagy a Beállításokban törölhető.")),
    S("originals_max_gb", "int", "sources", N_("Eredetik: legfeljebb"), N_("Fölötte a legrégebben használt eredeti "
      "törlődik."), min=1, max=100000, unit=N_("GB"), show_if=(("keep_originals", True),)),
    S("max_file_mb", "int", "sources", N_("Legnagyobb fájl"), "", min=1, max=100000, unit=N_("MB"), advanced=True),
    S("max_duration_min", "int", "sources", N_("Leghosszabb felvétel"), "", min=1, max=1440, unit=N_("perc"),
      advanced=True),
    # -- advanced --
    S("work_dir", "path", "advanced", N_("Munkamappa"), N_("Ide kerülnek az ideiglenes fájlok; üresen a rendszer "
      "ideiglenes mappája."), advanced=True, remote=False),
    S("secret_store", "choice", "advanced", N_("A kulcsok helye"), N_("Alapból a rendszer kulcstárolója, ha van."),
      choices=(("auto", N_("automatikus")), ("keyring", N_("a rendszer kulcstárolója")), ("file", N_("a beállítási fájl"))),
      advanced=True, remote=False),
)
BY_KEY = {s.key: s for s in SETTINGS}


def defaults() -> dict[str, Any]:
    out = {}
    for f in fields(Config):
        out[f.name] = f.default if f.default is not MISSING else f.default_factory()  # type: ignore[misc]
    return out


def describe() -> dict[str, Any]:
    """Everything a settings screen needs, as plain data (JSON)."""
    base = defaults()
    out = {"groups": []}
    for gid, title in GROUPS:
        items = []
        for s in SETTINGS:
            if s.group != gid:
                continue
            d = asdict(s)
            d.update(label=_(s.label), help=_(s.help), unit=_(s.unit) if s.unit else "")  # round 9
            d["choices"] = [{"value": v, "label": _(label)} for v, label in s.choices]
            d["show_if"] = [{"key": k, "value": v} for k, v in s.show_if]
            d["default"] = "" if s.kind == "secret" else base[s.key]
            items.append(d)
        out["groups"].append({"id": gid, "title": _(title), "settings": items})
    return out


def parse(key: str, raw: str) -> Any:
    """A typed value from text (the command line, a form), checked; ValueError says why in Hungarian."""
    s = BY_KEY.get(key)
    if s is None or s.readonly:
        raise ValueError(_("Ismeretlen vagy nem kézzel állítható beállítás: {key}").format(key=key))
    text = (raw or "").strip()
    if s.kind == "bool":
        low = text.lower()
        if low in TRUE:
            return True
        if low in FALSE:
            return False
        raise ValueError(_("{label}: igen vagy nem").format(label=_(s.label)))
    if s.kind in ("int", "float"):
        try:
            value: Any = int(text) if s.kind == "int" else float(text.replace(",", "."))
        except ValueError:
            raise ValueError(_("{label}: számot várok").format(label=_(s.label))) from None
        if (s.min is not None and value < s.min) or (s.max is not None and value > s.max):
            raise ValueError(_("{label}: {min:g} és {max:g} között").format(label=_(s.label), min=s.min, max=s.max))
        return value
    if s.kind == "choice":
        allowed = [v for v, _ in s.choices]
        if text not in allowed:
            raise ValueError(_("{label}: ezek egyike: {values}").format(
                label=_(s.label), values=", ".join(v or _("(üres)") for v in allowed)))
        return text
    if s.kind == "list":
        items = [v.strip() for v in text.split(",") if v.strip()]
        if s.choices:
            bad = [v for v in items if v not in dict(s.choices)]
            if bad:
                raise ValueError(_("{label}: ismeretlen érték: {join}").format(label=_(s.label), join=', '.join(bad)))
        return items
    if s.kind == "map":
        pairs = (part.split("=", 1) for part in text.split(";") if "=" in part)
        return {k.strip(): v.strip() for k, v in pairs if k.strip() and v.strip()}
    if s.kind == "url" and text and not text.lower().startswith(("http://", "https://")):
        raise ValueError(_("{label}: http:// vagy https:// kezdetű cím").format(label=_(s.label)))
    return text


def summary() -> str:
    """The settings as text, group by group (skipwise-runner config describe)."""
    lines = []
    for g in describe()["groups"]:
        lines += [g["title"], ""]
        for s in g["settings"]:
            if s["readonly"]:
                continue
            extra = []
            if s["choices"]:
                extra.append(" | ".join(c["value"] or _("(üres)") for c in s["choices"]))
            if s["unit"]:
                extra.append(s["unit"])
            lines.append(f"  {s['key']}{' (' + '; '.join(extra) + ')' if extra else ''}")
            lines.append(f"      {s['label']}." + (f" {s['help']}" if s["help"] else ""))
        lines.append("")
    return "\n".join(lines)
