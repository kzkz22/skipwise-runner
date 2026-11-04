"""Build the desktop packages (round 7c, decision 18.20): the Windows installer and the Linux AppImage.

    python packaging/build.py windows   # on Windows, with Inno Setup (iscc) on PATH or in its usual place
    # signed (the release workflow): windows-folder, windows-tosign, (SignPath), windows-putback DIR,
    # windows-installer --signed-uninstaller, (SignPath for the installer), windows-verify FILES
    python packaging/build.py appimage  # on Linux; downloads appimagetool

Both carry their own Python (the standalone CPython uv manages) with the runner and its [desktop,local]
extras installed into it; Linux also gets Qt for the window (pywebview[qt], decision 18.19). whisper.cpp
is not inside: the runner downloads the official release on first use, SHA-256 checked (decision 18.18).
The Windows package carries Tesseract for the slides (plan 19c round C); Linux uses the system's.
The package kind is written into the bundled Python (dwyt-runner-package.json), so the runner knows how
to update itself. Output: dist/dwyt-runner-setup-<version>.exe or dist/dwyt-runner-<version>-x86_64.AppImage.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "package"
DIST = ROOT / "dist"
PYTHON = "3.13"
APPIMAGETOOL = ("https://github.com/AppImage/appimagetool/releases/download/continuous/"
                "appimagetool-x86_64.AppImage")


def run(*cmd: str, **kw) -> str:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, **kw).stdout.strip()


def version() -> str:
    for line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("version = "):
            return line.split('"')[1]
    raise SystemExit("no version in pyproject.toml")


def wheel() -> Path:
    found = sorted(DIST.glob(f"dwyt_runner-{version()}-py3-none-any.whl"))
    if not found:
        run("uv", "build", "--wheel", "--out-dir", str(DIST), cwd=ROOT)
        found = sorted(DIST.glob(f"dwyt_runner-{version()}-py3-none-any.whl"))
    return found[-1]


def bundled_python(target: Path) -> Path:
    """Copy uv's standalone CPython into target; returns its python executable."""
    run("uv", "python", "install", PYTHON)
    exe = Path(run("uv", "python", "find", "--managed-python", PYTHON))
    home = exe.parent if os.name == "nt" else exe.parent.parent
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(home, target, symlinks=True)
    for marker in target.rglob("EXTERNALLY-MANAGED"):  # it is ours now: packages go into it
        marker.unlink()
    return target / ("python.exe" if os.name == "nt" else "bin/python3")


def install_runner(python: Path, kind: str, extra: list[str]) -> None:
    whl = wheel()
    run("uv", "pip", "install", "--python", str(python), f"dwyt-runner[desktop,local] @ {whl.as_uri()}", *extra)
    # console scripts hold the build machine's absolute path: the package starts with `python -m` instead
    scripts = python.parent / "Scripts" if os.name == "nt" else python.parent
    for f in scripts.glob("*"):
        if f.name.startswith(("dwyt-runner", "yt-dlp", "deno", "keyring", "normalizer", "httpx", "pywebview")):
            f.unlink()
    prefix = run(str(python), "-c", "import sys; print(sys.prefix)")
    (Path(prefix) / "dwyt-runner-package.json").write_text(json.dumps({"kind": kind, "version": version()}),
                                                           encoding="utf-8")
    print(run(str(python), "-m", "dwyt_runner.cli", "--version"))


ICON_SCRIPT = """
import sys
from pathlib import Path
from dwyt_runner.app.logo import write_icons
write_icons(Path(sys.argv[1]))
"""


def icons(python: Path, out: Path) -> None:
    """The dwyt logo as .png and .ico (Pillow is in the bundled Python with the desktop extra)."""
    run(str(python), "-c", ICON_SCRIPT, str(out))


RCEDIT = "rcedit@4.0.1"  # npm package carrying electron's rcedit.exe (Windows resource editor), pinned
LAUNCHER = "Skipwise.exe"


def launcher(python: Path, icon: Path) -> Path:
    """python\\Skipwise.exe: pythonw.exe under our name, with our description and icon, so Task Manager, the
    taskbar and Windows' "open with" lists say "Skipwise futtató" instead of Python. Same folder as python.exe,
    so Python finds its home and DLLs exactly as for pythonw."""
    exe = python.with_name(LAUNCHER)
    shutil.copy(python.with_name("pythonw.exe"), exe)
    tools = BUILD / "rcedit"
    if not (tools / "node_modules" / "rcedit").is_dir():
        run(shutil.which("npm") or "npm", "install", "--prefix", str(tools), "--no-audit", "--no-fund", RCEDIT)
    script = tools / "brand.js"
    script.write_text(
        "const rcedit = require('rcedit');\n"
        "const [exe, icon, version] = process.argv.slice(2);\n"
        "rcedit(exe, {icon, 'file-version': version, 'product-version': version, 'version-string': {\n"
        "  FileDescription: 'Skipwise futtató', ProductName: 'Skipwise futtató', CompanyName: 'Skipwise',\n"
        "  InternalName: 'Skipwise', OriginalFilename: 'Skipwise.exe', LegalCopyright: 'MIT'}})\n"
        "  .then(() => console.log('branded', exe), e => { console.error(e); process.exit(1); });\n",
        encoding="utf-8")
    run("node", str(script), str(exe), str(icon), version(), cwd=tools)
    probe = python.parent / "launcher-probe.txt"  # it must still be Python: it writes this file, then it goes
    subprocess.run([str(exe), "-c", f"open(r'{probe}', 'w').write('ok')"], check=True, timeout=60)
    if probe.read_text() != "ok":
        raise SystemExit(f"{LAUNCHER} does not run Python")
    probe.unlink()
    return exe


def iscc() -> str:
    found = shutil.which("iscc")
    if found:
        return found
    for base in (os.environ.get("ProgramFiles(x86)", ""), os.environ.get("ProgramFiles", ""),
                 os.path.expandvars(r"%LOCALAPPDATA%\Programs")):
        p = Path(base) / "Inno Setup 6" / "ISCC.exe"
        if p.is_file():
            return str(p)
    raise SystemExit("Inno Setup (ISCC.exe) not found")


WIN_SRC = BUILD / "windows"
ISS = ROOT / "packaging" / "windows" / "dwyt-runner.iss"


# Tesseract OCR for the slides (plan 19c round C): UB Mannheim's Windows build (Apache-2.0; the MinGW runtime and
# libraries keep their own licences, see tesseract/README.txt), taken out of its NSIS installer with 7-Zip: the
# program and the DLLs it loads, debug info stripped (libtesseract: 101 MB -> 4 MB). The language data is not
# inside: the runner fetches tessdata_fast files when needed, SHA-256 checked (slides.TESSDATA_SHA256).
TESSERACT_URL = ("https://github.com/UB-Mannheim/tesseract/releases/download/v5.4.0.20240606/"
                 "tesseract-ocr-w64-setup-5.4.0.20240606.exe")
TESSERACT_SHA256 = "c885fff6998e0608ba4bb8ab51436e1c6775c2bafc2559a19b423e18678b60c9"
TESSERACT_FILES = (
    "tesseract.exe", "libtesseract-5.dll", "libleptonica-6.dll", "libLerc.dll", "libarchive-13.dll", "libb2-1.dll",
    "libbz2-1.dll", "libcrypto-3-x64.dll", "libdeflate.dll", "libexpat-1.dll", "libgcc_s_seh-1.dll", "libgif-7.dll",
    "libiconv-2.dll", "libjbig-0.dll", "libjpeg-8.dll", "liblz4.dll", "liblzma-5.dll", "libopenjp2-7.dll",
    "libpng16-16.dll", "libsharpyuv-0.dll", "libstdc++-6.dll", "libtiff-6.dll", "libwebp-7.dll", "libwebpmux-3.dll",
    "libwinpthread-1.dll", "libzstd.dll", "zlib1.dll")
TESSERACT_README = """Tesseract OCR {version} for the Skipwise runner's slides (Windows build by UB Mannheim).
Source: https://github.com/tesseract-ocr/tesseract (Apache License 2.0, LICENSE in this folder)
Windows build: https://github.com/UB-Mannheim/tesseract ({url})
The DLLs are the MinGW-w64 builds of their projects, under their own licences (Leptonica BSD-2, libarchive BSD,
OpenSSL Apache-2.0, libiconv LGPL-2.1, GCC runtime with the runtime library exception, the image libraries under
their permissive licences); their sources are linked from the Windows build's page.
Language data: tessdata_fast 4.1.0 (Apache-2.0), downloaded by the runner when a language is needed.
"""


def seven_zip() -> str:
    found = shutil.which("7z")
    for p in (found, r"C:\Program Files\7-Zip\7z.exe"):
        if p and Path(p).is_file():
            return p
    raise SystemExit("7-Zip (7z) not found: needed to take Tesseract out of its installer")


def strip_tool() -> str | None:
    found = shutil.which("strip") or shutil.which("x86_64-w64-mingw32-strip")
    for p in (found, r"C:\mingw64\bin\strip.exe", r"C:\msys64\mingw64\bin\strip.exe"):
        if p and Path(p).is_file():
            return p
    return None


def windows_tesseract(dest: Path) -> None:
    import hashlib
    work = BUILD / "tesseract-setup"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    setup = work / "setup.exe"
    with urllib.request.urlopen(TESSERACT_URL, timeout=300) as resp:
        setup.write_bytes(resp.read())
    if hashlib.sha256(setup.read_bytes()).hexdigest() != TESSERACT_SHA256:
        raise SystemExit("the Tesseract installer's SHA-256 does not match")
    run(seven_zip(), "x", "-y", f"-o{work / 'x'}", str(setup))
    dest.mkdir(parents=True)
    for name in TESSERACT_FILES:
        shutil.copy2(work / "x" / name, dest / name)
    shutil.copy2(work / "x" / "doc" / "LICENSE", dest / "LICENSE")
    version = TESSERACT_URL.rsplit("-", 1)[-1].removesuffix(".exe")
    (dest / "README.txt").write_text(TESSERACT_README.format(version=version, url=TESSERACT_URL), encoding="utf-8")
    strip = strip_tool()
    if strip:
        for name in TESSERACT_FILES:
            run(strip, "--strip-debug", str(dest / name))
    else:
        print("warning: no strip tool; Tesseract's DLLs keep their debug info (about 100 MB more)")
    if os.name == "nt":
        print(run(str(dest / "tesseract.exe"), "--version"))
    shutil.rmtree(work)


def windows_folder() -> Path:
    """Everything the installer puts down, in build/package/windows."""
    src = WIN_SRC
    if src.exists():
        shutil.rmtree(src)
    python = bundled_python(src / "python")
    install_runner(python, "inno", [])
    windows_tesseract(src / "tesseract")
    icons(python, src)
    (src / "bin").mkdir()
    for name in ("skipwise-runner", "dwyt-runner"):  # the command, and its name before 0.17
        (src / "bin" / f"{name}.cmd").write_text('@"%~dp0..\\python\\python.exe" -m dwyt_runner.cli %*\r\n',
                                                 encoding="ascii")
    launcher(python, src / "dwyt-logo.ico")
    return src


def windows_installer(signed_uninstaller: bool = False) -> Path:
    DIST.mkdir(exist_ok=True)
    extra = ["/DSignUninstaller=1", f"/DUninstDir={UNINST_DIR}"] if signed_uninstaller else []
    run(iscc(), f"/DAppVersion={version()}", f"/DSrc={WIN_SRC}", f"/O{DIST}", *extra, str(ISS))
    out = DIST / f"skipwise-runner-setup-{version()}.exe"
    print("built", out)
    return out


def windows() -> Path:
    """The unsigned installer in one go (local builds, branches without signing)."""
    windows_folder()
    return windows_installer()


# ---- Authenticode signing through SignPath (decision 19.27) ----------------------------------------------------
# Smart App Control checks every EXE and DLL Windows loads, not only the installer: every PE file of the package
# that has no valid signature yet (Python, Skipwise.exe, the .pyd modules, wheels' DLLs) is signed, and so is Inno
# Setup's uninstaller. Files already signed (Microsoft's runtime DLLs, NVIDIA's) keep their own signature.
# SignPath signs .exe/.dll: a .pyd travels as <name>.pyd.dll, the uninstaller (Inno's .e32) as <name>.e32.exe.
PE_SUFFIXES = (".exe", ".dll", ".pyd")
SIGN_DIR = BUILD / "tosign"
UNINST_DIR = BUILD / "uninstaller"  # Inno Setup's SignedUninstallerDir
MANIFEST = "signing-manifest.json"
TRANSPORT = {".pyd": ".pyd.dll", ".e32": ".e32.exe"}
# SignPath's test-signing policy uses a certificate Windows does not trust: signed, but "UnknownError"
ACCEPTED = {"Valid"} | ({"UnknownError"} if os.environ.get("SIGNPATH_TEST") == "1" else set())


def transport_name(rel: str) -> str:
    for suffix, carried in TRANSPORT.items():
        if rel.lower().endswith(suffix):
            return rel[:-len(suffix)] + carried
    return rel


def original_name(rel: str) -> str:
    for suffix, carried in TRANSPORT.items():
        if rel.lower().endswith(carried):
            return rel[:-len(carried)] + suffix
    return rel


def pe_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in PE_SUFFIXES)


def signature_status(files: list[Path]) -> dict[Path, str]:
    """Authenticode status of each file as Windows sees it (Valid, NotSigned, HashMismatch ...)."""
    if not files:
        return {}
    listing = BUILD / "signature-files.txt"
    listing.write_text("\n".join(str(f) for f in files), encoding="utf-8")
    script = ("Get-Content -LiteralPath '" + str(listing) + "' | ForEach-Object { [pscustomobject]@{ p = $_; "
              "s = (Get-AuthenticodeSignature -LiteralPath $_).Status.ToString() } } | ConvertTo-Json -AsArray")
    out = run("pwsh", "-NoProfile", "-Command", script)
    return {Path(row["p"]): row["s"] for row in json.loads(out)}


def signing_plan(root: Path, uninstallers: list[Path], status: dict[Path, str]) -> dict[str, str]:
    """{path inside the signing ZIP: source file}: the package's unsigned PE files and the uninstallers."""
    plan = {}
    for f in pe_files(root):
        if status.get(f) != "Valid":
            plan["package/" + transport_name(f.relative_to(root).as_posix())] = str(f)
    for f in uninstallers:
        plan["uninstaller/" + transport_name(f.name)] = str(f)
    return plan


def uninstaller_to_sign() -> list[Path]:
    """Inno Setup writes its uninstaller to SignedUninstallerDir as an .e32 and stops when no signed copy is there
    yet; that file is signed outside and the next compile embeds its signature."""
    if UNINST_DIR.exists():
        shutil.rmtree(UNINST_DIR)
    UNINST_DIR.mkdir(parents=True)
    probe = BUILD / "iscc-probe"
    r = subprocess.run([iscc(), f"/DAppVersion={version()}", f"/DSrc={WIN_SRC}", f"/O{probe}", "/DSignUninstaller=1",
                        f"/DUninstDir={UNINST_DIR}", str(ISS)], text=True, capture_output=True)
    found = sorted(UNINST_DIR.glob("*.e32"))
    if not found:
        raise SystemExit("Inno Setup made no uninstaller to sign:\n" + r.stdout[-3000:] + r.stderr[-2000:])
    return found


def windows_tosign() -> Path:
    """build/package/tosign: the files to sign (uploaded as one GitHub artifact) and the manifest to put them back."""
    plan = signing_plan(WIN_SRC, uninstaller_to_sign(), signature_status(pe_files(WIN_SRC)))
    if SIGN_DIR.exists():
        shutil.rmtree(SIGN_DIR)
    for inside, source in plan.items():
        target = SIGN_DIR / inside
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (BUILD / MANIFEST).write_text(json.dumps(plan, indent=1), encoding="utf-8")
    print(f"{len(plan)} files to sign in {SIGN_DIR}")
    return SIGN_DIR


def put_back(signed: Path, plan: dict[str, str]) -> list[Path]:
    """Copy the signed files over the originals; every planned file must have come back."""
    missing = [inside for inside in plan if not (signed / inside).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} files did not come back signed, first: {missing[:5]}")
    for inside, source in plan.items():
        shutil.copyfile(signed / inside, source)
    return [Path(source) for source in plan.values()]


def windows_putback(signed: Path) -> None:
    plan = json.loads((BUILD / MANIFEST).read_text(encoding="utf-8"))
    files = put_back(signed, plan)
    bad = {f: s for f, s in signature_status(files).items() if s not in ACCEPTED}
    if bad:
        raise SystemExit(f"not validly signed: {list(bad.items())[:5]}")
    print(f"{len(files)} files signed and back in place")


def windows_verify(*paths: str) -> None:
    """Every PE file under the given paths (files or folders) must be validly signed."""
    files = [f for p in map(Path, paths) for f in (pe_files(p) if p.is_dir() else [p])]
    bad = {f: s for f, s in signature_status(files).items() if s not in ACCEPTED}
    if bad:
        raise SystemExit(f"{len(bad)} files not validly signed, first: {list(bad.items())[:10]}")
    print(f"{len(files)} files validly signed")


APPRUN = """#!/bin/sh
# dwyt runner AppImage: no arguments (or --hidden / --browser / --after) = the app with its window,
# anything else = the command line (dwyt-runner run, check, update ...)
HERE="$(dirname "$(readlink -f "$0")")"
export PATH="$HERE/python/bin:$PATH"
export LD_LIBRARY_PATH="$HERE/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYWEBVIEW_GUI="${PYWEBVIEW_GUI:-qt}"
case "${1:-}" in
  ""|--hidden|--browser|--after) exec "$HERE/python/bin/python3" -m dwyt_runner.app.main "$@" ;;
  *) exec "$HERE/python/bin/python3" -m dwyt_runner.cli "$@" ;;
esac
"""

# Qt 6.5+ needs libxcb-cursor for X11, which many desktops do not have (Ubuntu 22.04): carried along
BUNDLE_LIBS = ("libxcb-cursor.so.0",)


def bundle_libs(lib_dir: Path) -> None:
    lib_dir.mkdir(parents=True, exist_ok=True)
    listing = run("ldconfig", "-p")
    for name in BUNDLE_LIBS:
        path = next((line.split("=>")[-1].strip() for line in listing.splitlines()
                     if line.strip().startswith(name + " ") and "x86-64" in line), "")
        if not path:
            raise SystemExit(f"{name} is missing on the build machine (apt install libxcb-cursor0)")
        shutil.copy(Path(path).resolve(), lib_dir / name)


DESKTOP = """[Desktop Entry]
Type=Application
Name=Skipwise runner
Name[hu]=Skipwise futtató
Name[de]=Skipwise-Runner
Name[it]=Runner Skipwise
Name[es]=Runner de Skipwise
Name[fr]=Runner Skipwise
Comment=Transcription and summaries for Skipwise, on this computer
Comment[hu]=Átírás és összefoglaló a Skipwise-nak, ezen a gépen
Comment[de]=Transkription und Zusammenfassungen für Skipwise, auf diesem Rechner
Comment[it]=Trascrizione e riassunti per Skipwise, su questo computer
Comment[es]=Transcripción y resúmenes para Skipwise, en este equipo
Comment[fr]=Transcription et résumés pour Skipwise, sur cet ordinateur
Exec=skipwise-runner
Icon=dwyt-runner
Categories=AudioVideo;Utility;
Terminal=false
"""


def appimage() -> Path:
    appdir = BUILD / "AppDir"
    if appdir.exists():
        shutil.rmtree(appdir)
    appdir.mkdir(parents=True)
    python = bundled_python(appdir / "python")
    install_runner(python, "appimage", ["pywebview[qt]>=5"])
    bundle_libs(appdir / "usr" / "lib")
    (appdir / "AppRun").write_text(APPRUN, encoding="utf-8")
    (appdir / "AppRun").chmod(0o755)
    (appdir / "dwyt-runner.desktop").write_text(DESKTOP, encoding="utf-8")
    icons(python, BUILD / "icons")
    shutil.copy(BUILD / "icons" / "dwyt-runner.png", appdir / "dwyt-runner.png")
    tool = BUILD / "appimagetool"
    if not tool.exists():
        urllib.request.urlretrieve(APPIMAGETOOL, tool)
        tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    DIST.mkdir(exist_ok=True)
    out = DIST / f"skipwise-runner-{version()}-x86_64.AppImage"
    run(str(tool), "--appimage-extract-and-run", "--no-appstream", str(appdir), str(out),
        env={**os.environ, "ARCH": "x86_64"})
    print("built", out)
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "windows":
        windows()
    elif what == "windows-folder":
        windows_folder()
    elif what == "windows-tosign":
        windows_tosign()
    elif what == "windows-putback":
        windows_putback(Path(sys.argv[2]))
    elif what == "windows-installer":
        windows_installer(signed_uninstaller="--signed-uninstaller" in sys.argv)
    elif what == "windows-verify":
        windows_verify(*sys.argv[2:])
    elif what == "appimage":
        appimage()
    elif what == "wheel":
        print(wheel())
    else:
        raise SystemExit(__doc__)
