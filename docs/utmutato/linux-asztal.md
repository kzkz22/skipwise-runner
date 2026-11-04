# A Skipwise futtató Linux asztali gépre

64 bites Linux grafikus felülettel (GNOME, KDE, Xfce, Cinnamon …). Rendszergazdai jog nem kell.

## Telepítés

1. A Skipwise **Futtatók** oldalán: „Új futtató” → **Asztali gépre** → **Linux AppImage letöltése** (`skipwise-runner-x86_64.AppImage`, kb. 300 MB; az ablakhoz szükséges Qt is benne van).
2. Tedd egy állandó helyre (például `~/Alkalmazások/`), és tedd futtathatóvá: a fájlkezelőben Tulajdonságok → Jogosultságok → „Futtatható”, vagy terminálban:
   ```bash
   chmod +x ~/Alkalmazások/skipwise-runner-x86_64.AppImage
   ```
3. Indítsd el dupla kattintással. Ha nem indul, és a hibaüzenet a FUSE-ra panaszkodik: `sudo apt install libfuse2` (Ubuntu 24.04-től a csomag neve `libfuse2t64`).

Az első indítás lépései ugyanazok, mint Windowson: [lásd ott](windows.md#az-első-indítás). A kulcsokat a rendszer kulcstárolója (GNOME Keyring, KWallet) őrzi, ha van; különben a beállítási fájl, amelyet csak te olvashatsz.

## Mindennap

- Tálcaikon ott, ahol az asztal a hagyományos tálcaikonokat is mutatja (például Xfce, KDE, Cinnamon). Ahol nincs ilyen (GNOME), a futtató a háttérben dolgozik, és az AppImage újbóli elindítása előhozza az ablakot.
- Indulás bejelentkezéskor: Működés lap → „Indítás bejelentkezéskor” (a `~/.config/autostart` mappába ír).
- Helyek: beállítások `~/.config/dwyt-runner`, letöltött modellek `~/.local/share/dwyt-runner/models`, napló `~/.config/dwyt-runner/app.log`.

## Diák a videókból (nem kötelező)

A diák szövegéhez telepítsd a Tesseractot: `sudo apt install tesseract-ocr tesseract-ocr-hun` (Fedorán `sudo dnf install tesseract tesseract-langpack-hun`). Tesseract nélkül a diákat csak a látó modell olvassa, ha van.

## Frissítés és eltávolítás

A futtató magától frissül (aláírt kiadásból, két munka között); az AppImage fájlt cseréli ki, és újraindul. A Gépem lapon kérheted azonnal is, vagy állhatsz vissza az előzőre.

Eltávolítás: Működés → indulás bejelentkezéskor kikapcsolva, majd töröld az AppImage-et. Ha a beállításokat és a modelleket is: Működés → **Alaphelyzet** (a modellek külön pipával), utána töröld a fájlt. A Skipwise-ban a futtatót a Futtatók oldalon vond vissza.
