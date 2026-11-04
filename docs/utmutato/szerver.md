# A Skipwise futtató szerverre (VPS, LXC, NAS)

Linux systemd-vel (Debian, Ubuntu, Fedora, …), internetkapcsolattal. Ablak nincs: a beállítás a Skipwise-ból megy.

## Telepítés egy paranccsal

1. A Skipwise **Futtatók** oldalán: „Új futtató” → **Szerverre (Linux)** → **Telepítőparancs kérése**.
2. A kapott parancsot futtasd a szerveren (5 percig érvényes, egyszer használható):
   ```bash
   curl -fsSL https://skipwise.app/futtato.sh | sh -s -- ABCD-EFGH
   ```
   - **rendszergazdaként:** külön `dwyt-runner` felhasználót hoz létre (`/var/lib/dwyt-runner`), és rendszerszolgáltatásként indítja;
   - **saját felhasználóként:** a saját felhasználód szolgáltatásaként, rendszergazdai jog nélkül. Hogy kijelentkezés után is fusson, egy rendszergazda egyszer futtassa: `loginctl enable-linger FELHASZNÁLÓ`.
3. Kész: a futtató megjelenik a Futtatók oldalon.

Mit csinál a parancs pontosan: a `/futtato.sh` címen olvasható, érdemes elolvasni. Előbb beváltja a kódot, telepíti az uv-t és a futtatót, beírja a párosítást, végül systemd-szolgáltatást ír és elindítja.

## Beállítás a Skipwise-ból

A futtató kártyáján a **Beállítás** gomb: működési mód, modellek, LLM-cím, figyelt mappák stb. A kulcsokat (AssemblyAI, LLM) a böngésződ a futtató saját kulcsával titkosítja, így a Skipwise nem látja őket, és visszaolvasni sem lehet. A változást a futtató pár másodpercen belül átveszi, ellenőrzi, és visszajelez; minden változás a napló részbe kerül, és értesítés megy róla. A szerver saját útvonalai (whisper.cpp helye, sütifájl, munkamappa) csak a szerveren állíthatók: `skipwise-runner config set …`; rendszergazdaként telepítve a futtató felhasználójával: `runuser -u dwyt-runner -- env HOME=/var/lib/dwyt-runner /var/lib/dwyt-runner/.local/bin/dwyt-runner config set …`.

## Diák a videókból (nem kötelező)

Az egysoros telepítő rendszergazdaként a Tesseractot is telepíti a rendszer csomagjából (`tesseract-ocr`); a nyelvi adatokat a futtató maga tölti le, ha hiányoznak. Régebbi telepítésnél vagy saját felhasználóként kézzel: `sudo apt install tesseract-ocr`, utána `sudo systemctl restart dwyt-runner` (a futtató indításkor nézi meg, mivel tud olvasni).

## Frissítés

Ha van újabb változat, a Skipwise Futtatók oldalán a kártyán megjelenik a **Frissítés most** gomb: a futtató két munka között telepíti az aláírt kiadást, és újraindul. A szerveren kézzel: `skipwise-runner update` (`--check`: csak megnézi, `--back`: vissza az előzőre; rendszergazdaként telepítve a fenti `runuser …` előtaggal), utána `systemctl restart dwyt-runner`. Ha azt szeretnéd, hogy magától frissüljön: Beállítás → „A futtató frissítése magától” → be.

## Napló, leállítás, eltávolítás

| Mit | Rendszergazdaként telepítve | Saját felhasználóként telepítve |
| --- | --- | --- |
| Állapot | `systemctl status dwyt-runner` | `systemctl --user status dwyt-runner` |
| Napló | `journalctl -u dwyt-runner -f` | `journalctl --user -u dwyt-runner -f` |
| Leállítás | `systemctl disable --now dwyt-runner` | `systemctl --user disable --now dwyt-runner` |

Eltávolítás: állítsd le a szolgáltatást, töröld a szolgáltatásfájlt (`/etc/systemd/system/dwyt-runner.service`, illetve `~/.config/systemd/user/dwyt-runner.service`), majd `skipwise-runner reset --models --yes` és `uv tool uninstall dwyt-runner` (rendszergazdaként a `dwyt-runner` felhasználóval: `runuser -u dwyt-runner -- …`, utána `userdel -r dwyt-runner`). A Skipwise-ban a futtatót a Futtatók oldalon vond vissza.
