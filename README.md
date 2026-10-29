# Skipwise futtató (skipwise-runner)

A **Skipwise** futtatója. A saját gépeden vagy VPS-eden fut (Linux, Windows, macOS), és kifelé kapcsolódik a Skipwise központhoz ([dwyt-central](https://github.com/kzkz22/dwyt-central)): onnan kér munkát, és oda adja le az eredményt. Nyitott port, VPN vagy routerbeállítás nem kell.

Amit végez:

- **YouTube-felirat** letöltése (gyors, ingyenes, beszélők nélkül);
- **felhős átírás** az AssemblyAI-jal, a **saját kulcsoddal** (a költség a tiéd);
- **helyi átírás beszélőkkel** a saját GPU-don (Whisper; ehhez [skipwise-airouter](https://github.com/kzkz22/skipwise-airouter) és [skipwise-gpu](https://github.com/kzkz22/skipwise-gpu) kell);
- **magyar összefoglaló és fordítás** bármely OpenAI-kompatibilis szerverrel: LiteLLM, OpenRouter, llama-server, Ollama, vagy próbaként az AssemblyAI LLM Gateway-jével, ugyanazzal a kulccsal, mint az átírás.

A kulcsaid csak itt, a gépeden vannak; a központ soha nem látja őket. A YouTube a te IP-címedről, a te sávszélességeddel töltődik.

- Telepítési útmutatók: [docs/utmutato](docs/utmutato/README.md)
- A terv, a döntésnapló és a központ–futtató protokoll a projekt (nem nyilvános) dwyt-docs tárában van.

## Telepítés

**Windowsra és Linux asztali gépre:** a Skipwise Futtatók oldalán „Új futtató” → *Asztali gépre*: Windows-telepítő (rendszergazdai jog nélkül, kb. 70 MB) vagy Linux AppImage (kb. 300 MB). Mindkettő hozza a saját Pythonját; a whisper.cpp-t első használatkor tölti le. Útmutató: [Windows](docs/utmutato/windows.md), [Linux asztal](docs/utmutato/linux-asztal.md).

**Linux szerverre egy paranccsal (ajánlott):** a Skipwise Futtatók oldalán az „Új futtató” → *Szerverre (Linux)* → „Telepítőparancs kérése”. A kapott parancs (`curl -fsSL https://skipwise.app/futtato.sh | sh -s -- KÓD`, 5 percig érvényes, egyszer használható) párosít, telepíti az uv-t és a futtatót, majd systemd-szolgáltatásként elindítja: rendszergazdaként külön `dwyt-runner` felhasználóval (`/var/lib/dwyt-runner`), különben a saját felhasználód szolgáltatásaként. Utána minden beállítás a Skipwise-ban, a futtató **Beállítás** gombjával megy (lásd: Távoli beállítás). A kézi út:

Az [`uv`](https://docs.astral.sh/uv/) csomagkezelővel, rendszergazdai jog nélkül; a Pythont, az ffmpeg-et és a YouTube-hoz kellő `deno`-t is hozza.

1. Az `uv` telepítése (egyszer):
   - **Windows** (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - **macOS, Linux**: `curl -LsSf https://astral.sh/uv/install.sh | sh`

   Utána nyiss új terminált.
2. A futtató: `uv tool install --python 3.13 https://github.com/kzkz22/skipwise-runner/archive/refs/heads/main.zip` (gitre nincs szükség). A `--python 3.13` azért kell, mert a legújabb Pythonhoz (3.15) egyes csomagoknak még nincs kész változata, és akkor C fordítót kérne; a 3.13-at az uv magától letölti, a gép saját Pythonját nem érinti
3. Párosítás: `skipwise-runner pair https://skipwise.app`. Kiír egy kódot; a központ **Futtatók** oldalán írd be, és adj nevet a futtatónak.
4. Kulcsok és modellek (csak ami kell):

   ```
   skipwise-runner config set assemblyai_key KULCS            # felhős átírás
   skipwise-runner config set llm_url https://openrouter.ai/api/v1
   skipwise-runner config set llm_key KULCS
   skipwise-runner config set llm_public_model MODELLNÉV      # nyilvános videók összefoglalója
   skipwise-runner config set llm_private_model MODELLNÉV     # saját felvételekhez: helyi modell javasolt
   skipwise-runner config set llm_translate_model MODELLNÉV   # (nem kötelező) fordításhoz más modell a nyilvános videóknál
   skipwise-runner config set llm_ask_model MODELLNÉV         # (nem kötelező) a központban feltett kérdésekhez, például a gondolkodó „kerdes”
   ```

   A központban feltett kérdésekre (az elem oldalán és a Keresés oldalon) a futtató válaszol: nyilvános videóknál az `llm_ask_model`, ha nincs, az `llm_public_model`; saját felvételnél mindig az `llm_private_model`. A modell nem kap eszközöket, csak a központ által összeszedett átiratrészleteket.

   Saját LiteLLM-mel (például a dwyt-airouterrel) az `llm_url` a LiteLLM címe, a modellnevek az ott beállított nevek.

   **Helyi modellszerver ugyanezen a gépen (Ollama vagy LM Studio):** az `llm_url` maradjon üres, és add meg, melyiket használja; a modellnevek a szerver saját nevei:

   ```
   skipwise-runner config set local_llm ollama            # vagy: lmstudio
   skipwise-runner llm pull qwen3:8b                      # letöltés (LM Studióban például: qwen/qwen3-8b)
   skipwise-runner config set llm_private_model qwen3:8b
   skipwise-runner llm status                             # fut-e, milyen modellek vannak, mi van betöltve
   ```

   A futtató elindítja a szervert, ha telepítve van, de nem fut (LM Studiónál `lms server start`), és ugyanígy tölt be és ürít ki modellt (`skipwise-runner llm load NÉV`, `skipwise-runner llm unload`). Ha az LM Studióban bekapcsoltad az API-tokent: `skipwise-runner config set lmstudio_token TOKEN`. Más cím: `local_llm_url`. Ha az `llm_url` meg van adva, az elsőbbséget kap.

   Ha a központ elemoldalán más modellt is választhatnál egy összefoglalóhoz vagy fordításhoz, sorold fel őket (nem kötelező):

   ```
   skipwise-runner config set llm_models cloud/okos,osszefoglalo-gyors   # nyilvános videókhoz
   skipwise-runner config set llm_private_models helyi-nagy              # saját felvételekhez (csak amit ide engedsz)
   ```

   A futtató csak ezeket (és a fenti alapmodelleket) fogadja el; saját felvétel szövege más modellhez nem kerül.

   A központ Futtatók oldala mutatja, mi fut az egyes nevek mögött (modell, helyben vagy felhőben, gondolkodás, mi válaszol, ha a GPU-gép ki van kapcsolva). A dwyt-airouter LiteLLM-je ezt maga megadja (`/model/info`); a futtató óránként lekérdezi, és csak a leírást küldi tovább, kulcsot és címet nem. Más szervernél a leírást te adhatod meg:

   ```
   skipwise-runner config set llm_model_notes "osszefoglalo=Összefoglaló, gyors; cloud/okos=Felhős, alapos"
   ```

   Helyi Whisper (saját GPU, az AI-router llama-swapján át):

   ```
   skipwise-runner config set whisper_url http://AI-ROUTER-IP:8080/upstream/whisper
   skipwise-runner config set whisper_models large-v3,large-v3-turbo   # ami a GPU-gépen van
   ```

   Ha a GPU-gép épp nem érhető el, a helyi munka vár, és később újra próbálkozik. Felhőre nem vált át. A `whisper_timeout_min` (alapból 60) egy munka leghosszabb ideje; hosszú felvételnél a hossz négyszerese.
   **Helyi átírás ezen a gépen (külön GPU-gép nélkül):** a whisper.cpp és a sherpa-onnx (beszélők) a gépen fut. Ehhez a futtatót a `local` kiegészítővel kell telepíteni, és egyszer letölteni a modelleket (kb. 3,5 GB):

   ```
   uv tool install --force --python 3.13 "dwyt-runner[local] @ https://github.com/kzkz22/skipwise-runner/archive/refs/heads/main.zip"
   skipwise-runner whisper prepare                        # whisper.cpp, large-v3, beszédfelismerés, beszélőmodellek
   skipwise-runner whisper try felvetel.mp3               # próba: kiírja az átiratot, a központba nem küld semmit
   skipwise-runner config set whisper_builtin igen
   ```

   **Próbafuttatás:** `skipwise-runner trial` átír egy 33 másodperces mintát (J. F. Kennedy beiktatási beszédének közkincs részlete, háromszor), és ha van helyi modellszerver, egy rövid magyar összefoglalót íratja meg; a letöltések és az első betöltés nem számít bele. Megmondja, mennyi idő alatt készülne el egy 1 órás videó, hány token/mp a helyi modell, és melyik működési módot ajánlja (csak ajánl). Más helyi modellel: `skipwise-runner trial --llm MODELLNÉV`. A whisper.cpp modelljének betöltése (pár másodperc) nem számít bele a sebességbe, a beszélők szétválasztása igen. A `skipwise-runner machine` az utolsó próba eredményét is mutatja, és szól, ha azóta új videókártya került a gépbe.

   Ha ugyanezen a gépen helyi modellszerver is fut (`local_llm`), egy átírás előtt a futtató kiüríti a betöltött nyelvi modellt a videókártyáról (a szerver a következő kérésnél magától visszatölti). Amit csak a futtató használt, azt azonnal; amit más töltött be, annál *udvarias módban* (alapból be) előbb megvárja, hogy egy percig ne használja senki (a videókártya terhelése és Ollamánál a modell kérései alapján), legfeljebb 10 percig: `skipwise-runner config set polite_wait_min 10`, kikapcsolás: `skipwise-runner config set polite_mode nem`.

   Windowson a futtató a whisper.cpp hivatalos kiadását tölti le (NVIDIA-kártyánál a CUDA-s változatot); Linuxon és macOS-en a `whisper-cli` programnak a PATH-on kell lennie, vagy add meg: `skipwise-runner config set whisper_cli /út/whisper-cli` (egy kicsomagolt kiadás mappája is jó). A beszélők számát a futtató maga becsli (`diarize_threshold`, alapból 0,8; nagyobb érték kevesebb beszélőt ad); ha a beküldéskor megadod, azt használja. Ha a videókártyán nem sikerül, egyszer a processzoron megismétli, és ezt jelzi. Ha a Windows alkalmazásvezérlése (például az Intelligens alkalmazásvezérlés) nem engedi a whisper.cpp programját, a munka ezt az üzenetet adja; ezen a gépen ilyenkor a felhős átírás marad.

   Minden beállítás jelentése, megengedett értékei és alapértéke: `skipwise-runner config describe` (a `config set` ez alapján ellenőriz, és megmondja, ha egy érték nem jó).

5. Ellenőrzés: `skipwise-runner check`. A gép képességei: `skipwise-runner machine` (videókártya, memória, szabad hely, helyi modellszerver: Ollama vagy LM Studio; mire elég, és melyik működési módot ajánlja; `--json` a gépi feldolgozáshoz). Induláskor a futtató csendben újranézi, és szól, ha változott, például új videókártya került a gépbe.
6. Indítás: `skipwise-runner run` (leállítás: Ctrl+C). A képességeit (melyik mód, melyik modell) induláskor jelenti be, ezért beállításváltozás után indítsd újra.

A beállítások helye: Windowson `%APPDATA%\dwyt-runner\config.json`, macOS-en `~/Library/Application Support/dwyt-runner/config.json`, Linuxon `~/.config/dwyt-runner/config.json`. A fájlt csak te olvashatod; a `config show` a kulcsokat elrejtve mutatja. A kulcsok (a párosítás tokenje, az AssemblyAI- és az LLM-kulcs, az LM Studio tokenje) nem ebbe a fájlba kerülnek, hanem a rendszer kulcstárolójába: Windowson a Hitelesítőadat-kezelőbe, macOS-en a kulcskarikába, Linux-asztalon a Secret Service-be (GNOME Kulcstartó, KWallet). Ha a gépen nincs ilyen (például egy szerveren), a fájlban maradnak. A meglévő kulcsokat a futtató az első indításkor átteszi. Hol vannak, és működnek-e: `skipwise-runner keys` (kulcsonként egy próbakérés; a `config set` egy új kulcsot rögtön ki is próbál). Ha mégis a fájlt szeretnéd: `skipwise-runner config set secret_store file`.

A letöltött modellek (whisper.cpp, Whisper-modell, beszélőmodellek, együtt kb. 3,5 GB) külön helyen vannak: Windowson `%LOCALAPPDATA%\dwyt-runner\models` (nem a vándorló profilban), Linuxon `~/.local/share/dwyt-runner/models`, macOS-en a beállítások mellett. A 0.9.3 előtt a beállítások mellett voltak; a futtató az első indításkor egyszer átteszi őket (`DWYT_RUNNER_DATA` felülírja a helyet).

**yt-dlp frissítése:** a YouTube gyakran változik, és egy régi yt-dlp egyes videóknál „HTTP 403” hibával akad el. A futtató induláskor, utána naponta és egy ilyen hiba után megnézi a PyPI-n, van-e újabb yt-dlp, és két munka között telepíti (`ytdlp_auto_update`, alapból be). A Gépem lapon látszik a telepített és a legújabb változat, és az „Újabb keresése” gombbal azonnal is kérhető. A futtató újratelepítése (`uv tool install --force --refresh …`) is a legújabbat hozza. Ablak nélküli (szerveres) futtatónál: `skipwise-runner config set ytdlp_auto_update nem` (vagy `igen`), utána a szolgáltatás újraindítása (`systemctl restart dwyt-runner`).

**Próba egyetlen AssemblyAI-kulccsal (0.12):** Felhő módban a telepítő alapból azt ajánlja, hogy az összefoglalót is az AssemblyAI írja (LLM Gateway, `https://llm-gateway.assemblyai.com/v1`): így egy új felhasználónak elég egy AssemblyAI-regisztráció (50 $ induló kredit, bankkártya nélkül). Ingyenes fiókkal csak a kis `qwen3.5-4b-32k-fast` próba-modell érhető el; a futtató mindig alacsony hőmérséklettel (0,3) kéri, mert e nélkül kitalált szavak és idegen írásjelek kerülhetnek a magyar szövegbe, és a „túl sok kérés” (HTTP 429) válaszoknál vár, majd folytatja. A magyar összefoglaló gyengébb lehet; a telepítő, a Működés lap és a Skipwise is jelzi. Átállás OpenRouterre a Működés lapon (vagy a telepítő Hozzávalók lépésében); a gateway-en az `llm_key`-ben tárolt OpenRouter-kulcs megmarad, ott mindig az AssemblyAI-kulcs megy.

**A fájl sávjai (0.14):** feltöltött vagy figyelt mappás fájlnál a futtató megnézi a fájl hang- és feliratsávjait. A megadott nyelvű hangsávot írja át (üresen az alapértelmezettet); több hangsávnál csak azt. Felirat módban a fájlba ágyazott szöveges feliratot (SubRip, ASS/SSA, WebVTT) használja: azonnal, ingyen, beszélők nélkül. A képként tárolt felirat (PGS, VobSub) nem olvasható ki. A talált sávokat a Skipwise az elem oldalán mutatja.

**Kategóriák (0.15):** az összefoglaló után a futtató nyelvi modellje egy rövid külön kéréssel kiválasztja az elem fő és legfeljebb két mellékkategóriáját a Skipwise-ban beállított listából (a dwyt-central 0.21 küldi), és ritkán új kategóriát javasol; a javaslatról a Skipwise-ban te döntesz. A régebbi elemeket a Skipwise a háttérben, kis adagokban küldi besorolásra (rejtett „Kategorizálás” munka; a Skipwise Beállításaiban kikapcsolható). Saját felvételt csak a helyi modell sorol be, ahogy az összefoglalót is. Ha a választás nem sikerül, az összefoglaló ettől még elkészül.

**Fejezetek (0.18):** ha a videónak van feltöltői fejezete, a futtató az átírással együtt elküldi. Ha nincs, és a felvétel legalább 5 perces, a Skipwise az összefoglalóval együtt fejezeteket is kér: a futtató nyelvi modellje az összefoglaló után egy rövid külön kéréssel magyar fejezetcímeket ír az átirat időpontjaihoz. A régebbi elemekhez a Skipwise-ban gombbal kérhető (Fejezetek munka). Saját felvételnél csak a helyi modell dolgozik rajta. Ha a fejezetek nem sikerülnek, az összefoglaló ettől még elkészül.

**A futtató nyelve (0.21):** az ablak, a tálcaikon, a telepítő lépései, a parancssor, a beállítások leírása és a hibaüzenetek (amelyeket a Skipwise is megmutat) magyarul, angolul, németül, olaszul, spanyolul és franciául is megvannak. Alapból (`ui_language` = `auto`) a párosított Skipwise-fiók felületi nyelvét követi (a Skipwise minden munkakérésre megmondja), párosítás előtt a rendszerét; más nyelvű rendszeren angol. Kézzel: Működés → Részletes beállítások → *A futtató nyelve*, vagy `skipwise-runner config set ui_language de`. A Windows-telepítő a Windows nyelvén szól (a hat közül, különben angolul). Ha a fiók tartalomnyelve olyan, amelyen a helyi modellek érezhetően gyengébben írnak (litván, szlovák, szlovén, horvát, bolgár, lett), és az összefoglalót helyi modell írja, a Működés lap szól, és a Vegyes módot ajánlja. Hibás fordítást a Skipwise oldalainak alján lévő „Hibás fordítás?” linken lehet jelezni.

**Megbeszélések (0.22):** beküldéskor a Skipwise-ban megadható szakszavak és nevek (dwyt-central 0.31) az átírásba is bekerülnek: a beépített Whisper a kezdő promptjában kapja meg őket (ehhez olyan whisper.cpp kell, amely ismeri a `--carry-initial-prompt` kapcsolót; régebbinél figyelmeztetés jön, és a szavak nélkül fut), a GPU-gép Whispere szintén promptként, az AssemblyAI pedig `keyterms_prompt`-ként (ez óránként +0,05 USD, a költségbecslés tartalmazza). A Whisper ismétlő hurkait (ugyanaz a szó vagy rövid kifejezés legalább négyszer egymás után, ugyanaz a mondat legalább háromszor) a futtató kiveszi az átiratból, és ezt figyelmeztetésben jelzi. A bizonytalanul felismert szavakat (a beépített Whisper és az AssemblyAI valószínűsége, a GPU-gépen az illesztés pontszáma alapján) megjelöli, hogy a Skipwise aláhúzhassa őket. Új munkafajta az **Emlékeztető**: a nyelvi modell az átiratból kigyűjti a tanulságokat, döntéseket, felelőssel és határidővel kiosztott feladatokat, következő lépéseket és nyitott kérdéseket, mindegyiket az elhangzás időpontjával; csak azt, ami elhangzott (ki nem mondott felelős vagy határidő nem kerül bele). Hosszú felvételt részenként olvas, majd egy külön kéréssel összevonja. Saját felvételnél csak a helyi modell dolgozik rajta.

**Tudásalap, javítás, eredetik (0.23):** két új munkafajta a Skipwise 0.32-höz. A **Szójegyzék** munka az átiratból javaslatot gyűjt: ritka szakszavak, nevek, rövidítések, szervezetek, témák, a helyes alakjuk (a megadott résztvevőkből, szakszavakból és a könyvtári szójegyzékből) és az átiratban szereplő, valószínűleg félrehallott alakjuk; ami nincs az átiratban, azt a futtató kidobja. A **Javítás** munka csak a bizonytalan szavakat és a szójegyzék félrehallott alakjait írhatja át (a ragokat megtartva), mást nem; egy hely legfeljebb két szóval hosszabb lehet, új mondatot nem írhat. A javításokat a Skipwise jelölve mutatja, és egyenként visszavonhatók. A böngészőből feltöltött saját felvételt a futtató alapból megtartja (csak a saját fiókjáét, megosztott futtatóként másét nem) az **Eredetik** mappában (a modellek mellett, legfeljebb 50 GB-ig, fölötte a legrégebben használt törlődik), így egy új átírás nem kér új feltöltést; a Skipwise mutatja, melyik futtatón van, és onnan törölhető. Kikapcsolás: Működés → Részletes beállítások → *Feltöltött felvételek megőrzése*, vagy `skipwise-runner config set keep_originals false`.

**Kisebb ablak (0.12):** a futtató ablaka a képernyőhöz igazodik (legfeljebb 1040 × 800, a képernyő 92 × 86 százaléka), a betűméret kisebb, és ha a tartalom hosszabb az ablaknál, görgetősáv látszik.

**Gondolkodás felhős modellnél:** OpenRouter-modelleknél a `llm_reasoning` beállítás (Működés → Részletes beállítások): *a modell alapértelmezése*, *be* vagy *ki*. A futtató az OpenRouter modellistájából azt is kiolvassa, hogy az adott modell alapból gondolkodik-e, és mekkora a kontextusa; ezt a Skipwise Futtatók oldala mutatja. Ahol a modell mindig gondolkodik, ott a „ki” nem hat.

**Helyi modell kontextusa:** az LM Studio első használatkor gyakran csak 4096 tokennyi kontextussal tölti be a modellt, ebbe egy hosszabb átirat nem fér bele. A futtató ezért maga tölti be a modellt `local_llm_context` (alapból 32 768) tokennel; ha a videókártya memóriája nem bírja, kevesebbel (legfeljebb 8192-ig lefelé). Az Ollamánál a futtató kiolvassa, mekkora kontextussal fut a modell (ha a futtató indítja az Ollamát, ezzel a beállítással indítja). A hosszabb átiratot ehhez mérten bontja részekre, a kérdések hátterét pedig ehhez vágja.

**Alaphelyzet** (mintha most telepítenéd először): a Működés lap alján, vagy `skipwise-runner reset [--models]`. Kikapcsolja a bejelentkezéskori indulást, törli a kulcsokat és a beállításokat, a modelleket csak külön kérésre; utána a telepítő lépései jönnek. A Skipwise-ban a régi futtatót a Futtatók oldalon vond vissza.

## Ablak (asztali kezelőfelület)

A futtató saját ablakban is futhat: négy lap, ahogy a mintákon: **Állapot** (párosítás, a mostani munka, a legutóbbiak, a mai összesítés), **Működés** (működési mód: Felhő, Vegyes, Saját gép vagy Egyéni, ábrával; udvarias mód; részletes beállítások), **Gépem** (gépvizsgálat, próbafuttatás egy gombnyomásra), **Kulcsok** (beírás és próba; kiolvasni nem lehet).

A telepítő, a parancsikonok, az AppImage és az ablak a Skipwise logóját használja (videókeret előretekeréssel, alatta az összefoglaló sora); a tálcaikon is a logó, a sarkában színes pötty mutatja az állapotot (zöld: vár, kék: dolgozik, narancs: videókártya-csere, piros: leállt, szürke: nincs párosítva).

**Telepítő lépések:** egy még nem párosított futtató ablaka a telepítő hat lépésével nyílik: *Üdvözlés*, *Gépvizsgálat* (lámpák, próbafuttatás, ajánlott mód), *Működési mód* (négy kártya, ábrával), *Hozzávalók* (amit a választott mód kér: a beszédfelismerő letöltése folyamatjelzővel, az OpenRouter-kulcs próbával, AssemblyAI-kulcs, helyi modellszerver és modell, letöltéssel), *Párosítás* (a futtató megnyitja a Skipwise Futtatók oldalát kitöltött kóddal; ott egy gombbal jóváhagyod, miután összevetetted a kódot az ablakban láthatóval) és *Kész* (indulás bejelentkezéskor, a Skipwise megnyitása). A *Mégse* kihagyja a lépéseket; később az Állapot lapról újra elindíthatók. A kitöltött kódot a dwyt-central 0.14.2 fogadja.

```
uv tool install --force --python 3.13 "dwyt-runner[desktop,local] @ https://github.com/kzkz22/skipwise-runner/archive/refs/heads/main.zip"
skipwise-runner app
```

Az ablak a gépen belül, a 127.0.0.1-en futó oldal: csak az az ablak láthatja, amelyik megnyitotta. Ha a saját ablak nem indul (Linuxon a pywebview GTK-t vagy Qt-t kér), a `skipwise-runner app --browser` a böngészőben nyitja meg. A beállítások a futó munkát nem zavarják: a következő munkától érvényesek. 
A tálcaikon színe mutatja, mi történik: **zöld** – vár a következő munkára, **kék** – dolgozik, **narancs** – modellcsere (a videókártyán a beszédfelismerő fut, most ne használd a helyi nyelvi modelleket), **piros** – leállt (erről értesítést is kapsz), **szürke** – nincs párosítva. Az ablak bezárása csak elrejti, a futtató tovább dolgozik; az ikonra kattintva jön vissza. Ha a futtatót újra elindítod, nem indul el másodszor, hanem a már futó ablaka jön elő; ahol nincs tálca (néhány Linuxos asztal), így lehet visszahozni az ablakot. Kilépni a tálcaikon menüjéből vagy az ablak jobb felső sarkában lévő **Kilépés** gombbal lehet. A `skipwise-runner app --hidden` az ablak megnyitása nélkül indul (csak ha van tálcaikon).

**Indítás bejelentkezéskor:** a Működés fülön egy kapcsolóval (vagy `skipwise-runner autostart on|off|status`) a futtató a gép bekapcsolása után magától elindul a tálcán, ablak nélkül. Rendszergazdai jog nem kell: Windowson a felhasználó saját indítási listájába (a Feladatkezelő *Indítási alkalmazások* lapján látszik), Linuxon a `~/.config/autostart` mappába kerül. Windowson ilyenkor a konzolablak nélküli `dwyt-runner-app` indul; a naplója a beállítások mellett, az `app.log` fájlban van.

A telepítő a következő lépésben jön.

## Saját felvételek

Két úton:

- **Figyelt mappa** (ajánlott): a futtató a megadott mappák új hang- és videófájljait magától átíratja. A fájl el sem hagyja a gépedet; a központ csak az átiratot kapja. A fájlt a futtató nem mozgatja, nem módosítja és nem törli; egy fájlt egyszer dolgoz fel (a tartalma szerint).

  ```
  skipwise-runner config set watch_dirs "/home/zoli/Felvételek"        # több mappa vesszővel elválasztva
  skipwise-runner config set watch_mode local                          # local (alapból) vagy cloud
  skipwise-runner config set watch_language hu                         # üresen felismeri
  ```

  Utána indítsd újra a futtatót. A mappákat percenként nézi át; egy fájlt akkor vesz fel, ha már nem változik (a másolás befejeződött). Windowson a mappa például `C:\Users\Zoli\Felvételek`.
- **Feltöltés a böngészőből** a központ „Új átirat” oldalán. A fájl csak addig van a központon, amíg a futtatód le nem töltötte; a futtató a tartalmát ellenőrzi (SHA-256), mielőtt átíratja. A legnagyobb elfogadott méret: `max_file_mb` (alapból 4096).

Mindkét esetben a felvétel „saját”: más felhasználó nem kaphatja meg az eredményét, és az összefoglaló meg a fordítás csak az `llm_private_model` modellel készül.

## Mely oldalakról tölthet le

Alapból a YouTube-ról és néhány videós, podcastos oldalról: `youtube.com`, `youtu.be`, `telex.hu`, `444.hu`, `hvg.hu`, `index.hu`, `indavideo.hu`, `videa.hu`, `mediaklikk.hu`, `megaphone.fm`, `podcasts.apple.com`, `facebook.com`, `fb.watch`, `vimeo.com`, `ted.com`, `soundcloud.com` (a 0.13 előtti, csak YouTube-os alaplista magától erre cserélődik; a kézzel szerkesztett lista marad). Ha a központtól más oldalra szóló munka jönne, a futtató visszautasítja. Módosítás: `skipwise-runner config set allowed_domains youtube.com,youtu.be,vimeo.com` (`*` = bármi).

**Oldal mint forrás (0.13):** ha a yt-dlp nem ismeri a beküldött oldalt (például egy Telex-, 444- vagy Index-cikket, ahol a lejátszót JavaScript rakja be), a futtató letölti az oldalt, és maga keresi meg benne a felvételt: elsőként YouTube-videót (annak a felirata is használható), utána a yt-dlp által ismert beágyazott lejátszót (Indavideo, Megaphone), a cikk napjáról való hangfájlt, Apple Podcasts-epizódot, végül egyéb hang- vagy videófájlt. A Spotify védett, azt nem használja. Az oldal szövege háttérként az összefoglalóhoz kerül; a Skipwise elemoldalán látszik, melyik felvételt választotta, és a többiből is kérhető átirat. YouTube-on kívüli fájlt csak az engedélyezett oldalakról vagy az oldal aldomainjéről tölt le. Az Indavideo-videókat (0.13.2) a futtató úgy kéri le, ahogy az Indavideo saját lejátszója, mert a yt-dlp-féle letöltést az Indavideo jelenleg elutasítja (HTTP 403).

## Futás a háttérben

- **Linux szerveren, LXC-ben, VPS-en (rendszerszolgáltatás, ez az ajánlott):**
  ```bash
  wget -O /etc/systemd/system/dwyt-runner.service \
    https://raw.githubusercontent.com/kzkz22/skipwise-runner/main/deploy/dwyt-runner-system.service
  systemctl daemon-reload
  systemctl enable --now dwyt-runner
  systemctl status dwyt-runner        # napló: journalctl -u dwyt-runner -f
  ```
  (Az egysoros telepítő ezt magától megcsinálja, külön felhasználóval.) A fájl rootként futtatja a programot (`/root/.local/bin/skipwise-runner`, beállítások: `/root/.config/dwyt-runner`). Ha másik felhasználóval telepítettél és párosítottál, a `User=`, `Environment=HOME=` és `ExecStart=` sort írd át az ő adataira.
- **Linux asztali gépen (felhasználói szolgáltatás, root nélkül):**
  ```bash
  mkdir -p ~/.config/systemd/user
  wget -O ~/.config/systemd/user/dwyt-runner.service \
    https://raw.githubusercontent.com/kzkz22/skipwise-runner/main/deploy/dwyt-runner.service
  systemctl --user daemon-reload
  systemctl --user enable --now dwyt-runner
  loginctl enable-linger $USER        # hogy kijelentkezés után is fusson
  systemctl --user status dwyt-runner # napló: journalctl --user -u dwyt-runner -f
  ```
  Itt minden parancsban kell a `--user`; a `--user` nélküli `systemctl` a rendszerszolgáltatást keresi, és „could not be found” hibát ad.
- **Windows:** Feladatütemező → Alapfeladat létrehozása → „Bejelentkezéskor” → program: `skipwise-runner`, argumentum: `run`.
- **macOS:** egyelőre egy terminálablakban: `skipwise-runner run`.

## Távoli beállítás

A futtató beállításai a Skipwise Futtatók oldalán, a **Beállítás** gombbal is módosíthatók (`remote_settings`: `auto` | `on` | `off`; automatikusan a `skipwise-runner run` futtatónál be, az asztali alkalmazásban ki, ott a Működés lapon kapcsolható). A futtató elküldi a beállításai leírását, a nem titkos értékeket és egy saját RSA-kulcs nyilvános felét (`remote_key.pem` a beállítási mappában, csak a tulajdonosa olvassa). A kulcsokat (AssemblyAI, LLM) a böngésző ezzel titkosítja, így a Skipwise nem látja őket. Minden értéket a futtató maga ellenőriz, és csak a Skipwise-ból állíthatónak jelölt mezőket fogadja el; a gép saját útvonalai (whisper.cpp helye, sütifájl, munkamappa), a kulcsok helye és maga ez a kapcsoló csak helyben állítható. Minden változás, a gépen tett is, a dwyt-beli naplóba kerül.

**Aláírt frissítés:** a futtató csak a projekt minisign-kulcsával aláírt kiadást telepíti (`latest.json` + aláírás a GitHub-kiadásban; a fájlokat a listában lévő SHA-256 védi). Az asztali alkalmazás magától frissül két munka között (Gépem lap: változat, „Frissítés most”, „Vissza az előzőre”), a szerver csak szól (`auto_update`: auto | on | off). Parancssorból: `skipwise-runner update [--check | --back]`. Kiadás és a kulcs: [docs/kiadas.md](docs/kiadas.md).

**Frissítés most:** ha a GitHubon újabb változat van, a Skipwise-ban a szerveres futtató kártyáján megjelenik a gomb. A futtató két munka között ugyanazt a parancsot futtatja, mint a kézi frissítés, a meglévő kiegészítőkkel, majd újraindul (systemd alatt 75-ös kóddal kilép, és a szolgáltatás elindítja az újat). Aláírás-ellenőrzés még nincs; az a telepítőcsomaggal jön.

## Frissítés

`uv tool install --force --refresh --python 3.13 https://github.com/kzkz22/skipwise-runner/archive/refs/heads/main.zip` (a GitHubról mindig a legfrissebbet telepíti; a `skipwise-runner --version` mutatja a változatot), utána indítsd újra a futtatót. A yt-dlp-t az `uv tool upgrade dwyt-runner` is frissíti; a YouTube gyakran változik, ezért ha a letöltés hibázni kezd, először ezt érdemes futtatni. Ha a központ új protokollt kér, a futtató kiírja, és leáll.

## Fejlesztés

```bash
uv venv && uv pip install -e ".[test]"
uv run pytest -q
```

A tesztek nem érnek el semmilyen külső szolgáltatást: a központ, a yt-dlp, az AssemblyAI és az LLM is utánzat (`tests/`).

## Kódaláírás

A Windows-telepítőt és az általa telepített futtatható fájlokat a SignPath Foundation tanúsítványával írjuk alá (ingyenes kódaláírás nyílt forrású projekteknek, [SignPath.io](https://about.signpath.io)). A szabályzat és az adatkezelés: [docs/code-signing.md](docs/code-signing.md); a kiadás menete: [docs/kiadas.md](docs/kiadas.md).

## Licenc

MIT, lásd [LICENSE](LICENSE).
