# A Skipwise futtató Windowsra

Windows 10 vagy 11, 64 bites. Rendszergazdai jog nem kell.

## Telepítés

1. A Skipwise **Futtatók** oldalán: „Új futtató” → **Asztali gépre** → **Windows-telepítő letöltése** (`skipwise-runner-setup.exe`, kb. 70 MB).
2. Indítsd el. A telepítő még nincs kódaláírással ellátva, ezért a Windows figyelmeztethet („A Windows megvédte a számítógépet”): **További információ** → **Futtatás mindenképp**. Ha a géped céges házirendje vagy az Intelligens alkalmazásvezérlés ezt nem engedi, a telepítő nem fog elindulni; ilyenkor egyelőre a [szerveres](szerver.md) változat marad.
3. A telepítő a saját mappádba telepít (`%LOCALAPPDATA%\Programs\Skipwise`; a 0.17 előtt telepített futtató a régi `Programs\dwyt-runner` mappában marad), és a Start menübe tesz egy „Skipwise futtató” ikont. A végén elindítja a futtatót.

## Az első indítás

A futtató ablaka lépésenként végigvezet:

1. **Üdvözlés.**
2. **A géped:** videókártya, memória, szabad hely; a **Próbafuttatás** egy 33 másodperces mintát ír át, és egy rövid összefoglalót készít, majd lámpákkal megmondja, mire elég a gép.
3. **Működési mód:** Felhő, Vegyes, Saját gép vagy Egyéni (lásd az [útmutatók elején](README.md)).
4. **Hozzávalók:** csak ami a választott módhoz kell. Például a beszédfelismerő letöltése (kb. 3,5 GB, egyszer), az OpenRouter- vagy AssemblyAI-kulcs (a futtató azonnal kipróbálja; Felhő módban kiválasztod, ki írja az összefoglalót: az AssemblyAI próba-modellje ugyanazzal a kulccsal, vagy az OpenRouter), a helyi modellszerver (Ollama / LM Studio) elindítása és egy modell letöltése. A kulcsokat a Windows Hitelesítőadat-kezelője őrzi; a Skipwise nem látja őket.
5. **Párosítás:** a futtató megnyitja a Skipwise Futtatók oldalát a kóddal kitöltve. Nézd meg, hogy a kód ugyanaz-e, mint a futtató ablakában, és nyomd meg a **Jóváhagyás** gombot.
6. **Kész:** ha szeretnéd, a futtató bejelentkezéskor magától elindul (csak a tálcaikon, ablak nélkül).

## Mindennap

- A **tálcaikon** színe: zöld = vár, kék = dolgozik, narancs = a videókártyáról éppen kiüríti a nyelvi modellt, piros = leállt (erről értesítés is jön), szürke = nincs párosítva. Ha nem látod, a tálca „^” jele alatt van; a Windows beállításaiban kitűzheted.
- Az ablak bezárása csak elrejti: a futtató a háttérben dolgozik tovább. Kilépni a tálcaikon menüjéből vagy az ablak **Kilépés** gombjával lehet.
- **Állapot** lap: mi fut most, mi készült el. **Működés:** mód, indulás bejelentkezéskor, részletes beállítások, és hogy a Skipwise-ból is állítható legyen-e. **Gépem:** a gép, a próbafuttatás, a yt-dlp és a futtató változata. **Kulcsok:** a kulcsok megadása és kipróbálása.

## Diák a videókból (nem kötelező)

A diák szövegét kiolvasó Tesseract a telepítővel együtt érkezik (0.24.1-től), és a futtató frissítésével együtt frissül; külön telepíteni nem kell. A nyelvi adatokat (magyar, angol és a felvétel nyelve) a futtató az első diás munkánál tölti le, néhány MB nyelvenként. Ha már telepítetted a UB Mannheim-féle Tesseractot, az sem zavar: a futtató a saját példányát használja.

## Frissítés

A futtató naponta megnézi, van-e újabb változat, és két munka között magától telepíti. Csak a Skipwise kiadói kulcsával aláírt kiadást fogadja el. A **Gépem** lapon látszik a változat; ott kérheted azonnal is („Frissítés most”), és ott állhatsz vissza az előzőre. A beállítások, kulcsok és modellek megmaradnak.

## Eltávolítás

Gépház → Alkalmazások → „Skipwise futtató” → Eltávolítás. A bejelentkezéskori indulás mindig törlődik. A program megkérdezi, törölje-e a beállításokat, a kulcsokat és a letöltött modelleket is (alapból nem, hogy egy újratelepítés ne vigye el őket). A Skipwise-ban a futtatót a Futtatók oldalon vond vissza.

Ha mindent elölről kezdenél eltávolítás nélkül: **Működés** lap → **Alaphelyzet**.

## Ha valami nem megy

| Jelenség | Mit tegyél |
| --- | --- |
| Az ablak helyett a böngésző nyílik meg | A WebView2 hiányzik (Windows 10-en előfordul): telepítsd a Microsoft „WebView2 Runtime” csomagját, vagy maradj a böngészős ablaknál, az is teljes értékű. |
| A helyi átírásnál „az alkalmazásvezérlés letiltotta” | A Windows nem engedi az aláíratlan whisper.cpp-t. A futtató ezt nem kerüli meg; válaszd a Felhő módot, vagy a Windows biztonsági beállításaiban engedélyezd. |
| A tálcaikon piros | Nyisd meg az ablakot: az Állapot lap megmondja, mi történt (például visszavonták a futtatót, vagy a Skipwise nem érhető el). |
| Naplófájl kell | `%APPDATA%\dwyt-runner\app.log` |
