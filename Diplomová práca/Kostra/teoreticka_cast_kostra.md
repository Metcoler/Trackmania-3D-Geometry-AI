# Kostra teoretickej časti

Tento dokument je pracovná osnova teoretickej časti diplomovej práce. Je zámerne písaný abstraktne: bez konkrétnych súborov projektu, bez konkrétnych máp, bez presných experimentálnych parametrov a bez výsledkov. Tieto veci patria až do návrhu riešenia a experimentálnej časti.

Úlohou teórie je postaviť slovník a matematický rámec. Čitateľ má po tejto časti rozumieť, čo je agent, prostredie, policy, reward, neurónová sieť, supervised learning, reinforcement learning, genetický algoritmus a viackriteriálna optimalizácia. Až potom má zmysel ukázať, ako tieto pojmy používame v konkrétnom riešení.

## 1. Umelá inteligencia a autonómne rozhodovanie

Táto časť má čitateľa uviesť do problému na najvšeobecnejšej úrovni.

Treba vysvetliť:

- čo rozumieme pod umelou inteligenciou,
- rozdiel medzi pravidlovým systémom a systémom, ktorý sa učí zo skúsenosti,
- že inteligentné správanie nemusí znamenať vedomé myslenie, ale môže znamenať schopnosť voliť vhodné akcie v prostredí,
- autonómne rozhodovanie ako postupné vyberanie akcií na základe pozorovaného stavu,
- autonómne riadenie ako príklad sekvenčného rozhodovania.

Dôležité je nepísať túto časť príliš filozoficky. Má to byť krátky vstup: od všeobecnej umelej inteligencie sa chceme dostať k agentovi, ktorý pozoruje prostredie a rozhoduje sa.

Možné zdroje:

- základné učebnice umelej inteligencie,
- definície inteligentného agenta,
- úvodné zdroje k sekvenčnému rozhodovaniu.

## 2. Agent, prostredie a policy

Tu treba zaviesť základnú terminológiu. Toto je jeden z najdôležitejších teoretických stavebných kameňov celej práce.

Základný model:

- prostredie sa nachádza v stave `s`,
- agent zvolí akciu `a`,
- prostredie prejde do nového stavu `s'`,
- agent tento proces opakuje počas epizódy.

Pojmy, ktoré treba vysvetliť:

- stav prostredia,
- pozorovanie,
- akcia,
- prechodová funkcia,
- krok,
- epizóda,
- trajektória,
- deterministické prostredie,
- nedeterministické prostredie,
- synchrónne prostredie,
- asynchrónne prostredie.

Policy:

- policy je pravidlo, podľa ktorého agent vyberá akciu,
- deterministická policy má tvar `a = π(s)`,
- stochastická policy môže mať tvar `π(a|s)`,
- cieľom učenia agenta je nájsť policy, ktorá sa v prostredí správa dobre podľa zvoleného hodnotenia.

V texte je dobré zdôrazniť rozdiel medzi stavom a pozorovaním. Agent často nevidí celý skutočný stav sveta, ale iba jeho časť alebo spracovanú reprezentáciu. To je všeobecný problém, nie špecifikum konkrétnej implementácie.

Akčný priestor:

- akčný priestor určuje, aké rozhodnutia môže agent vykonať,
- akcie môžu byť diskrétne, napríklad výber z konečnej množiny možností,
- alebo spojité, napríklad hodnota riadenia v určitom intervale,
- v riadiacich úlohách môžu byť niektoré akcie binárne a iné spojité,
- hranice akčného priestoru sú dôležité, lebo policy nesmie produkovať hodnoty, ktoré sa nedajú aplikovať v prostredí.

Closed-loop správanie:

- pri open-loop predikcii model iba odpovedá na dané vstupy,
- pri closed-loop riadení jeho vlastné akcie ovplyvňujú budúce stavy,
- malá chyba v jednej akcii sa preto môže preniesť do ďalších pozorovaní,
- úspešná predikcia na datasete nemusí automaticky znamenať dobré správanie pri samostatnej jazde.

Čas a vzorkovanie:

- sekvenčné rozhodovanie prebieha v krokoch, ale reálne prostredie môže bežať v čase,
- fyzikálny krok môže byť fixný alebo premenlivý,
- oneskorenie, frekvencia rozhodovania a veľkosť časového kroku môžu meniť účinok rovnakej akcie,
- pri realtime prostrediach je preto dôležité rozlišovať medzi herným časom, fyzikálnym krokom a wall-clock časom.

Tieto pojmy budú neskôr dôležité pri vysvetľovaní, prečo rovnaká policy nemusí pôsobiť rovnako v rôznych časovacích podmienkach.

## 3. Hodnotenie správania agenta

Keď agent vykonáva akcie, potrebujeme nejako určiť, či sa správal dobre. Táto kapitola má vysvetliť, že učenie agenta vždy potrebuje nejakú formu hodnotenia.

Treba vysvetliť:

- reward funkciu ako číselné hodnotenie stavu, akcie alebo prechodu,
- celkový návrat cez epizódu,
- maximalizačný problém,
- optimálnu policy `π*`,
- vzťah medzi rewardom, fitness a stratovou funkciou,
- že rôzne metódy učenia môžu používať odlišný jazyk, ale stále nejakým spôsobom optimalizujú správanie.

Odporúčaný abstraktný zápis:

```text
π* = arg maxπ J(π)
```

kde `J(π)` označuje hodnotenie policy.

Treba tiež uviesť problém zle zvoleného hodnotenia. Ak reward alebo fitness nevyjadruje skutočný cieľ, agent môže nájsť formálne dobré, ale prakticky nežiaduce správanie. Toto je dôležitý most k neskoršej reward/ranking časti práce.

Ak je hodnotenie vyjadrené jedným číslom, porovnávanie agentov je jednoduché. Vyššia hodnota znamená lepšiu policy, nižšia horšiu. To je praktické pre algoritmy, ktoré potrebujú kandidátov zoradiť alebo vybrať najlepších jedincov. Problém vznikne vtedy, keď nechceme hodnotiť iba jednu vlastnosť správania. Agent môže byť napríklad rýchly, ale riskantný, alebo bezpečný, ale pomalý. V takom prípade musíme buď viacero metrík spojiť do jedného skalárneho hodnotenia, alebo zaviesť spôsob porovnávania, ktorý s viacerými metrikami pracuje priamo.

Jednokriteriálne a viackriteriálne hodnotenie:

- pri jednom kritériu hľadáme maximum jednej funkcie,
- pri viacerých kritériách môže vzniknúť konflikt,
- nie vždy existuje jednoznačne najlepšie riešenie bez určenia priorít.

## 4. Neurónové siete ako aproximátory funkcií

Táto kapitola má vysvetliť neurónovú sieť od základov, ale bez zbytočného zahltenia detailmi. Cieľ je pripraviť čitateľa na myšlienku, že policy môžeme reprezentovať neurónovou sieťou.

Začať od perceptrónu:

- vstupy,
- váhy,
- bias,
- vážený súčet,
- aktivačná funkcia,
- výstup neurónu.

Potom prejsť na vrstvu:

- vrstva ako viac neurónov naraz,
- maticový zápis,
- výpočet typu `y = f(Wx + b)` alebo ekvivalentne podľa zvolenej orientácie vektorov,
- forward pass ako postupné skladanie vrstiev.

Viacvrstvová neurónová sieť:

- skladanie lineárnych transformácií a nelinearít,
- aktivačné funkcie zavádzajú nelinearitu,
- bez nelinearity by viac vrstiev bolo stále iba lineárne zobrazenie,
- neurónová sieť sa dá chápať ako parametrizovaná funkcia.

Policy ako neurónová sieť:

```text
πθ(s) = a
```

kde `θ` označuje parametre siete, teda váhy a biasy.

Výstup siete:

- výstup neurónovej siete musí zodpovedať typu úlohy,
- pri regresii model predikuje spojitú hodnotu,
- pri klasifikácii model vyberá triedu alebo pravdepodobnosti tried,
- pri policy pre riadenie môže výstup reprezentovať priamo akciu alebo parametre distribúcie akcií,
- výstupné aktivačné funkcie alebo následné orezanie hodnôt zabezpečia, že akcia ostane v povolenom rozsahu.

Normalizácia vstupov:

- neurónové siete sa často trénujú stabilnejšie, keď sú vstupné hodnoty v podobných rozsahoch,
- veľmi veľké rozdiely medzi mierkami vstupov môžu spôsobiť, že niektoré príznaky budú dominovať,
- normalizácia je preto praktický krok medzi surovým pozorovaním a vstupom do modelu,
- vo finálnom texte stačí vysvetliť princíp, nie konkrétne normalizačné konštanty.

Kľúčová myšlienka:

- hľadáme také parametre `θ`, aby sieť pre dané pozorovanie produkovala vhodnú akciu,
- priestor všetkých možných parametrov je vysokorozmerný,
- celý tento priestor nedokážeme prehľadať hrubou silou,
- potrebujeme metódu, ktorá v ňom bude hľadať dobré riešenie.

Tu je vhodné spomenúť univerzálnu aproximáciu iba opatrne: neurónové siete majú silnú aproximačnú schopnosť, ale samotná existencia vhodnej siete neznamená, že ju vieme ľahko nájsť alebo dobre natrénovať.

## 5. Techniky strojového učenia

Táto kapitola má vytvoriť prehľad hlavných prístupov, nie hlbokú učebnicu každého z nich. Dôležité je ukázať, že rôzne metódy odpovedajú na rovnakú otázku iným spôsobom: ako nájsť dobrú funkciu alebo dobrú policy.

### Supervised learning

Supervised learning používa trénovacie dáta vo forme dvojíc vstup-výstup.

Treba vysvetliť:

- dataset,
- vstupné vektory,
- cieľové hodnoty,
- predikciu modelu,
- loss funkciu,
- minimalizáciu loss funkcie.

Obmedzenie:

- model sa učí minimalizovať chybu na dátach,
- nemusí priamo optimalizovať skutočné správanie v prostredí,
- dobrý loss nemusí znamenať dobrú closed-loop jazdu alebo dobré sekvenčné rozhodovanie.

Toto rozlíšenie bude v práci dôležité: model môže na uložených príkladoch predikovať rozumné akcie, ale po spustení v prostredí začne navštevovať stavy, ktoré sám vytvoril svojimi predchádzajúcimi rozhodnutiami.

### Imitation learning

Imitation learning je učenie napodobňovaním experta. V jednoduchom prípade ide o behavior cloning: model sa učí z akcií experta.

Treba vysvetliť:

- expert,
- demonštrácie,
- behavioral cloning,
- distribučný posun,
- prečo malá chyba môže viesť k stavom, ktoré sa v expertných dátach nenachádzali.

Táto časť pripraví neskoršiu diskusiu o tom, prečo samotné napodobňovanie nemusí stačiť.

### Unsupervised learning

Unsupervised learning stačí spomenúť stručne.

Použiť ho ako všeobecný kontext:

- hľadanie štruktúry v dátach bez explicitných labelov,
- zhlukovanie,
- redukcia dimenzie,
- učenie skrytých príznakov.

Nie je potrebné mu venovať veľa priestoru, ak ho práca ďalej nepoužíva ako hlavnú metódu.

### Reinforcement learning

Reinforcement learning sa učí cez interakciu s prostredím. Agent nezískava správnu akciu pre každý stav, ale spätnú väzbu vo forme rewardu.

Treba uviesť:

- agent skúša akcie,
- prostredie reaguje novým stavom a odmenou,
- cieľom je maximalizovať dlhodobý návrat,
- metóda priamo rieši správanie v prostredí, ale môže byť citlivá na reward a prieskum.

### Genetické algoritmy

Genetický algoritmus pracuje s populáciou kandidátnych riešení.

Treba uviesť:

- kandidát,
- fitness,
- selekcia,
- kríženie,
- mutácia,
- opakovanie generácií.

Spoločný pohľad:

- supervised learning optimalizuje chybu voči dátam,
- reinforcement learning optimalizuje správanie cez reward z interakcie,
- genetické algoritmy optimalizujú kandidátov podľa fitness hodnotenia.

## 6. Reinforcement learning

Táto kapitola môže ísť hlbšie než prehľad v predchádzajúcej časti, pretože RL bude neskôr porovnávacou vetvou a zároveň nadväzuje na bakalársku prácu.

Teoretický rámec:

- Markovov rozhodovací proces,
- stav `s`,
- akcia `a`,
- prechodová pravdepodobnosť,
- reward,
- discount faktor,
- návrat.

Dôležité pojmy:

- value funkcia,
- action-value funkcia,
- policy,
- optimálna policy,
- exploration vs exploitation,
- credit assignment problém,
- sparse reward,
- reward shaping.

Policy-gradient metódy:

- policy má parametre,
- učíme tieto parametre tak, aby rástol očakávaný návrat,
- gradient nie je priamo obyčajný supervised gradient z labelov, ale odhad z interakcie.

Actor-critic:

- actor vyberá akcie,
- critic odhaduje hodnotu správania alebo akcií,
- critic pomáha stabilizovať učenie policy.

PPO, SAC a TD3:

- PPO predstaviť ako policy-gradient/actor-critic metódu s obmedzovaním príliš veľkých aktualizácií policy,
- SAC predstaviť ako off-policy actor-critic metódu s entropiou, ktorá podporuje prieskum,
- TD3 predstaviť ako off-policy actor-critic metódu navrhnutú pre kontinuálne akcie a stabilnejšie Q-odhady.

Netreba zachádzať do implementačných detailov knižníc. Cieľom je, aby čitateľ chápal princíp a dôvod, prečo tieto algoritmy patria do porovnania.

## 7. Genetické algoritmy a neuroevolúcia

Táto kapitola pripravuje hlavný optimalizačný smer práce.

Genetický algoritmus:

- populácia kandidátov,
- fitness funkcia,
- výber najlepších alebo pravdepodobnostný výber podľa kvality,
- kríženie rodičov,
- mutácia potomkov,
- elitizmus,
- opakovanie v generáciách.

Treba vysvetliť rozdiel medzi:

- exploráciou: hľadanie nových oblastí priestoru riešení,
- fine-tuningom: jemné dolaďovanie už sľubných riešení.

Neuroevolúcia:

- kandidátom nie je iba jednoduchý vektor akcií,
- kandidátom môžu byť parametre neurónovej siete,
- genóm predstavuje váhy a biasy siete,
- vyhodnotenie kandidáta znamená nechať policy konať v prostredí a ohodnotiť celú epizódu.

Výhoda:

- nepotrebujeme explicitný gradient cez prostredie,
- môžeme optimalizovať aj nehladké alebo nespojité hodnotenia,
- prirodzene vyhodnocujeme celé správanie.

Nevýhoda:

- vyžaduje veľa vyhodnotení,
- môže byť citlivý na veľkosť populácie, mutácie a selekčný tlak,
- bez dobrej fitness funkcie môže konvergovať k nežiaducemu správaniu.

## 8. Viackriteriálna optimalizácia

Táto kapitola má vysvetliť, čo sa stane, keď jedno číslo nestačí.

Motivácia:

- agent môže byť rýchly, ale riskantný,
- agent môže byť bezpečný, ale pomalý,
- agent môže maximalizovať krátkodobý progres, ale zlyhať neskôr,
- rôzne metriky môžu preferovať rôzne riešenia.

Skalárna agregácia:

- viac cieľov sa spojí do jednej hodnoty,
- často cez váhy,
- problémom je výber váh a interpretácia výsledku.

Ak sú jednotlivé metriky normalizované a majú známy rozsah, dá sa nimi vyjadriť aj priorita. Jedna možnosť je použiť váhy rôznych rádov, napríklad mocniny desiatky, aby dôležitejšia metrika prevážila všetky menej dôležité metriky. Pri vhodne zvolených rozsahoch sa tým dá napodobniť lexikografické poradie v jednom čísle. Napríklad najvyššia priorita dostane najväčší rád, ďalšia menší rád a posledná iba jemne dolaďuje výsledok.

Tento prístup je matematicky možný, ale vyžaduje opatrnosť:

- metriky musia byť normalizované alebo aspoň ohraničené,
- váhy musia byť zvolené tak, aby nižšia priorita neprebila vyššiu,
- výsledné číslo môže byť menej čitateľné než explicitné poradie metrík,
- pri zle zvolených váhach sa do hodnotenia môžu dostať skryté “woodoo” konštanty.

Lexikografické poradie:

- ciele sú usporiadané podľa priority,
- druhý cieľ rozhoduje až vtedy, keď je prvý rovnaký alebo nerozlišuje,
- výhodou je čitateľnosť priorít,
- nevýhodou je tvrdosť poradia.

Lexikografické porovnanie preto netreba chápať ako inú metriku, ale ako iný spôsob rozhodovania medzi kandidátmi. Namiesto toho, aby sme všetko najprv zabalili do jedného čísla, porovnávame metriky postupne podľa priority. Tento postup je vhodný vtedy, keď vieme jasne povedať, že určitý cieľ má prednosť pred iným cieľom.

Pareto optimalizácia:

- riešenie dominuje iné riešenie, ak je aspoň také dobré vo všetkých cieľoch a lepšie aspoň v jednom,
- Pareto fronta obsahuje nedominované riešenia,
- nemusí existovať jedno najlepšie riešenie bez dodatočných preferencií.

NSGA-II:

- príklad evolučného algoritmu pre viackriteriálnu optimalizáciu,
- používa non-dominated sorting,
- používa diverzitu riešení, napríklad crowding distance,
- je vhodný, keď chceme skúmať kompromisy medzi cieľmi.

## 9. Simulácie a virtuálne prostredia

Táto kapitola má stručne vysvetliť, prečo pri učení a vyhodnocovaní agentov potrebujeme prostredie, v ktorom možno správanie opakovane skúšať. Nemá ísť o širokú úvahu o sim-to-real prenose. V tejto práci stačí pripraviť pojmy, ktoré neskôr použijeme pri hre, lokálnych experimentoch a realtime vyhodnocovaní.

Jadro kapitoly:

- simulácia alebo virtuálne prostredie umožňuje spustiť agenta, aplikovať akcie a merať dôsledky jeho správania,
- výhodou je bezpečnosť, opakovateľnosť a nižšia cena veľkého množstva epizód,
- pri porovnávaní politík je dôležité, aby prostredie poskytovalo konzistentné podmienky a logovateľné metriky,
- netreba tu opisovať konkrétny 2D sandbox, mapový export ani implementáciu.

Ponechať iba pojmy, ktoré sú priamo potrebné:

- deterministické prostredie: rovnaký stav a akcia vedú k rovnakému ďalšiemu stavu,
- stochastické alebo prakticky variabilné prostredie: výsledok môže ovplyvniť šum, neistota, časovanie alebo vonkajšie podmienky,
- synchrónne prostredie: rozhodovanie prebieha v pravidelných krokoch,
- realtime/asynchrónne prostredie: dĺžka kroku alebo čas medzi akciami môže ovplyvniť účinok rovnakej akcie.

Vyhnúť sa v tejto kapitole:

- dlhému výkladu doménovej medzery,
- sim-to-real terminológii, ak ju priamo nepotrebujeme,
- tvrdeniam o prenose medzi simuláciou a cieľovým prostredím, ktoré budú patriť až do návrhu riešenia alebo experimentov.

## 10. Trackmania ako prostredie pre agenta

Pred súvisiacimi prácami je vhodné krátko vysvetliť, čo je Trackmania a prečo ju v práci chápeme ako cieľové virtuálne prostredie. Táto časť má byť krátka orientácia pre čitateľa, nie opis našej implementácie ani katalóg herných prvkov.

Treba vysvetliť:

- Trackmania je pretekárska hra orientovaná na čo najrýchlejší prejazd trate,
- pri time-attack úlohe je cieľom čas a trajektória, nie priama interakcia so súpermi,
- z pohľadu agenta ide o sekvenčné rozhodovanie v reálnom čase,
- rovnaký vstup môže mať odlišný účinok podľa rýchlosti, polohy, natočenia auta a časovania,
- malé odchýlky v trajektórii môžu ovplyvniť ďalší priebeh jazdy,
- trate môžu byť stavebnicové a obsahovať rôzne tvary, výšky alebo povrchy, ale v teórii to stačí spomenúť len veľmi stručne.

Nepísať tu:

- OpenPlanet, export mapy, lidar, observation vektor ani akčný priestor našej implementácie,
- konkrétne tréningové mapy,
- výsledky experimentov,
- detailný opis editoru tratí alebo herných režimov.

Vizuál:

- naplánovať jeden ilustračný obrázok alebo TODO: náhľad trate s prejazdom/trajektóriou a časom,
- ideálne použiť neskôr existujúci náhľad zo supervised dát alebo z náhľadového skriptu,
- obrázok má ukázať pojem trajektórie a time-attack úlohy, nie architektúru nášho agenta.

## 11. Krátky slovník riadenia vozidla a pretekárskej jazdy

Táto časť má byť veľmi krátka. Nemá vysvetľovať fyziku auta ani bežné pojmy na úrovni učebnice. Má iba pomenovať tie vlastnosti jazdy, ktoré budeme neskôr používať pri interpretácii správania agenta.

Ponechať:

- rýchlosť, brzdenie a zatáčanie ako základné veličiny a akcie relevantné pre jazdu,
- trakciu a povrch ako dôvod, prečo rovnaká akcia nemusí mať vždy rovnaký účinok,
- šmyk alebo odlišný smer pohybu voči natočeniu auta, ak sa bude neskôr používať pri vysvetľovaní správania,
- trajektóriu ako kľúčový pojem: kvalitu jazdy hodnotíme na celom priebehu, nie podľa jednej okamžitej akcie.

Nepísať:

- školské definície rýchlosti alebo zrýchlenia,
- vzorce, ktoré nepomáhajú ďalšiemu výkladu,
- podrobnú dynamiku vozidla.

Ak bude teoretická časť príliš dlhá, túto sekciu možno zlúčiť s Trackmania sekciou do niekoľkých viet. Ak ostane samostatná, mala by mať približne 2 až 3 odseky a jednu oporu v literatúre k dynamike vozidiel.

## 12. Základy počítačovej grafiky pre virtuálne senzory

Táto kapitola má ostať, ale treba ju preformulovať okolo hlavnej myšlienky: virtuálny svet možno reprezentovať geometriou a z tejto geometrie možno počítať kompaktné pozorovania pomocou lúčov. Nemá ísť o učebnicové vysvetľovanie súradníc, bodov a vektorov.

Hlavná argumentačná niť:

- virtuálne prostredie nemusí byť pre agenta reprezentované iba obrazom,
- herný alebo simulovaný svet môže mať geometrickú reprezentáciu,
- bežnou reprezentáciou povrchov je trojuholníková sieť,
- ak poznáme geometriu, môžeme sa pýtať na priestorové vzťahy matematicky,
- raycasting hľadá prienik lúča s geometriou,
- výsledkom raycastingu môže byť vzdialenosť k najbližšej prekážke alebo povrchu,
- viac takýchto vzdialeností môže tvoriť virtuálny senzor pre agenta.

Formálny zápis ponechať iba tam, kde pomáha:

- lúč ako `r(t) = o + td`,
- prípadne lokálna/globálna transformácia iba jednou vetou alebo vzorcom, ak je potrebná pre pochopenie objektov v priestore,
- nevysvetľovať podrobne, čo je súradnica `x`, `y`, `z`.

Obrázky:

- ponechať obrázok trojuholníkovej siete Stanford Bunny ako ilustračný príklad mesh reprezentácie,
- ponechať samostatný detail raycastingu ako prienik lúča s trojuholníkom,
- pri bunny obrázku uviesť zdroj Stanford 3D Scanning Repository,
- obrázky majú niesť časť vysvetlenia, aby text nemusel opisovať samozrejmé 3D základy.

Zdôrazniť výhodu a hranicu geometrických senzorov:

- výhodou je kompaktný a interpretovateľný vstup namiesto celého obrazu,
- hranicou je závislosť od dostupnej a dostatočne presnej geometrie,
- konkrétne získanie geometrie z Trackmanie a náš spôsob výpočtu senzorov patria až do návrhu riešenia.

## 13. Skôr netreba uvádzať - Metriky a experimentálne vyhodnotenie 

Teória musí pripraviť aj spôsob, akým budeme neskôr čítať výsledky. Nestačí ukázať jednu úspešnú jazdu.

Treba vysvetliť:

- čo je metrika,
- rozdiel medzi tréningovou a testovacou metrikou,
- úspešnosť,
- kvalita riešenia,
- stabilita správania,
- rýchlosť riešenia,
- rozptyl výsledkov.

Štatistické pojmy:

- priemer,
- maximum,
- minimum,
- medián,
- percentily,
- smerodajná odchýlka alebo rozptyl,
- trend v čase.

Tréningové krivky:

- ukazujú vývoj kvality počas učenia,
- treba interpretovať nielen najlepší bod, ale aj stabilitu,
- jednorazový úspech nemusí znamenať robustnú policy.

Reprodukovateľnosť:

- experiment má mať opísané podmienky,
- treba rozlišovať medzi jedným behom a opakovaným výsledkom,
- negatívny výsledok je užitočný, ak je dobre zdokumentovaný.

Generalizácia:

- tréningový výkon ukazuje, čo sa model naučil v podmienkach, v ktorých bol optimalizovaný,
- testovací alebo validačný výkon ukazuje, či sa správanie prenáša aj mimo týchto podmienok,
- dobrý výsledok na jednej trati nemusí automaticky znamenať všeobecnú schopnosť jazdiť ľubovoľnú trať,
- pri generalizácii treba rozlišovať medzi tým, že agent trať dokončí, jazdí ju stabilne a jazdí ju rýchlo.

Vizualizácia správania:

- pri sekvenčnom riadení často nestačia iba tabuľkové metriky,
- trajektória ukazuje, kadiaľ agent išiel a kde vznikali chyby,
- heatmapy, rýchlostne zafarbené trasy a crash body môžu pomôcť vysvetliť správanie, ktoré jedno číslo nezachytí.

Táto kapitola má prirodzene uzavrieť teoretickú časť a otvoriť cestu k návrhu riešenia: už vieme, čo je agent, policy, hodnotenie, neurónová sieť, metódy učenia a ako budeme výsledky posudzovať.

## Odporúčané poradie v texte

Najprirodzenejšie poradie je:

1. najprv vysvetliť agenta a prostredie,
2. potom hodnotenie správania,
3. potom neurónovú sieť ako reprezentáciu policy,
4. potom metódy učenia, ktoré hľadajú dobrú policy,
5. nakoniec viackriteriálne hodnotenie, simulácie, Trackmaniu ako prostredie, voliteľne základnú jazdnú dynamiku, voliteľne počítačovú grafiku pre virtuálne senzory a experimentálne vyhodnotenie.

Takto čitateľ pochopí, že neurónová sieť nie je cieľ sama o sebe. Je to iba reprezentácia rozhodovacej funkcie. Skutočným problémom je nájsť také parametre tejto funkcie, aby sa agent v prostredí správal dobre.

## Čo do teoretickej časti ešte nepatrí

Do čistej teórie zatiaľ nepatrí:

- názvy konkrétnych súborov projektu,
- konkrétne mapy,
- konkrétne hodnoty architektúr neurónových sietí,
- konkrétne ranking tuple z experimentov,
- detailný opis mapového exportu,
- detailná implementácia lidaru,
- logging, checkpointy a technické detaily trénera,
- výsledky experimentov,
- finálna konfigurácia tréningu.

Tieto veci patria do návrhu riešenia, implementácie a experimentálnej časti.

## Citácie, ktoré bude treba dohľadať

Pri písaní tejto kapitoly bude potrebné nájsť a korektne citovať zdroje pre:

- základné definície umelej inteligencie a inteligentných agentov,
- Markovove rozhodovacie procesy a reinforcement learning,
- neurónové siete a univerzálnu aproximáciu,
- supervised learning,
- imitation learning a behavior cloning,
- DAgger alebo podobné prístupy k distribučnému posunu,
- genetické algoritmy,
- neuroevolúciu,
- NSGA-II a Pareto optimalizáciu,
- reward shaping,
- simulácie a virtuálne prostredia pre autonómne riadenie alebo autonomous racing,
- Trackmaniu ako pretekársku hru, editor tratí a time-attack prostredie,
- základné pojmy riadenia vozidla, trakcie a pretekárskej jazdy, ak túto voliteľnú časť zaradíme,
- základy počítačovej grafiky, 3D meshe, raycasting a virtuálne senzory,
- experimentálne vyhodnocovanie strojového učenia.

Nevymýšľať citácie. Pri písaní finálneho textu treba ku každému netriviálnemu teoretickému tvrdeniu dohľadať zdroj a uložiť ho do bibliografie.

## Konkrétne zdrojové kotvy

Prvé stabilné zdrojové kotvy sú pripravené v `Diplomová práca/Latex/references.bib`. Pri písaní teórie sa oplatí začať týmito zdrojmi:

- umelá inteligencia a agenti: `russell_norvig_aima2021`, voliteľne historicky `turing1950computing`,
- neurónové siete a univerzálna aproximácia: `goodfellow2016deep`, `cybenko1989approximation`,
- Markovove rozhodovacie procesy a učenie posilňovaním: `sutton_barto2018rl`,
- PPO, SAC, TD3 a praktická RL knižnica: `schulman2017ppo`, `haarnoja2018sac`, `fujimoto2018td3`, `raffin2021sb3`,
- behavior cloning a imitation learning: `pomerleau1989alvinn`, `bojarski2016endtoend`, `ross2011dagger`,
- genetické algoritmy, neuroevolúcia a viackriteriálna optimalizácia: `goldberg1989ga`, `stanley2002neat`, `deb2002nsga2`, `coello1999survey`,
- reward shaping a zle navrhnuté odmeny: `ng1999shaping`, `knox2023reward_misdesign`,
- AI v hrách: `yannakakis_togelius2018aigames`, `millington_funge2019ai_for_games`, `iovino2022behavior_trees`,
- počítačová grafika, 3D geometria a raycasting: `hughes2014computer_graphics`, `moller1997raytriangle`, `dunn2011mathprimer`.

Tieto kotvy nie sú povinné citovať všetky. Slúžia ako bezpečný začiatok, aby sme pri písaní kapitoly nesiahali po slabších internetových zdrojoch tam, kde existuje štandardná literatúra.

## Pracovný verdikt

Táto kostra je úmyselne abstraktnejšia než návrh riešenia. Jej cieľom je pripraviť čitateľa na neskoršie kapitoly bez toho, aby sme ho hneď zatiahli do detailov konkrétnej implementácie.

V ďalších kapitolách potom môžeme povedať: máme agenta, máme prostredie, máme policy reprezentovanú neurónovou sieťou, máme hodnotenie správania a máme viac spôsobov, ako hľadať vhodné parametre tejto policy. Až tam začne náš konkrétny prípad Trackmanie.
