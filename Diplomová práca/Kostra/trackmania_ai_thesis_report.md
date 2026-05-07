# Trackmania AI Thesis Report

Tento report sumarizuje kontext bakalárskej práce, posudkov, aktuálneho zadania diplomovej práce a súčasného stavu projektu. Slúži ako pracovný podklad pre písanie diplomovej práce a ako zdroj pre Codex skill `trackmania-ai-thesis`.

## Prečítané zdroje

- `Bakalárska práca/zadanie_bakalarska.PDF`
- `Bakalárska práca/posudok_skolitel.pdf`
- `Bakalárska práca/posudok_oponent.pdf`
- `Bakalárska práca/latex project/*.tex`
- `Bakalárska práca/Implementacia/*.py`
- `Bakalárska práca/Implementacia/Plugins/get_data_driver/main.as`
- `Diplomová práca/zadanie_diplomova.PDF`
- `Diplomová práca/Latex vzor/*`
- `CODEX_PROJECT_CONTEXT.md`
- `https://github.com/davidmisiak/thesis-template`

Video `Bakalárska práca/Implementacia/agent_driver.mp4` sa nepodarilo dekódovať dostupnými lokálnymi nástrojmi. V texte práce ho preto netreba používať ako tvrdý zdroj výsledkov, iba ako prípadný vizuálny doplnok, ak ho neskôr manuálne otvoríme a zdokumentujeme.

## Zadanie bakalárskej práce

Bakalárska práca mala názov `Trénovanie autonómneho vozidla v hre Trackmania pomocou strojového učenia`.

Hlavné ciele:

- oboznámiť sa s virtuálnym prostredím Trackmania
- preskúmať spôsoby reprezentácie stavu sveta
- navrhnúť, implementovať a porovnať viacero agentov pre riadenie autonómneho vozidla
- pracovať s rôznymi metódami strojového učenia

Dôležitý pôvodný motív bol správny: Trackmania je bezpečné virtuálne prostredie, OpenPlanet umožňuje získať živé údaje o aute a agent môže dostať iný vstup než iba obraz z obrazovky.

## Zadanie diplomovej práce

Diplomová práca má názov `Autonómne jazdiaci agent pre hru Trackmania`.

Hlavné ciele podľa zadania:

- nadviazať na bakalársku prácu
- navrhnúť a implementovať lepšiu verziu autonómne jazdiaceho agenta
- zlepšiť samotnú jazdu agenta a priblížiť sa časom ľudským hráčom
- plne automatizovať tréning bez zásahov človeka
- podporiť trate s vertikálnymi zmenami, kopcami a svahmi
- podporiť rôzne povrchy, napríklad asfalt, hlinu a trávu
- empiricky vyhodnotiť kvalitu jazdy vhodnou metrikou
- porovnať nové riešenie s pôvodnou verziou agenta

Diplomovka teda nemá byť len ďalšia implementácia. Musí ukázať merateľný posun oproti bakalárskej práci: lepšia automatizácia, lepšia reprezentácia prostredia, lepšie experimenty, lepšie metriky a dôveryhodnejšie závery.

## Čo bolo v bakalárskej implementácii

Stará implementácia obsahovala základné stavebné kamene, ktoré sú stále rozpoznateľné v dnešnom projekte.

### Map extraction

V priečinku `Bakalárska práca/Implementacia/Map Extractor C#` bol C# exportér mapových blokov z `.Map.Gbx` súborov. Dnešný projekt stále používa rovnakú ideu: mapu z Trackmanie previesť na textový zoznam blokov, pozícií a rotácií.

### Map.py

Stará `Map.py` načítavala `Maps/ExportedBlocks/*.txt`, skladala 3D bloky z `.obj` súborov, rozdeľovala mesh na cestu a steny podľa normál a konštruovala logickú cestu od štartu do cieľa.

Toto je priamy predchodca dnešnej `Map.py`. Dnešná verzia je väčšia, podporuje viac blokov, výškové rozdiely, povrchy, tile-aligned path instructions a centrálne cesty cez `ProjectPaths.py`, ale základný koncept ostal rovnaký.

### Car.py

Stará `Car.py`:

- prijímala OpenPlanet TCP dáta na porte `9002`
- uchovávala aktuálnu pozíciu, smer, rýchlosť a side speed
- simulovala 15 lidar lúčov v 180-stupňovom rozsahu
- používala `SIGHT_TILES = 10`
- počítala `total_progress`
- používala `next_point_direction` ako dot product medzi smerom auta a ďalším segmentom trate

Dnešná `Car.py` je oveľa robustnejšia. Pridali sme najnovší packet-only model, dense progress, surface instructions, height instructions, slip mean, accel/yaw metriky, heading error so znamienkom, 5 path instructions, 160-unit lidar a vertikálny block-grid lidar.

### Enviroment.py

Stará `Enviroment.py` bola Gym-like wrapper:

- aplikovala akcie cez `vgamepad`
- používala `Box` observation a action space
- resetovala hru jednoduchým stlačením `B`
- mala ručne skladaný reward
- používala `race_terminated` s hodnotami `1`, `0`, `-1`
- končila epizódu pri finishi alebo pri veľkom odklone `next_point_direction < -0.5`

Dnešná `Enviroment.py` má rovnakú rolu, ale dôležito sa zmenila: reset je potvrdený handshakeom cez záporný herný čas, reward sa presunul preč z prostredia, stav epizódy sa rozdelil na `finished` a `crashes`, podporuje sa `max_touches`, timeout, stall, wall-ride guard a konzistentné výpisy pre live debug.

### Training.py

Stará `Training.py` používala Stable-Baselines3 PPO:

- `PPO("MlpPolicy", env, ...)`
- `n_steps = RacingGameEnviroment.STEPS`
- `learning_rate = 0.000001`
- checkpoint každých `100_000` timestepov
- trénovanie na `10_000_000` timestepov

Toto je dôležité historicky: bakalárska práca tvrdila RL/PPO smer, ale text neobhájil dostatočne reward funkciu, parametre, fázy ani výsledky. Dnešný projekt preto musí experimenty zapisovať oveľa presnejšie.

### Actor.py a Driver.py

Stará `Actor.py` zbierala supervised dáta z Xbox ovládača a live observation. Stará `Driver.py` načítala PPO model a jazdila s ním v Trackmanii.

Dnešný projekt túto dvojicu zachoval významovo:

- `Actor.py` zbiera supervised dáta
- `SupervisedTraining.py` trénuje torch policy
- `Driver.py` prehráva model v Trackmanii
- `GeneticTrainer.py` vie použiť supervised model ako seed pre GA

## Čo sa v bakalárskej práci podarilo

Technická myšlienka bola dobrá:

- nepozerať sa len na obraz z hry
- použiť OpenPlanet na živé dáta
- exportovať mapu a pracovať s geometriou
- simulovať lidar nad virtuálnou mapou
- posielať agentovi lokálne aj globálne informácie o trati
- uvedomiť si problém krátkeho lokálneho horizontu
- porovnať Trackmania AI, TMRL a Sophy-like princípy

Práca už pomenovala veľa problémov, ktoré sú dnes jadrom diplomovky:

- spracovanie obrazu je pomalé
- lokálne vnímanie nestačí
- agent potrebuje informácie o nasledujúcich zákrutách
- Trackmania je realtime/asynchrónne prostredie
- tréning v živej hre je časovo drahý
- 2D reprezentácia má limity pri výškových rozdieloch

## Čo bolo v bakalárskej práci slabé

Z posudkov a textu vyplývajú opakované problémy.

### Nedostatočne popísaná metóda učenia

Oba posudky vytkli, že práca používa PPO, ale nevysvetľuje ho dostatočne. V diplomovke nesmie vzniknúť rovnaká chyba. Ak použijeme GA, SAC alebo supervised learning, musíme popísať:

- čo presne algoritmus optimalizuje
- aký je vstup a výstup siete
- ako vzniká akcia
- ako sa vyhodnocuje jedinec alebo epizóda
- aké parametre boli použité
- prečo boli zvolené práve tieto hodnoty

### Slabé výsledky experimentov

Posudky explicitne kritizujú, že práca spomína experimenty s počtom lúčov a learning rate, ale výsledky neuvádza. Toto je najväčší poučný bod.

V diplomovke treba pri každom experimente uvádzať:

- mapu
- počet generácií/epizód
- populáciu alebo počet epizód
- architektúru siete
- observation dimension
- reward/ranking funkciu
- mutation/SAC parametre
- max time
- collision mode
- počet behov/seeds, ak relevantné
- wall-clock čas
- virtuálny herný čas
- grafy progresu, finish rate, času, crash rate a variability

### Prehnane silné tvrdenia

Bakalárska práca tvrdila konkurenčný výkon, ale autor tomu spätne neverí a vyhodnotenie bolo slabé. V diplomovke treba písať opatrnejšie:

- netvrdiť “konkurenčný výkon”, ak nemáme férové porovnanie
- rozlišovať kvalitatívnu ukážku od kvantitatívneho experimentu
- priznať zlyhané vetvy, napríklad SAC alebo starý vertical lidar, ak zlyhali
- ukázať, čo presne je zlepšenie oproti bakalárskej verzii

### Formálne problémy

Posudky spomínajú:

- zabudnuté TODO v texte
- nekonzistentnú skratku `TMRL` / `TMLR`
- preklepy a gramatické chyby
- chýbajúce odkazy na obrázky
- nejednotné citácie
- priveľa priestoru pre NEAT, ktorý sa ďalej nepoužil
- málo všeobecného úvodu do evolučných algoritmov

Diplomovka by mala mať finálnu kontrolu:

- žiadne TODO
- jednotné názvy: Trackmania, OpenPlanet, TMRL, SAC, GA
- každý obrázok musí byť citovaný v texte
- každý graf musí mať jasnú interpretáciu
- bibliografia musí byť kompletná a jednotná

## Aktuálny projekt oproti bakalárskej verzii

Dnešný projekt je výrazne väčší a zrelší. Najdôležitejšie posuny:

### Názvoslovie

Aktuálne pomenovania:

- `NeuralPolicy.py` pre spoločnú neurónovú sieť
- `Individual.py` pre jedinca/genóm
- `GeneticTrainer.py` pre live GA tréning
- `ObservationEncoder.py` pre normalizáciu observation
- `Experiments/` pre lokálnu 2D simuláciu

V texte diplomovky odporúčané pomenovanie:

- primárne používať `genetický algoritmus (GA)`
- vysvetliť, že GA patrí medzi evolučné algoritmy
- keďže genóm sú váhy neurónovej siete, ide o formu neuroevolúcie

### Observation

Aktuálna canonical observation má 53 vstupov pri `vertical_mode=True`.

Obsahuje:

- 15 lidar vzdialeností
- 5 path instructions ako signed curvature
- speed
- side speed
- current/next segment heading error
- dt ratio
- slip mean
- 5 surface traction instructions
- 5 height instructions
- longitudinal acceleration
- lateral acceleration
- yaw rate
- 5 clearance-rate sectors
- vertical speed
- forward_y
- support_normal_y
- cross_slope
- 5 surface elevation sectors

Flat debug observation má 44 vstupov.

Toto je veľký posun oproti bakalárskej verzii, ktorá používala 15 lidarov, 10 path instructions, speed, side_speed, next_point_direction a previous action.

### Path instructions

Historicky boli path instructions interpretované ako veľkosť zákruty. Dnes reprezentujú signed curvature:

- straight = 0
- Curve1 = ±1.0
- Curve2 = ±0.5
- Curve3 = ±0.333
- Curve4 = ±0.25

To je fyzikálne zmysluplnejšie, lebo Curve1 je najostrejšia zákruta.

### Progress metriky

Stará bakalárska verzia používala `total_progress` a `map_progress`. Dnešný projekt tieto historické aliasy odstraňuje.

Aktuálne metriky:

- `discrete_progress`: potvrdený progres po path tiles
- `dense_progress`: plynulý progres medzi path tiles získaný projekciou auta na segment stredovej osi

Pre reward/ranking experimenty je dense progress kľúčový, lebo zmenšuje problém sparse progress bucketov.

### Outcome metriky

Starý `term` bol trojhodnotový:

- `1` finish
- `0` timeout/running
- `-1` crash

Dnes sa metrika rozdeľuje:

- `finished = 1/0`
- `crashes = počet crashov/touchov`

Timeout sa odvádza ako `finished == 0 and crashes == 0`.

Toto je čistejšie pre lexikografické aj multiobjective hodnotenie.

### GA

Dnešný GA obsahuje:

- explicitné ranking tuple výrazy, napríklad `(finished, progress, -time, -crashes, -distance)`
- dense progress ranking
- arithmetic-mean crossover
- mutation probability a mutation sigma
- annealing mutation parametrov
- checkpointy vrátane mutation state
- elitism
- elite cache
- no-repeat parent pairing
- mirror evaluation
- max touches
- robustnejší reset Trackmanie
- detailné per-generation a per-individual logy

Toto je vhodné jadro diplomovky, lebo rieši presne problém drahého realtime prostredia.

### SAC/RL

Dnešný projekt obsahuje dva RL smery:

- vlastný run-based REINFORCE prototyp
- Stable-Baselines3 SAC v `RL_test/` a lokálny SAC sandbox v `Experiments/`

Z experimentov zatiaľ vyplýva:

- SAC v live Trackmanii je citlivý na reward a exploration
- terminálny reward je pre SAC často príliš sparse
- dense delta reward môže pomôcť, ale môže vytvoriť kamikaze správanie
- SAC je vhodné uviesť ako skúmanú alternatívu, nie ako hlavný úspešný výsledok, kým nemáme stabilné dôkazy

### Lokálna 2D simulácia

`Experiments/` je zásadný nový príspevok oproti bakalárskej práci.

Motivácia:

- Trackmania je realtime
- beží iba jedna inštancia
- nedá sa jednoducho spustiť headless a paralelne
- experimenty v živej hre trvajú hodiny

Lokálna simulácia umožňuje:

- rýchlo testovať reward funkcie
- porovnať GA a SAC
- spúšťať populáciu paralelne
- logovať per-individual metriky
- merať virtuálny herný čas
- robiť grafy vhodné do práce

Toto by malo byť v diplomovke prezentované ako metodologický nástroj, nie ako náhrada reálnej Trackmanie.

### Povrchy

Dnešný projekt pridal surface traction instructions:

- RoadTech/asphalt: 1.00
- Grass: 0.70
- Plastic: 0.75
- Dirt: 0.50
- Ice: 0.05
- Snow: 0.15

Tieto hodnoty sú kompaktný proxy vstup pre rozdielne povrchy. Nie je to fyzikálne presné meranie trakcie, ale praktická reprezentácia pre policy.

### Výškové rozdiely

Bakalárska práca navrhovala 3D mapu, ale reálne ostala hlavne v rovine. Diplomovka má dnes konkrétnejšiu implementáciu:

- block-grid surface-following lidar
- fitted road planes per block
- vertical-mode fallback na 2D raycast, keď je trať plochá
- slope blocks vrátane 1x1, 2x1 a curve slope variantov
- height instructions v lookahead

Dôležitá história: prvý vertikálny prototyp s fixed-step marching bol zahodený, lebo bol pomalý a nespoľahlivý. Aktuálne riešenie je jednoduchšie, rýchlejšie a viac využíva štruktúru Trackmania blokov.

## Poučenia z posudkov pre diplomovku

Diplomovka musí priamo odpovedať na kritiku bakalárskej práce:

### 1. Každý použitý algoritmus musí byť vysvetlený

Ak používame GA, treba vysvetliť:

- populácia
- genóm
- fitness/ranking
- elitizmus
- selection
- crossover
- mutation
- stopping condition
- checkpointing

Ak používame SAC, treba vysvetliť:

- actor/critic
- off-policy replay
- entropy
- gamma
- episode-based update v našom real-time prostredí
- prečo SAC môže byť problémový pri sparse rewarde

### 2. Reward/ranking funkcie musia byť formálne definované

Nestačí napísať “odmeňovali sme progres a čas”. Treba uviesť presnú funkciu:

```python
return (finished, progress, -time, -crashes, -distance)
```

a vysvetliť lexikografické poradie. Pri multiobjective GA treba uviesť objective vector, dominanciu a tiebreak.

### 3. Experimenty musia byť reprodukovateľné

Pri každom experimente uviesť:

- príkaz alebo config
- mapu
- seed
- FPS režim
- collision mode
- počet generácií/epizód
- počet jedincov
- architektúru siete
- action space
- observation mode
- reward/ranking
- trvanie wall-clock
- virtuálny herný čas

### 4. Výsledky musia byť interpretované, nie iba ukázané

Každý graf potrebuje odpovedať:

- čo sa meria
- čo znamená vyššia/nižšia hodnota
- kde nastal zlom
- či ide o stabilný trend alebo náhodný výkyv
- čo experiment ukazuje o hypotéze

### 5. Zlyhania sú použiteľné, ak sú dobre vysvetlené

SAC, starý vertical lidar alebo zlé reward funkcie netreba skrývať. Sú dobrým vedeckým materiálom, ak ukazujú:

- prečo prístup zlyhal
- aké lokálne minimá vznikli
- ako sme chybu diagnostikovali
- čo sme zmenili

## Odporúčaná štruktúra diplomovej práce

### Úvod

Vysvetliť problém:

- autonómny agent v Trackmanii
- cieľ: prejsť trať čo najrýchlejšie, bezpečne a autonómne
- nadviazanie na bakalársku prácu
- hlavný posun: lepšie pozorovanie, lepší tréning, lepšie experimentálne vyhodnotenie

### Teoretické východiská

Odporúčané časti:

- Trackmania ako real-time learning environment
- OpenPlanet a živé telemetry
- neurónové siete ako policy
- lidar/raycasting a geometrická reprezentácia prostredia
- supervised learning
- reinforcement learning a SAC
- genetické algoritmy a neuroevolúcia
- multiobjective optimization a NSGA-II/Pareto front
- reward shaping a problém lokálnych miním

Nedávať priveľa priestoru metódam, ktoré nepoužijeme.

### Analýza pôvodného riešenia

Toto je vhodná kapitola:

- čo robila bakalárska verzia
- čo bolo technicky dobré
- čo sa ukázalo ako slabé
- prečo výsledky neboli presvedčivé
- aké požiadavky z toho vznikli pre novú verziu

### Návrh novej architektúry

Popísať:

- dataflow OpenPlanet -> Car -> Map -> ObservationEncoder -> policy -> Enviroment -> vgamepad
- observation space
- action space
- map extraction and path construction
- surface representation
- height-aware lidar
- reset handshake
- logging and checkpoints

### Tréningové metódy

Samostatne:

- supervised learning ako baseline/seed
- GA ako hlavný live tréning
- SAC ako porovnávaný RL prístup
- lokálna 2D simulácia ako experimentálny sandbox

### Experimenty

Odporúčané skupiny:

1. lokálny 2D reward/ranking experiment
2. lexikografický GA experiment
3. multiobjective GA experiment
4. SAC experiment
5. live Trackmania GA overenie
6. surface/height behavior smoke tests
7. porovnanie s bakalárskou verziou

### Výsledky

Metriky:

- best progress
- dense progress
- finish rate
- best finish time
- mean finish time
- crash rate
- mean crashes
- timeout rate
- virtual driving time
- wall-clock training time
- response FPS/raycast latency

### Diskusia

Otázky:

- prečo lexikografické poradie mení správanie
- kde vznikajú lokálne minimá
- čo vyriešil multiobjective prístup
- prečo SAC nebolo automaticky lepšie
- limity lokálnej simulácie oproti live Trackmanii
- ako sa výsledky líšia od bakalárskej práce

### Záver

Zhrnúť konkrétne príspevky:

- robustnejšia architektúra
- automatizovaný GA tréning
- dense progress a explicitné outcome metriky
- surface-aware observation
- height-aware lidar
- lokálny experimentálny sandbox
- reprodukovateľnejšie vyhodnotenie

## Pracovný plán písania diplomovej práce

Tento plán je živý dokument. Prácu ešte nepíšeme finálnym štýlom; cieľom je
presne určiť, aký príbeh má diplomovka rozprávať, aké dôkazy musíme mať a kde
sa majú jednotlivé experimenty použiť. Štruktúra by mala byť odborná, ale nie
suchý zoznam implementačných detailov. Ideálny štýl je postupné vyšetrovanie
problému: čitateľ vidí rovnaké otázky, na ktoré sme narážali počas vývoja, a
každá ďalšia kapitola prirodzene rieši slabinu tej predchádzajúcej.

### Hlavná dejová línia

Diplomovka by mala stáť na tejto postupnosti:

1. Máme hru Trackmania a chceme autonómne jazdiaceho agenta.
2. Nechceme spracovávať iba obraz z obrazovky, pretože vieme získať lepšie
   štruktúrované dáta cez OpenPlanet a exportovanú geometriu mapy.
3. Vytvoríme virtuálnu reprezentáciu trate a observation space: lidar, budúce
   smerové inštrukcie, rýchlosť, sklz, dense progress, povrchy a výšky.
4. Agent je dopredná neurónová sieť, ktorá mapuje observation na akciu.
5. Najskôr ukážeme, že supervised learning je prirodzený baseline, ale viaže
   sa na kvalitu a množstvo ľudských dát.
6. Hneď potom ukážeme druhý typ imitácie: interaktívne imitation learning /
   DAgger-style zber dát, kde časť akcií vykonáva agent a človek dodáva
   korekčné labely pre stavy, do ktorých by sa pri čistej ľudskej jazde
   bežne nedostal.
7. Potom prejdeme k neuroevolúcii: neimitujeme človeka, ale optimalizujeme
   správanie priamo cez jazdu po trati.
8. Kľúčová otázka znie: čo znamená "dobrá jazda"?
9. Postupne skúmame ranking/fitness funkcie a ukazujeme, že poradie metrík
   zásadne mení správanie agenta.
10. Lexikografický GA funguje, ale medzi rýchlosťou, progresom a bezpečnosťou
   vzniká konflikt.
11. Tento konflikt prirodzene motivuje multi-objective GA / NSGA-II a Pareto
    frontu.
12. Po vyriešení tréningovej metodiky rozširujeme prostredie: výškové rozdiely
    a rôzne povrchy.
13. Finálne hodnotíme agenta na známych aj neznámych tratiach a porovnávame ho
    s bakalárskou verziou aj s ľudským hráčom.

Toto má byť "detektívka": nepredstierať, že sme riešenie poznali od začiatku.
Naopak, ukázať, prečo jednoduché nápady nestačia, aké lokálne minimá vznikajú
a ako nás dáta postupne doviedli k lepšiemu návrhu.

### Navrhovaná kapitola 1: Úvod

Úvod by mal odpovedať na otázku, prečo je Trackmania zaujímavé prostredie:

- je bezpečná, deterministická alebo aspoň kontrolovateľná virtuálna hra
- má reálny čas a fyziku, takže nie je triviálna grid-world úloha
- umožňuje presné telemetry cez OpenPlanet
- trate sú zložené z blokov, takže vieme využiť geometriu prostredia
- problém je blízky autonómnemu riadeniu, ale bez rizík reálneho vozidla

V úvode treba pomenovať aj nedostatok bakalárskej práce: technická myšlienka
bola dobrá, ale výsledky neboli presvedčivé a experimenty neboli dostatočne
reprodukovateľné. Diplomovka má preto dva ciele naraz: zlepšiť agenta a
zlepšiť spôsob, akým jeho kvalitu meriame.

### Navrhovaná kapitola 2: Teoretické východiská

Čitateľ je informatik, ale nemusí poznať strojové učenie ani Trackmaniu. Táto
kapitola má zaviesť pojmy tak, aby sa na ne dalo neskôr odkazovať bez
opakovaného vysvetľovania.

Témy, ktoré treba pokryť:

- prostredie, stav, observation, agent, policy, akcia a epizóda
- synchronné vs. asynchrónne realtime prostredie
- supervised learning, reinforcement learning a neuroevolúcia ako tri odlišné
  prístupy k učeniu policy
- dopredná neurónová sieť ako skladanie lineárnych transformácií a aktivácií
- viacvrstvová neurónová sieť ako univerzálny aproximátor, ale nie záruka
  dobrého správania bez vhodného tréningu
- genetický algoritmus: populácia, jedinec, genóm, fitness/ranking, elitizmus,
  selection, crossover, mutation
- neuroevolúcia: genóm sú váhy neurónovej siete
- reinforcement learning: reward, return, policy gradient, actor-critic,
  PPO/SAC/TD3 iba v rozsahu potrebnom pre experimenty
- reward shaping a problém lokálnych miním
- multi-objective optimalizácia, Pareto dominancia, NSGA-II, crowding distance
- geometrický lidar/raycasting ako alternatíva k image-based perception

Teória nemá byť cieľom práce. Má byť dostatočne presná, aby obhájila použité
metódy, ale nesmie prerásť implementačnú a experimentálnu časť.

### Navrhovaná kapitola 3: Súvisiace práce

Táto kapitola potrebuje deep research. Zdroje treba hľadať cielene v týchto
oblastiach:

- Trackmania AI a TMRL: aké observation/reward/control prístupy sa používajú
- Gran Turismo Sophy: racing RL, reward shaping, multi-objective racing goals
- autonomous racing / TORCS / CarRacing-v0: progress, speed, safety a track
  following rewards
- neuroevolution pre riadenie agentov: GA, NEAT, CMA-ES, evolution strategies
- multi-objective neuroevolution a NSGA-II v herných agentoch
- imitation learning / behavioral cloning pre autonómne riadenie
- sim-to-real alebo sim-to-game gap: rozdiel medzi rýchlym lokálnym sandboxom
  a reálnym realtime prostredím

Pri zdrojoch treba sledovať nie iba "kto dosiahol dobrý výsledok", ale hlavne:

- aké metriky optimalizovali
- ako riešili konflikt medzi rýchlosťou a bezpečnosťou
- či používali dense reward alebo terminálny reward
- ako validovali naučenú policy
- či mali viacero tratí/seeds
- ako reportovali failure modes

### Navrhovaná kapitola 4: Analýza pôvodnej bakalárskej práce

Toto má byť krátky, ale dôležitý most. Netreba bakalársku prácu zhadzovať, ale
treba ju úprimne použiť ako východisko:

- map extractor, OpenPlanet telemetry a lidar boli dobrý základ
- stará policy a PPO tréning neboli dostatočne vysvetlené
- reward fázy a experimenty neboli reprodukovateľné
- výsledný agent jazdil nepresvedčivo, často "opito" zo strany na stranu
- porovnanie s inými riešeniami bolo príliš silné vzhľadom na dôkazy

Z tejto kapitoly má vyplynúť požiadavka na novú prácu: rovnaký problém treba
riešiť systematicky, s lepším observation space, lepším tréningovým frameworkom
a tvrdším experimentálnym vyhodnotením.

### Navrhovaná kapitola 5: Architektúra nového systému

Tu sa má detailne popísať celý runtime dataflow:

```text
OpenPlanet -> TCP packet -> Car -> Map -> ObservationEncoder
            -> NeuralPolicy -> Enviroment -> vgamepad -> Trackmania
```

Podkapitoly:

- OpenPlanet plugin a streamované telemetry
- export blokov mapy a skladanie virtuálnej reprezentácie trate
- 2D/3D lidar a prečo raycastujeme proti geometrii, nie proti obrázku
- path instructions ako signed curvature
- dense progress ako projekcia na stredovú os trate
- action space: gas, brake, steer a target-action režim
- reset handshake v reálnej Trackmanii
- logovanie, checkpointy a reprodukovateľnosť

Túto kapitolu treba robiť veľmi vizuálne. Odporúčané obrázky:

- bloková schéma dataflow
- screenshot/map plot s lidar lúčmi
- ukážka centerline a dense progress projekcie
- ukážka path instructions na zákrute
- ukážka observation vektorových blokov
- ukážka reset/evaluation slučky GA

### Navrhovaná kapitola 6: Supervised learning ako baseline

Supervised learning zatiaľ nemá finálne výsledky, ale stojí za samostatné
zaradenie, ak stihneme nazbierať dáta a otestovať ho.

Hypotéza:

- supervised agent vie napodobniť hráča a môže prejsť trať
- jeho kvalita je však limitovaná dátami a kvalitou ľudského vodiča
- nemusí nájsť lepšiu jazdu než človek, iba priemer jeho ukážok
- môže slúžiť ako seed pre GA alebo RL

Minimálny experiment:

- nazbierať niekoľko jázd na `AI Training #5`
- trénovať `NeuralPolicy` rovnakou architektúrou ako GA
- vyhodnotiť samostatný supervised Driver
- prípadne použiť supervised model ako inicializáciu GA populácie

Ak sa supervised nestihne dôkladne, treba ho uviesť ako baseline/pipeline a
otvorené pokračovanie, nie ako hlavný výsledok.

### Navrhovaná kapitola 7: Interaktívne imitation learning

Táto kapitola by mala nasledovať hneď po supervised learningu, lebo rieši jeho
najprirodzenejší problém. Čistý supervised learning / behavioral cloning trénuje
policy iba na stavoch, ktoré vznikli pri ľudskej jazde. Ak sa agent pri
samostatnej jazde odchýli od tejto distribúcie, môže sa dostať do stavu, kde v
tréningových dátach prakticky nemá odpoveď. Typický príklad je pomalší vjazd do
zákruty: človek v dátach možno zatáčal naplno pri vyššej rýchlosti, ale agent pri
nižšej rýchlosti a odlišnej polohe potrebuje jemnejšiu korekciu.

Tento problém treba pomenovať ako distribučný posun medzi expert datasetom a
stavmi navštívenými agentom. V práci ho môžeme vysvetliť bez prehnanej
formalizácie:

```text
supervised dataset: stavy vytvorené človekom
driver runtime: stavy vytvorené agentom
```

Ak sa tieto dve distribúcie líšia, dobrý offline loss nemusí znamenať dobrú
jazdu. Preto zavádzame interaktívny imitation režim:

1. Na začiatku jazdí človek a model je iba pasívny pozorovateľ.
2. Po úspešnom a potvrdenom attempte sa pokus uloží a sieť sa krátko dotrénuje.
3. V ďalších attemptoch sa lineárne zvyšuje pravdepodobnosť, že malú časť akcií
   vykoná agent.
4. Človek stále drží ovládač a jeho akcia sa zapisuje ako správny korekčný label
   pre aktuálny stav.
5. Do hry sa vykoná buď ľudská akcia, agentova akcia, alebo ich blend podľa
   zvoleného režimu.

Dôležitý detail pre text práce: ak do hry vykonáme agentovu alebo perturbovanú
akciu, nesmieme ju automaticky považovať za správny supervised label. Správny
label je korekcia človeka pre stav, v ktorom sa nachádzame. Preto dataset rozlišuje:

```text
human_actions / actions  = ľudská korekčná akcia, supervised target
executed_actions         = akcia skutočne vykonaná v Trackmanii
```

Týmto režimom získame dáta z menej ideálnych stavov bez toho, aby sme sieť učili
kopírovať vlastné chyby. V texte ho môžeme označiť ako DAgger-style prístup,
pretože ide o podobnú myšlienku agregovania dát z distribúcie navštevovanej
aktuálnou policy, ale implementácia ostáva vlastná a prispôsobená realtime
Trackmanii.

Parametre, ktoré treba v experimentoch reportovať:

- počet akceptovaných attemptov `num_attempts`
- počiatočná a maximálna pravdepodobnosť zásahu agenta
- spôsob miešania akcie: `switch` alebo `blend`
- počet epoch krátkeho dotrénovania po každom attempte
- či sa používa mirror augmentácia
- či Trackmania počúva výhradne virtuálny gamepad

Táto kapitola nemusí byť hlavným výsledkom práce. Je však výborný logický most:
ukáže, že napodobňovanie človeka vieme spraviť lepšie ako čistý offline
supervised learning, no stále zostávame limitovaní tým, že cieľom je imitovať a
nie priamo optimalizovať čas alebo bezpečnosť. To prirodzene otvorí cestu ku GA.

### Navrhovaná kapitola 8: Neuroevolúcia a lexikografický GA

Tu začína hlavný experimentálny príbeh.

Najprv vysvetliť reprezentáciu:

- jedinec = neurónová sieť
- genóm = flatten všetkých váh a biasov
- populácia = množina náhodných alebo seedovaných policy
- rollout = jeden pokus prejsť trať
- ranking = porovnanie jedincov podľa metrík jazdy

Potom ukázať, prečo nestačí cieľ `(finished, -time)`. Náhodný agent prakticky
nikdy nedôjde do cieľa, takže nemáme signál. Preto pridáme progress.

Odporúčaná postupnosť ranking funkcií:

```python
(finished, progress)
(finished, progress, -time)
(finished, progress, -time, -crashes)
(finished, progress, -crashes, -time)
```

Interpretácia:

- `(finished, progress)` naučí dôjsť ďalej, ale nerieši rýchlosť
- `(finished, progress, -time)` je rýchly a jednoduchý baseline
- `(finished, progress, -time, -crashes)` môže preferovať agresívne riešenia
- `(finished, progress, -crashes, -time)` je bezpečnejší, ale môže byť
  pomalší alebo viac konzervatívny

Dôležitá téza: lexikografické poradie je čitateľné a bez váhových "voodoo"
konštánt, ale nie je neutrálne. Poradie metrík samo definuje osobnosť agenta.

### Navrhovaná kapitola 9: Lokálny 2D experimentálny sandbox

Táto kapitola má vysvetliť, prečo vznikli `Experiments/`:

- živá Trackmania je realtime a drahá
- máme iba jednu inštanciu hry
- nejde rozumne robiť veľa reward experimentov priamo v hre
- potrebujeme rýchly spôsob overiť ranking funkcie a algoritmy

Treba zdôrazniť, že sandbox nie je náhrada Trackmanie, ale výskumný nástroj.

Čo má byť v texte:

- mapa sa skladá z rovnakých Trackmania blokov
- steny sú projektované do 2D geometrie
- observation sa snaží kopírovať hlavný projekt bez výšok/povrchovej fyziky
- variabilné FPS simuluje rozdielne frame times v reálnej hre
- `dense_progress` je default pre experimenty
- logujeme per-generation aj per-individual metriky

Grafy:

- learning curve best/mean dense progress
- finish count per generation
- best finish time
- crash/timeout rates
- distribúcia populácie
- virtual driving time vs. wall-clock time

### Navrhovaná kapitola 10: Multi-objective GA / NSGA-II

Táto kapitola má byť prirodzené vyústenie lexikografických konfliktov.

Problém:

- chceme do cieľa
- chceme ďalší progress
- chceme nižší čas
- chceme menej crashov
- chceme rozumnú trajektóriu bez zbytočnej vzdialenosti

Jedna lexikografická funkcia musí zvoliť pevné poradie. Multi-objective GA
namiesto toho porovnáva objektívny vektor:

```python
(finished, progress, -time, -crashes, -distance)
```

alebo normalizovaný monotónne ekvivalentný tvar v implementácii.

Vysvetliť:

- Pareto dominancia
- non-dominated sorting
- Pareto fronta 0
- crowding distance
- priority tiebreak v rámci fronty
- prečo to nie je "magická váha", ale iný spôsob výberu kompromisov

Aktuálny výsledok:

- MOO s `finished,progress,neg_time,neg_crashes,neg_distance` našlo veľmi
  rýchly tréningový finish
- validácia zatiaľ nie je tak robustná ako najlepšie lexikografické varianty
- výsledok je výskumne cenný, lebo ukazuje diverzitu a konflikt cieľov

Túto kapitolu netreba predať ako jednoznačné "MOO vyhralo všetko", ale ako
principiálne čistejšiu metódu na skúmanie kompromisov.

### Navrhovaná kapitola 11: Reinforcement learning ako porovnávacia vetva

RL netreba tlačiť do hlavného príbehu, ak nemá lepší výsledok než GA. Má však
hodnotu ako porovnanie a návrat k bakalárskej práci.

Odporúčané zaradenie:

- po GA/MOO kapitole ako experimentálna porovnávacia vetva
- vysvetliť, že bakalárska práca používala PPO, ale slabý reward a slabé
  vyhodnotenie
- ukázať, že v lokálnom 2D sandboxe PPO s
  `delta_finished_progress_time + gas_steer` vie mapu vyriešiť
- ukázať, že SAC zlyhávalo napriek tomu, že je teoreticky vhodné pre
  kontinuálne akcie
- ukázať negatívny kontrolný experiment s `gas_brake_steer`

Hlavná interpretácia:

- RL je citlivé na reward, action layout, exploration a credit assignment
- PPO sa lokálne ukázalo stabilnejšie než SAC/TD3
- GA ostáva hlavný prístup pre live Trackmaniu, lebo lepšie sedí na drahé
  run-based vyhodnocovanie

Ak nebude dosť času, RL môže byť kratšia diskusná kapitola, nie hlavná vetva.

### Navrhovaná kapitola 12: Tréningové vylepšenia GA

Toto je miesto pre praktické "engineering" príspevky:

- elite caching
- mutation probability a sigma
- mutation decay ako prechod z explorácie do fine-tuningu
- arithmetic mean crossover
- parent pairing bez opakovania
- mirror evaluation alebo mirror probability
- viac crashov / `max_touches`
- seeding zo supervised modelu
- checkpointy vrátane mutation state
- robustný reset handshake

Každú funkciu treba uviesť rovnako:

1. aký problém rieši
2. ako je implementovaná
3. aký očakávame efekt
4. aké riziko alebo trade-off prináša
5. ak máme dáta, ukázať graf

Príklad: elite cache šetrí čas, ale môže zvýhodniť lucky riskantného jedinca,
preto finálne modely treba validovať bez cache.

### Navrhovaná kapitola 13: Výškové rozdiely

Výšky zaradiť až po stabilnom GA príbehu. Najprv treba čitateľovi ukázať
problém:

- 2D lidar ignoruje cestu a rieši len steny
- pri kopci smerom hore by cesta mohla vyzerať ako prekážka
- pri klesaní môže obyčajný lúč "prestreliť" relevantnú geometriu

Historická linka:

- prvý prototyp fixed-step surface marching bol pomalý a nespoľahlivý
- triangle-by-triangle idea bola teoreticky pekná, ale prakticky ťažká
- finálne riešenie využíva štruktúru Trackmania blokov

Aktuálne riešenie:

- každý blok má fitted road plane
- lúče idú po block-grid traversal
- flat úseky môžu fallbacknúť na rýchly 2D raycast
- slope blocks majú side-curtain helper geometriu
- height instructions rozširujú observation

Vizuálne dôkazy:

- obrázok slope bloku
- laser pred kopcom / na kopci / za kopcom
- porovnanie starého a nového prístupu
- FPS/performance porovnanie, ak ho máme

### Navrhovaná kapitola 14: Rôzne povrchy

Povrchy zaradiť podobne ako výšky: ako rozšírenie po tom, čo máme stabilný
agent framework.

Vysvetliť:

- Trackmania bloky môžu mať rovnakú geometriu, ale iný materiál
- agent musí vedieť, že rovnaká zákruta na ľade nie je rovnaký problém ako na
  asfalte
- one-hot surface by nafúkol observation, preto používame kompaktný traction
  coefficient proxy

Aktuálne surface instruction hodnoty:

- asphalt/RoadTech: `1.00`
- grass: `0.70`
- plastic: `0.75`
- dirt: `0.50`
- ice: `0.05`
- snow: `0.15`

Treba jasne napísať, že nejde o presný fyzikálny koeficient z hry, ale o
praktickú reprezentáciu relatívnej trakcie. Ak sa podarí nazbierať telemetry
zo slip/ground-material diagnostiky, môžeme tieto hodnoty podložiť lepšie.

### Navrhovaná kapitola 15: Finálne vyhodnotenie

Finálne hodnotenie musí byť silnejšie než v bakalárskej práci.

Minimálne porovnania:

- nový GA agent vs. bakalársky agent
- nový GA agent vs. supervised baseline, ak bude existovať
- nový GA agent vs. PPO lokálny alebo live RL baseline, ak bude férové
- agent vs. človek na rovnakej trati
- agent na neznámej testovacej trati

Metriky:

- finish rate
- best finish time
- mean/median finish time
- crash rate
- mean crashes
- timeout rate
- dense progress pri neúspešných jazdách
- virtuálny tréningový čas
- wall-clock tréningový čas
- trajektória po mape s farbou podľa rýchlosti

Odporúčané grafy:

- trajectory plot so speed colormap
- finish-time distribution
- progress distribution for failures
- crash/time scatter
- bubble plot: x = čas, y = finish rate, size = crash rate
- Pareto plot bezpečnosť vs. rýchlosť
- learning curve nad virtuálnym herným časom

### Research checklist pred písaním

Pred písaním teoretickej a súvisiacej časti treba urobiť deep research a
uložiť zdroje do bibliografie:

- Trackmania AI / TMRL
- Gran Turismo Sophy
- autonomous racing RL
- TORCS / CarRacing / racing gym environments
- reward shaping v RL
- lexicographic optimization alebo constrained objective design
- NSGA-II a multi-objective evolutionary algorithms
- neuroevolution neural policies
- behavioral cloning / imitation learning for driving
- lidar/raycasting alebo geometry-based perception v simulátoroch
- sim-to-real alebo sim-to-game transfer

Pri každom zdroji si zapísať:

- čo rieši
- aký algoritmus používa
- aké observation/action/reward používa
- aké metriky reportuje
- čo si z toho berieme a čo sa na náš problém nevzťahuje

### Otvorené experimentálne úlohy

Pred finálnym písaním by bolo dobré ešte dokončiť:

- supervised baseline: nazbierať dáta, natrénovať, vyhodnotiť a rozhodnúť, či
  ho použiť ako baseline alebo seed
- interaktívne imitation learning: otestovať `ImitationTrainer.py`, overiť
  curriculum cez `num_attempts`, porovnať čistý supervised model s modelom
  dotrénovaným na stavoch vytvorených agentom
- real Trackmania GA s `(finished, progress, -time)` a dense progress
- crash-aware real GA variant alebo aspoň validácia top jedincov bez cache
- dlhší alebo opakovaný MOO experiment s rovnakými primitive metrics
- real alebo lokálny test výškových máp
- surface test na zmiešaných povrchoch
- finálna unseen-map validácia
- porovnanie s človekom a bakalárskou verziou

### Pravidlá písania

- Nepísať "agent je dobrý", kým nie je graf alebo tabuľka.
- Každý experiment musí mať reprodukovateľnú konfiguráciu.
- Každý graf musí mať interpretáciu.
- Zlyhania písať vecne: čo sa stalo, prečo je to problém, čo sme zmenili.
- Nepreháňať matematiku, ale pri kľúčových veciach použiť formálny zápis:
  ranking tuple, Pareto dominancia, observation vector, policy mapping.
- Písať odborne, ale čitateľne. Cieľ je, aby človek rozumel, prečo bol ďalší
  krok prirodzený.
- Vizuálne vysvetľovať geometriu, observation a metriky vždy, keď sa dá.
- Rozlišovať lokálny 2D dôkaz od dôkazu v reálnej Trackmanii.

## Formátovanie a šablóna

Lokálny priečinok `Diplomová práca/Latex vzor` obsahuje starší FMFI-like `book` template:

- `geometry` s okrajmi top/bottom 2.5 cm, left 3.5 cm, right 2 cm
- `inputenc`, `fontenc`, `babel[slovak]`
- `linespread{1.25}`
- `listings`
- `graphicx`
- `pdfpages`
- `url`
- `hyperref[hidelinks,breaklinks]`

GitHub template `davidmisiak/thesis-template` je čistejšie organizovaný:

- `src/variables.tex`
- `src/chapters`
- `src/definitions.tex`
- `src/references.bib`
- `src/images`
- build cez `make -C src main`

Odporúčanie: ak ostaneme pri lokálnej FMFI šablóne, oplatí sa prevziať aspoň organizačný štýl z GitHub template:

- samostatné kapitoly
- premenné práce v jednom súbore
- definície a makrá mimo hlavného dokumentu
- obrázky a zadanie v `images`
- literatúra v jednom `.bib`

## Čo musí Codex pri písaní diplomovky rešpektovať

- Bakalársku prácu nepovažovať za kvalitatívny vzor textu, ale za historický a technický kontext.
- Nepreberať jej sebavedomé tvrdenia bez dôkazov.
- Pri každom novom tvrdení hľadať experiment, graf alebo log, ktorý ho podporuje.
- V texte rozlišovať:
  - implementačný fakt
  - hypotézu
  - experimentálne potvrdený výsledok
  - subjektívnu interpretáciu
- Všetky reward/ranking funkcie zapisovať explicitne.
- Všetky experimenty robiť reprodukovateľne.
- Zlyhané experimenty použiť ako motiváciu pre lepšie riešenie.
- V práci hovoriť primárne o genetickom algoritme, nie všeobecne o evolučnom algoritme, ale vysvetliť vzťah GA/neuroevolúcia.
- Používať aktuálne názvy súborov: `NeuralPolicy.py`, `GeneticTrainer.py`, `Individual.py`.
- Staré názvy `EvolutionPolicy.py` a `EvolutionTrainer.py` uvádzať iba historicky.

## Najväčšie riziká diplomovky

- Znovu ukázať veľa implementácie, ale málo výsledkov.
- Miešať staré a nové názvoslovie.
- Vysvetliť SAC/GA povrchne.
- Používať grafy bez interpretácie.
- Zameniť lokálnu 2D simuláciu za dôkaz správania v reálnej Trackmanii.
- Mať reward funkcie, ktoré vyzerajú rozumne, ale vedú k lokálnym minimám.
- Nesledovať virtuálny herný čas, čím sa stratí férové porovnanie tréningovej náročnosti.

## Hlavný pracovný záver

Diplomová práca by mala byť napísaná ako príbeh korekcie pôvodného slabého výsledku:

1. Pôvodná bakalárska práca ukázala sľubný smer: OpenPlanet telemetry, geometrický lidar, virtuálna mapa.
2. Slabé miesto bolo experimentálne vyhodnotenie a nepresvedčivý tréning.
3. Nový projekt preto buduje infraštruktúru, ktorá umožňuje skúmať tréning systematicky:
   - lokálny 2D sandbox
   - explicitné ranking/reward funkcie
   - per-individual logging
   - dense progress
   - GA a multiobjective GA
   - surface/height-aware observation
4. Úspech diplomovky nebude len v tom, že agent prejde mapu, ale v tom, že vieme ukázať prečo, za akých podmienok, ako rýchlo sa to naučil a aké kompromisy medzi rýchlosťou a bezpečnosťou vznikli.
