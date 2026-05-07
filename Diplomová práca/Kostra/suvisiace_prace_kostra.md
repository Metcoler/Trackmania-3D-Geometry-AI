# Kostra kapitoly Súvisiace práce

Tento dokument je pracovná osnova kapitoly o súvisiacich prácach. Cieľom nie je ešte písať finálny text, ale určiť, aké oblasti treba pokryť, aké zdroje vyzerajú použiteľne a akú argumentačnú niť má kapitola sledovať.

Kapitola má mať dve úlohy:

- ukázať, z čoho sa dá pri návrhu autonómneho agenta inšpirovať,
- vysvetliť, v čom majú existujúce prístupy obmedzenia vzhľadom na náš problém.

Nemá to byť zoznam náhodných článkov. Súvisiace práce by mali postupne priviesť čitateľa k otázke: ako reprezentovať svet pre vodiaceho agenta v hre, ako ho trénovať a prečo nestačí iba pravidlová AI, iba obrazovka alebo iba jeden univerzálny RL prístup.

## 0. Krátky most z teórie

Pred súvisiacimi prácami má teoretická časť obsahovať iba krátku orientáciu „Trackmania ako prostredie pre agenta“. Nemá tam byť detailný opis hry, editoru tratí ani nášho spôsobu vnímania. Čitateľ má z teórie vedieť len to, že Trackmania je realtime pretekárske virtuálne prostredie s time-attack cieľom, kde o kvalite jazdy rozhoduje celá trajektória.

V kapitole súvisiace práce potom môžeme nadviazať takto:

- hry sú prirodzené testovacie prostredie pre AI,
- pretekárske hry sú špecifické tým, že rozhodnutia majú plynulú dynamiku a oneskorené následky,
- Trackmania je zaujímavá najmä kvôli realtime riadeniu, time-attack cieľu a dôrazu na presnú trajektóriu.

V tejto kapitole už netreba znovu vysvetľovať základ hry. Treba ju zasadiť medzi existujúce prístupy: tradičnú game AI, autonómne riadenie z obrazu alebo senzorov, racing simulátory, moderné deep RL a praktické Trackmania projekty.

## 1. Umelá inteligencia v hrách: od pravidiel k učeniu

Táto časť má vysvetliť historický kontext. V hrách sa pojem „AI“ dlho nepoužíval iba pre učenie, ale aj pre ručne navrhnuté správanie postáv.

Treba spomenúť:

- pravidlové systémy,
- konečné automaty,
- behavior trees,
- pathfinding,
- skriptované správanie,
- rozdiel medzi AI pre zábavné správanie v hre a AI ako vedeckým agentom optimalizujúcim cieľ.

Hlavná interpretácia:

- tradičná game AI je často navrhnutá vývojárom tak, aby pôsobila dobre na hráča,
- nemusí sa učiť zo skúsenosti,
- v pretekárskej hre môže byť scripted vodič použiteľný ako herný protivník, ale nie je to automaticky riešenie problému učenia autonómnej policy.

Použiteľné zdroje:

- Yannakakis a Togelius: *Artificial Intelligence and Games*, pre širší prehľad AI v hrách a rozdelenie použití AI na hranie hier, generovanie obsahu a modelovanie hráčov: https://link.springer.com/book/10.1007/978-3-031-83347-2
- Oficiálna stránka knihy *Artificial Intelligence and Games*: https://gameaibook.org/
- Behavior trees survey: upozorňuje, že behavior trees vznikli v hrách a reagovali aj na škálovateľnostné problémy konečných automatov: https://www.sciencedirect.com/science/article/pii/S0921889022000513

Čo z toho chceme použiť:

- Hry nie sú iba zábava, ale aj dlhodobý testbed pre AI.
- Staršia game AI často používala ručne navrhnuté rozhodovacie štruktúry.
- Naša práca sa posúva od ručne navrhnutého správania k učeniu policy.

## 2. Reprezentácia vodiča a sveta v pretekárskych úlohách

Táto časť má premostiť od všeobecnej game AI k autonómnemu vodičovi.

Otázky:

- Čo agent vidí?
- Je vstupom obraz, telemetry, lidar, mapa alebo ich kombinácia?
- Je výstupom priama akcia, cieľová trajektória alebo plán?
- Je správanie ručne skriptované, učené zo vzorov alebo optimalizované cez reward?

Typické reprezentácie:

- obraz z kamery,
- nízkorozmerné telemetry,
- lidar/rangefinder,
- lokálna mapa alebo centerline,
- predikovaná trajektória,
- kombinácia naučenej siete a ručne navrhnutého kontroléra.

Použiteľné zdroje:

- NVIDIA PilotNet / end-to-end learning: CNN mapuje obraz z prednej kamery na steering, čo je dobrý príklad image-based behavior cloning: https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
- Pan et al., imitation learning for agile autonomous driving: ukazuje, že online imitation learning rieši covariate shift lepšie než čisté batch behavioral cloning: https://journals.sagepub.com/doi/10.1177/0278364919880273
- Ross, Gordon, Bagnell, DAgger: teoretický základ problému, že v sekvenčnom rozhodovaní budú budúce pozorovania závisieť od predchádzajúcich akcií agenta: https://proceedings.mlr.press/v15/ross11a.html

Čo z toho chceme použiť:

- Obraz je prirodzený vstup, ale môže byť výpočtovo náročný a vyžaduje naučiť aj percepciu.
- Telemetry alebo geometrické vstupy môžu byť kompaktnejšie a stabilnejšie.
- Behavioral cloning je užitočný, ale samostatne trpí distribučným posunom.

## 3. Autonómne pretekanie v simulátoroch

Táto časť má ukázať, že autonómne pretekanie je existujúca výskumná téma, nie izolovaný nápad.

Treba spomenúť:

- TORCS a Simulated Car Racing Championship,
- súťaže v simulovanom pretekaní,
- neural-network a evolutionary prístupy,
- reinforcement learning v pretekárskych simulátoroch,
- rozdiel medzi vysoko kontrolovaným simulátorom a komerčnou hrou.

Použiteľné zdroje:

- Loiacono et al., *The 2009 Simulated Car Racing Championship*: súťaž v TORCS, popis regulácií, frameworku a najlepších prístupov tímov: https://www.researchgate.net/publication/224138588_The_2009_Simulated_Car_Racing_Championship
- Simulated Car Racing Championship software manual: architektúra súťažného softvéru, senzory a akčné rozhranie: https://www.researchgate.net/publication/236118882_Simulated_Car_Racing_Championship_Competition_Software_Manual
- Human-like TORCS controller: neurónové siete trénované z dát ľudského hráča, plus scripted policy na sledovanie trajektórie a cieľovej rýchlosti: https://e-archivo.uc3m.es/entities/publication/9e786051-8f85-4610-b7e2-d1cd1ff6daf2
- Learn-to-Race: novší príklad prostredia pre autonómne pretekanie s dôrazom na simuláciu a RL: https://learn-to-race.org/

Čo z toho chceme použiť:

- Vodiaci agent môže používať rôzne vstupy: senzory, mapu, obraz, trajectory prediction.
- Mnohé riešenia kombinujú učenie a ručne navrhnuté komponenty.
- Simulované pretekanie je užitočné, ale prenos do konkrétnej hry alebo realtime prostredia prináša nové obmedzenia.

## 4. Moderné úspechy: Gran Turismo Sophy a hranice deep RL

Táto časť má ukázať najbližší „veľký“ úspech AI v pretekárskej hre.

Gran Turismo Sophy je dôležitý príklad, pretože:

- ide o pretekársku hru,
- používa hlboké reinforcement learning,
- rieši vysokorýchlostnú jazdu,
- rieši aj taktické interakcie a pravidlá fair play,
- ukazuje, že reward dizajn je kritický aj pri veľkých systémoch.

Použiteľný zdroj:

- Wurman et al., *Outracing champion Gran Turismo drivers with deep reinforcement learning*, Nature 2022: https://www.nature.com/articles/s41586-021-04357-7

Čo z toho chceme použiť:

- Pretekárske hry sú relevantné pre výskum komplexného riadenia.
- Úspešný deep RL systém potrebuje veľkú infraštruktúru, silný reward dizajn a veľa tréningu.
- To je dobré kontrastovať s naším cieľom: menší projekt, dostupné telemetry/geometria, jedna hra a praktická diplomová infraštruktúra.

## 5. Trackmania a existujúce AI prístupy

Táto časť má byť najbližšia nášmu problému. Základ hry už bude predstavený v teórii, preto tu netreba opakovať, čo je Trackmania. Cieľom tejto časti je ukázať, aké existujú Trackmania AI prístupy, aké vstupy používajú, ako trénujú agenta a v čom sú pre našu prácu inšpiratívne alebo obmedzujúce.

Trackmania ako kontext:

- oficiálny Ubisoft zdroj použiť iba na krátke pripomenutie hry a jej tvorivého/kompetitívneho charakteru: https://www.ubisoft.com/en-us/company/about-us/our-brands/trackmania
- Trackmania Wiki použiť opatrne iba pri komunitne dokumentovaných prvkoch, napríklad editor alebo time-attack režim: https://www.trackmania.wiki/wiki/Trackmania_Wiki
- tieto zdroje nemajú niesť hlavnú vedeckú argumentáciu; tú majú niesť články, benchmarky, TMRL dokumentácia a porovnanie prístupov.

TMRL:

- TMRL je framework pre deep RL v real-time aplikáciách a obsahuje pipeline pre Trackmania 2020: https://github.com/trackmania-rl/tmrl
- používa real-time Gym prístup,
- trénuje policy v Trackmanii,
- podporuje raw screenshots aj jednoduchšie LIDAR/rangefinder pozorovania,
- používa algoritmy ako SAC alebo REDQ,
- ovláda hru cez virtuálny gamepad.

OpenPlanet:

- OpenPlanet je dôležitý ako nástroj prístupu k runtime dátam a skriptovaniu v Trackmanii: https://www.trackmania.wiki/wiki/Openplanet
- dokumentácia ukazuje, že existuje prístup k herným triedam a stavom cez Trackmania API: https://mp4.openplanet.dev/TrackMania

Komunitné a neformálne zdroje:

- existujú videá a komunitné pokusy s RL v Trackmanii,
- tieto zdroje môžu byť použité iba opatrne ako ilustrácia záujmu komunity,
- nemali by byť hlavným vedeckým zdrojom, ak nemajú článok, kód alebo presný popis experimentu.

Čo z toho chceme použiť:

- TMRL je najbližší príbuzný projekt.
- Veľa Trackmania AI prístupov používa obraz alebo screenshot-based pipeline.
- Niektoré prístupy používajú lidar/rangefinder, ale často odvodený z obrazu alebo runtime prostredia.
- Náš neskorší rozdiel treba formulovať opatrne: nechceme tvrdiť, že všetko ostatné je zlé, ale že náš dôraz je na vlastnú geometrickú reprezentáciu, explicitné metriky a systematické GA experimenty.

## 6. Problémy existujúcich prístupov, ktoré sú relevantné pre našu prácu

Táto časť má vytvoriť zápletku kapitoly. Súvisiace práce nie sú len inšpirácia, ale aj dôvod, prečo robíme vlastný návrh.

Problémy image-based prístupov:

- obraz je bohatý, ale výpočtovo drahší,
- model sa musí naučiť percepciu aj riadenie naraz,
- odozva a stabilita môžu byť problém v realtime hre,
- interpretácia naučeného správania je ťažšia.

Problémy čistého behavioral cloning:

- model kopíruje experta,
- netrénuje sa priamo na cieľovú metriku,
- distribučný posun spôsobuje akumuláciu chýb.

Problémy čistého RL:

- reward dizajn je náročný,
- sparse reward môže spôsobiť slabý signál,
- tréning v realtime prostredí je drahý,
- exploration môže produkovať veľa nepoužiteľných alebo nebezpečných pokusov.

Problémy ručne navrhnutej game AI:

- pravidlá a automaty môžu byť čitateľné,
- ale pri plynulom riadení a veľkej variabilite tratí môžu byť krehké,
- ručne navrhnuté správanie nemusí nájsť lepšiu jazdu než pôvodný dizajn.

Problémy simulátorov:

- simulácia nemusí presne zachytiť cieľové prostredie,
- agent sa môže naučiť využívať chyby simulácie,
- pri prenose do konkrétnej hry treba stále validovať v cieľovom prostredí.

## 7. Ako kapitola pripraví miesto pre náš prístup

Na konci súvisiacich prác treba jasne, ale nie reklamne, povedať:

- existujú tradičné herné AI prístupy, ale tie sú skôr ručne navrhnuté,
- existujú image-based end-to-end prístupy, ale sú ťažšie a výpočtovo drahšie,
- existujú Trackmania RL projekty, najmä TMRL, ale sú primárne RL frameworky a často stavajú na screenshot/LIDAR pipeline,
- existujú simulátorové racing benchmarky, ale nie sú totožné s Trackmaniou,
- preto má zmysel skúmať vlastný smer: kompaktná reprezentácia sveta, explicitné hodnotenie jazdy, neuroevolúcia/GA a systematické experimenty.

Formulovať opatrne:

- nie „náš prístup je lepší“,
- ale „náš prístup rieši inú kombináciu obmedzení“,
- hlavne realtime hru, dostupné telemetry, geometrickú reprezentáciu, interpretovateľné metriky a drahé vyhodnocovanie policy.

## Konkrétne zdrojové kotvy

Prvé zdrojové kotvy pre kapitolu súvisiacich prác sú pripravené v `Diplomová práca/Latex/references.bib` a pracovné PDF sú v `Diplomová práca/Sources/`.

- AI v hrách a tradičné správanie agentov: `yannakakis_togelius2018aigames`, `millington_funge2019ai_for_games`, `iovino2022behavior_trees`,
- racing simulátory a benchmarky: `wymann2000torcs`, `loiacono2010scrc`, `game_ai_pro_racetrack2013`,
- moderné racing RL: `gt_sport_drl_fuchs2021`, `wurman2022gt_sophy`, `offline_mania2024`,
- Trackmania a najbližší praktický príbuzný projekt: `tmrl_github2024`, `ubisoft_trackmania2026`, `openplanet_wiki2026`, `trackmania_wiki2026`,
- opatrný Trackmania kontext, nie hlavný vedecký dôkaz: `trackmania_drl_preprint2024`.

Pri finálnom texte treba oddeliť recenzované vedecké zdroje od softvéru, dokumentácie a komunitných zdrojov. TMRL je veľmi relevantný prakticky, ale má byť citovaný ako softvér/dokumentácia. Trackmania preprint má zostať iba ako kontext, lebo nejde o stabilný recenzovaný článok.

## Navrhovaná štruktúra kapitoly

Finálna kapitola môže mať približne toto poradie:

1. AI v hrách a tradičné správanie agentov.
2. Autonómne riadenie a reprezentácia vodiča.
3. Pretekárske simulátory a súťaže.
4. Moderné deep RL v pretekárskych hrách.
5. Trackmania AI prístupy a TMRL.
6. Identifikované obmedzenia existujúcich prístupov.
7. Motivácia vlastného riešenia.

## Zdroje na spracovanie do bibliografie

Pri písaní finálneho textu bude vhodné pripraviť BibTeX záznamy minimálne pre:

- Georgios N. Yannakakis, Julian Togelius: *Artificial Intelligence and Games*. Springer, 2018 alebo 2025.
- Michele Colledanchise, Petter Ögren a ďalší survey o behavior trees v robotike a AI: https://www.sciencedirect.com/science/article/pii/S0921889022000513
- Daniele Loiacono et al.: *The 2009 Simulated Car Racing Championship*.
- Jorge Muñoz et al.: *A human-like TORCS controller for the Simulated Car Racing Championship*.
- Peter R. Wurman et al.: *Outracing champion Gran Turismo drivers with deep reinforcement learning*.
- Stéphane Ross, Geoffrey Gordon, Drew Bagnell: *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*.
- Yunpeng Pan et al.: *Imitation learning for agile autonomous driving*.
- Mariusz Bojarski et al.: *End to End Learning for Self-Driving Cars*.
- TMRL GitHub alebo dokumentácia: https://github.com/trackmania-rl/tmrl
- Ubisoft/Trackmania oficiálny opis hry: https://www.ubisoft.com/en-us/company/about-us/our-brands/trackmania
- Trackmania Wiki pre editor/time-attack kontext: https://www.trackmania.wiki/wiki/Track_Editor
- OpenPlanet wiki alebo dokumentácia pre všeobecný kontext dostupnosti runtime skriptovania: https://www.trackmania.wiki/wiki/Openplanet

## Pracovný verdikt

Súvisiace práce by nemali pôsobiť ako „pozrite, existuje RL a hry“. Majú vytvoriť logický tlak: tradičná game AI je príliš ručná, image-based RL je silné, ale drahé a menej interpretovateľné, čisté behavioral cloning má problém distribučného posunu a racing simulátory nie sú totožné s realtime Trackmaniou. Z toho prirodzene vznikne miesto pre náš projekt: autonómny Trackmania agent, ktorý používa kompaktnú geometrickú reprezentáciu sveta, explicitné metriky jazdy a systematicky vyhodnocované učenie policy.
