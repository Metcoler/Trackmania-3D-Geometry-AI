# Kostra kapitoly Vyhodnotenie výsledného agenta

Táto kapitola je posledná veľká kapitola pred záverom. Už nemá opisovať, ako sme tréning ladili, ani prečo sme vybrali konkrétny reward, architektúru alebo hyperparametre. To patrí do predchádzajúcich kapitol. Tu má čitateľ vidieť výsledok: ako sa agent správa, čo dokáže, kde zlyháva a ako sa porovnáva s človekom, bakalárskou verziou a jednoduchšími baseline prístupmi.

Tón kapitoly má byť dôkazový, nie tréningový. V predchádzajúcej kapitole sme riešili vyšetrovanie: ako nájsť policy. Tu už ukazujeme, čo sa stane, keď túto policy pustíme na trať.

Hlavný dôraz majú mať:

- prejazdy tratí,
- trajektórie na mape,
- dokončenie trate,
- čas prejazdu,
- stabilita opakovaných jázd,
- správanie na nevidených mapách,
- správanie na výškových rozdieloch,
- správanie na rôznych povrchoch.

Tréningové krivky môžu byť v kapitole spomenuté iba krátko alebo odkazom dozadu. Hlavným dôkazom už nemá byť, že krivka počas tréningu rástla, ale že výsledný agent vie jazdiť.

## 1. Cieľ finálneho vyhodnotenia

Na začiatku treba jasne povedať, čo táto kapitola overuje.

Predchádzajúce kapitoly ukázali:

- ako agent vníma prostredie,
- ako reprezentujeme policy,
- ako sme vyberali architektúru siete,
- ako sme skladali ranking funkciu,
- ako sme nastavovali genetický algoritmus,
- a ktoré tréningové vylepšenia mali zmysel.

Teraz už nejde o to, ktorá mutácia bola lepšia alebo ktorý reward našiel prvý finish. Teraz ide o výsledné správanie agenta.

Otázky kapitoly:

- Dokáže agent dokončiť trať, na ktorej bol trénovaný?
- Dokáže jazdiť stabilne, alebo ide iba o jeden šťastný prejazd?
- Ako vyzerá jeho trajektória?
- Vie prejsť trať, ktorú počas tréningu nevidel?
- Vie jazdiť na tratiach s rôznymi povrchmi?
- Vie jazdiť na tratiach s výškovými rozdielmi?
- Ako sa jeho čas porovná s človekom?
- Ako sa jeho správanie líši od bakalárskej verzie alebo supervised baseline?

Dôležitá veta do finálneho textu:

```text
Cieľom tejto kapitoly nie je hľadať ďalšie nastavenie tréningu, ale overiť, čo výsledná policy reálne dokáže po spustení v prostredí.
```

## 2. Výber agentov na porovnanie

Treba jasne pomenovať, ktoré agenty alebo metódy porovnávame.

Hlavný agent:

- finálny GA/neuroevolučný agent,
- používa aktuálnu geometrickú observation,
- je trénovaný s najlepšie podporenou ranking funkciou,
- predstavuje hlavný výsledok diplomovej práce.

Supervised alebo imitation baseline:

- ukazuje, čo dokáže agent, ktorý sa učil najmä napodobňovať človeka,
- je užitočný na vysvetlenie rozdielu medzi napodobňovaním a optimalizáciou správania,
- nemal by byť prezentovaný ako finálne riešenie, ale ako porovnávací krok.

Bakalárska verzia:

- slúži ako historický a metodický baseline,
- treba ju používať opatrne, pretože staré experimenty nemusia byť úplne porovnateľné,
- je vhodná najmä na kvalitatívne porovnanie: presnejšia reprezentácia, lepšie metriky, lepšie vyhodnotenie.

Ľudský hráč:

- slúži ako praktický orientačný benchmark,
- ukazuje, ako ďaleko je agent od človeka,
- netreba predstierať, že jeden ľudský čas je absolútne vedecký ground truth,
- ale pre prácu je to veľmi zrozumiteľné porovnanie.

Voliteľný RL baseline:

- PPO môže byť spomenuté ako porovnávacia RL vetva,
- SAC a TD3 ako negatívne výsledky nie sú hlavní súperi finálneho agenta,
- RL baseline patrí skôr do diskusie, ak chceme ukázať, že GA bol prakticky silnejší smer.

Odporúčaná tabuľka:

```text
Agent / metóda | Zdroj | Úloha v porovnaní | Poznámka
Finálny GA agent | neuroevolúcia | hlavný výsledok | najlepší podporený tréning
Supervised agent | behavior cloning | baseline napodobňovania | ukazuje limity BC
Bakalársky agent | starý prototyp | historické porovnanie | opatrne, nie úplne rovnaké podmienky
Človek | referenčná jazda | praktický benchmark | orientačný cieľ
PPO | RL screening | doplnkové porovnanie | sľubné, ale nie hlavná metóda
```

## 3. Výber testovacích tratí

Pri každej trati treba povedať, či bola použitá pri tréningu alebo iba pri teste. Toto je dôležité pre interpretáciu generalizácie.

Typy tratí:

- hlavná tréningová trať,
- nevidená 2D/asphalt trať,
- mapa s rôznymi povrchmi,
- mapa s výškovými rozdielmi,
- prípadne kombinovaná mapa s výškou aj povrchmi, ak bude mať finálny dôkaz.

Tréningová trať:

- ukazuje, či agent zvládol úlohu, na ktorej bol optimalizovaný,
- je férové očakávať najlepší výkon práve tu,
- slúži ako hlavný dôkaz, že tréning našiel funkčnú policy.

Nevidená 2D/asphalt trať:

- ukazuje, či agent nezostal úplne naviazaný na jednu mapu,
- nemusí dosahovať rovnaký čas ako na tréningovej trati,
- dôležité je, či dokáže jazdiť a či jeho správanie dáva zmysel.

Surface mapa:

- testuje, či agent vie pracovať s rôznymi povrchmi,
- treba ukázať mapu s farebne odlíšenými povrchmi,
- interpretácia má byť opatrná: prejsť mapu nie je to isté ako dokonale optimalizovať trakciu.

Height mapa:

- testuje, či agent zvládne výškové rozdiely,
- treba ukázať výškový gradient mapy,
- vhodné je sledovať, či trajektória ostáva na ceste aj pri svahoch alebo zmenách výšky.

Odporúčaná tabuľka:

```text
Mapa | Typ | Bola použitá pri tréningu? | Čo overuje
AI Training #5 | hlavná asfaltová trať | áno | hlavný výkon agenta
single_surface_flat | 2D/asphalt | podľa experimentu | základná generalizácia / baseline
multi_surface_flat | surface | podľa experimentu | rôzne povrchy
single_surface_height | height | podľa experimentu | výškové rozdiely
```

Ak finálne mapy budú iné, tabuľku neskôr upraviť. Dôležité je zachovať logiku: tréningová mapa, nevidená mapa, surface mapa, height mapa.

## 4. Metriky finálneho hodnotenia

V tejto kapitole už nie sú hlavné metriky typu tréningový loss alebo progress počas generácií. Tie vysvetľujú tréning, ale nie finálne správanie.

Hlavné finálne metriky:

- či agent trať dokončil,
- finish rate pri opakovaných jazdách,
- najlepší čas,
- priemerný alebo mediánový čas úspešných jázd,
- stabilita času,
- počet crashov alebo dotykov,
- progress pri nedokončených jazdách,
- či agent opustil cestu,
- či trajektória dáva zmysel vizuálne.

Čas:

- najlepší čas je dobrý na porovnanie potenciálu,
- priemer alebo medián je lepší na porovnanie stability,
- ak máme iba jeden prejazd, treba to explicitne povedať.

Crash/dotyk:

- crash rate ukazuje bezpečnosť,
- počet dotykov ukazuje kvalitu prejazdu,
- pri viacdotykovej diagnostike treba rozlíšiť tvrdý crash a wall-hug správanie.

Progress:

- pri nedokončených jazdách ukazuje, kam sa agent dostal,
- v tejto práci znamená plynulý geometrický progress,
- netreba ho zamieňať s discrete/block progressom.

Odporúčaná tabuľka:

```text
Agent | Mapa | Pokusy | Finish rate | Best time | Median time | Crashes/touches | Mean progress | Poznámka
```

Pri porovnaní s človekom:

- uviesť čas človeka,
- uviesť, či ide o autorovu jazdu, priemerný pokus alebo najlepší známy pokus,
- nepísať prehnane, ak ľudský benchmark nie je rozsiahly.

## 5. Trajektórie a správanie na mape

Hlavný vizuálny dôkaz tejto kapitoly majú byť path grafy. Tréningové grafy ukazujú, že sa agent učil, ale path graf ukazuje, čo vlastne robí.

Pri každej dôležitej mape je vhodné mať:

- mapový podklad,
- štart a cieľ,
- povrchy farebne odlíšené,
- výšku vyjadrenú jasom alebo gradientom,
- trajektóriu agenta,
- prípadne rýchlostný gradient na trajektórii,
- crash alebo touch body,
- teacher alebo human trajectory, ak je relevantná.

Pri interpretácii trajektórie sa treba pýtať:

- drží sa agent stredu trate alebo reže zákruty?
- brzdí pred zákrutou alebo až v nej?
- je jeho trajektória plynulá?
- opakuje podobnú líniu pri viacerých pokusoch?
- kde sa líši od ľudského vodiča?
- kde vznikajú nárazy alebo spomalenia?

Toto je dôležité: obrázok nemá byť iba pekná ilustrácia. Každý path graf musí byť v texte vysvetlený.

Príklad interpretácie:

```text
Na prvej rovinke agent zrýchľuje podobne ako učiteľské jazdy, ale pred zákrutou volí skorší nájazd k vnútornej hrane. To skracuje dráhu, ale zároveň zvyšuje riziko dotyku pri výjazde zo zákruty.
```

## 6. Vyhodnotenie generalizácie

Generalizácia znamená, že agent nepôsobí rozumne iba na trati, na ktorej bol optimalizovaný, ale vie použiť naučené správanie aj inde.

Tu treba byť veľmi presný. Sú rôzne úrovne generalizácie:

- agent dokáže prejsť inú mapu,
- agent dokáže prejsť inú mapu stabilne,
- agent dokáže prejsť inú mapu rýchlo,
- agent dokáže prejsť výrazne odlišný typ mapy.

Tieto tvrdenia nie sú rovnaké. Ak agent dokončí nevidenú mapu pomaly alebo s dotykmi, stále je to zaujímavý výsledok, ale netreba písať, že dokonale generalizuje.

Do tejto časti patria:

- path graf na nevidenej mape,
- finish/progress/time/crash tabuľka,
- porovnanie s tréningovou mapou,
- stručná interpretácia, čo sa prenieslo a čo nie.

Ak holdout/generalization experiment nevyšiel ideálne, formulovať to opatrne:

- “generalizácia je obmedzená”,
- “agent preniesol základné správanie, ale nie optimálnu rýchlu líniu”,
- “výsledok ukazuje potrebu širšieho tréningového rozdelenia tratí”.

Netreba z toho robiť katastrofu. Aj obmedzená generalizácia je dôležitý výsledok.

## 7. Vyhodnotenie výškových rozdielov

Táto časť má ukázať, či agent zvláda svet, ktorý nie je iba rovinný.

V texte treba najprv pripomenúť:

- výškový modul rozširuje observation,
- lidar/raycasting musí zohľadňovať 3D geometriu,
- svahy a kopce menia interpretáciu priestoru pred autom.

Potom ukázať:

- mapu s výškovým gradientom,
- trajektóriu agenta na height mape,
- miesta, kde agent ide do kopca alebo z kopca,
- či ostáva na ceste,
- či sa správanie výrazne líši od 2D asphalt prípadu.

Metriky:

- finish,
- čas,
- crash/touch count,
- progress,
- prípadne stabilita cez opakované prejazdy.

Interpretácia:

- ak agent mapu dokončí, je to dôkaz, že rozšírená reprezentácia je použiteľná,
- ak jazdí pomalšie alebo s dotykmi, treba povedať, že výška úlohu sťažuje,
- ak zlyhá na určitých miestach, práve tie miesta treba vizuálne ukázať.

## 8. Vyhodnotenie rôznych povrchov

Táto časť má ukázať, či agent zvláda mapy, kde sa nemení iba geometria, ale aj povrch.

V texte treba pripomenúť:

- rôzne povrchy môžu meniť trakciu,
- rovnaká akcia môže mať iný účinok,
- preto agent potrebuje informáciu o povrchu v observation.

Do kapitoly patria:

- surface mapa s farebnou legendou,
- trajektória agenta,
- interpretácia správania na jednotlivých povrchoch,
- crash/touch body, ak vznikli.

Treba nepísať viac, než vieme:

- ak agent trať dokončí, môžeme povedať, že surface observation umožnila použiteľné správanie na tejto mape,
- ak nemáme detailnú analýzu trakcie, netreba tvrdiť, že agent optimálne využíva každý povrch,
- ak sa správa opatrnejšie, môže to byť dobrý výsledok aj bez najlepšieho času.

## 9. Časové porovnanie

Táto časť má byť pre čitateľa veľmi zrozumiteľná. Po všetkých metrikách, grafoch a trajektóriách sa každý prirodzene opýta: ako rýchlo jazdí?

Porovnať:

- finálny agent,
- človek,
- supervised/BC baseline,
- bakalárska verzia, ak máme porovnateľný údaj,
- voliteľne PPO alebo iný RL baseline.

Odporúčaná tabuľka:

```text
Metóda | Mapa | Best time | Finish rate | Crash/touch rate | Poznámka
Človek | ... | ... | ... | ... | referenčný čas
Finálny GA agent | ... | ... | ... | ... | hlavný výsledok
BC/imitation agent | ... | ... | ... | ... | napodobňovací baseline
Bakalársky agent | ... | ... | ... | ... | historické porovnanie
PPO baseline | ... | ... | ... | ... | doplnkový RL výsledok
```

Pri interpretácii:

- ak agent nie je rýchlejší než človek, stále môže ísť o úspešný výsledok, ak je výrazne lepší než predchádzajúca verzia,
- ak je agent rýchly, ale nestabilný, treba to priznať,
- ak človek jazdil iba niekoľko pokusov, treba uviesť, že ide o orientačný benchmark.

Porovnanie s bakalárskou prácou:

- najlepšie formulovať cez zlepšenie metodiky a kvality vyhodnotenia,
- ak máme časové porovnanie, uviesť ho,
- ak nemáme priamo porovnateľné metriky, nepísať silné kvantitatívne tvrdenie.

## 10. Diskusia správania agenta

Táto časť má spojiť čísla a obrázky do celkového hodnotenia.

Silné stránky:

- agent dokáže jazdiť bez obrazového vstupu,
- využíva geometrickú reprezentáciu prostredia,
- vie dokončiť hlavné trate,
- path grafy ukazujú zmysluplnú trajektóriu,
- GA tréning našiel policy, ktorá nie je iba napodobnením človeka,
- systém je analyzovateľný cez metriky aj mapové trajektórie.

Slabiny:

- generalizácia môže byť obmedzená,
- agent môže byť citlivý na konkrétnu mapu,
- výška a povrchy zvyšujú náročnosť,
- TM2D a live Trackmania nie sú totožné,
- jazda nemusí byť tak plynulá ako ľudská,
- agent môže mať wall-touch alebo riskantné miesta.

Dôležité rozlíšenie:

- čo výsledky dokazujú,
- čo iba naznačujú,
- čo zatiaľ nedokazujú.

Príklady formulácií:

- “Výsledok ukazuje, že zvolená reprezentácia je dostatočná na naučenie prejazdu trate.”
- “Výsledok ešte nedokazuje všeobecnú schopnosť jazdiť ľubovoľnú Trackmania mapu.”
- “Agent sa na nevidenej mape nespráva optimálne, ale zachováva základnú schopnosť sledovať trať.”
- “Rozšírenia o výšku a povrchy sú funkčné na testovaných mapách, ale ich robustnosť by vyžadovala širší testovací súbor.”

## 11. Prechod do záveru

Záver tejto kapitoly má pripraviť záverečnú kapitolu práce.

Treba zhrnúť:

- vytvorili sme reprezentáciu sveta bez obrazu,
- navrhli sme policy reprezentovanú neurónovou sieťou,
- našli sme spôsob, ako ju trénovať,
- výsledný agent dokáže jazdiť v testovaných podmienkach,
- vyhodnotili sme ho nielen tréningovými metrikami, ale aj trajektóriami a porovnaním.

Prechod do záveru:

```text
Tým sa uzatvára praktická časť práce. V závere môžeme zhrnúť, ktoré ciele zadania sa podarilo naplniť, kde ostávajú limity a aké smerovanie by malo zmysel pri ďalšom výskume.
```

## Obrázky a tabuľky pre kapitolu

Povinné alebo veľmi vhodné:

- path graf finálneho agenta na hlavnej mape,
- path graf na nevidenej 2D/asphalt mape,
- path graf na surface mape,
- path graf na height mape,
- tabuľka časového porovnania človek vs agent vs BC baseline,
- tabuľka finish/time/crash/progress pre testovacie mapy.

Voliteľné:

- overlay viacerých prejazdov jedného agenta,
- speed-colored trajectory,
- crash markers,
- porovnanie trajektórie človeka a agenta,
- krátky graf stability času pri opakovaných prejazdoch.

Pri mapách používať aktuálny map renderer:

- start zelený,
- finish červený,
- povrchy farebne odlíšené,
- výška cez jas alebo gradient,
- okraje cesty zvýraznené,
- pri flat mapách bez zbytočnej height legendy.

## Čo do tejto kapitoly nepatrí

Do tejto kapitoly nepatrí:

- podrobný opis reward sweepu,
- podrobný opis hyperparameter gridu,
- dlhý výklad BC/imitation tréningu,
- debug grafy,
- smoke testy,
- neúspešné experimenty bez priameho vzťahu k finálnemu agentovi,
- implementačné detaily súborov.

Ak sa niektorý tréningový výsledok spomína, má slúžiť iba na vysvetlenie, odkiaľ finálny agent pochádza. Hlavná otázka kapitoly je: ako výsledný agent jazdí?

## Miesta na doplnenie po finálnych jazdách

Tieto údaje bude treba doplniť, keď budú finálne runy pripravené:

- finálny best time hlavného agenta,
- počet opakovaných prejazdov,
- finish rate na hlavnej mape,
- výsledok na nevidenej mape,
- výsledok na height mape,
- výsledok na surface mape,
- ľudský referenčný čas,
- porovnanie s BC/bakalárskou verziou,
- finálne path grafy.

Je úplne v poriadku, ak pracovná verzia kapitoly obsahuje placeholdery. Lepšie je mať miesto jasne označené, než do textu vymyslieť čísla pred finálnym vyhodnotením.

## Pracovný verdikt

Táto kapitola má byť pokojnejšia než kapitola o tréningu. Už to nie je vyšetrovanie, kde skúšame, ktorá stopa vedie ďalej. Je to prezentácia dôkazov. Čitateľ má po tejto kapitole vidieť, že agent nie je len záznam v tabuľke alebo pekná tréningová krivka. Je to policy, ktorá sa dá spustiť, prejde trať, zanechá trajektóriu na mape a dá sa porovnať s človekom aj so staršími verziami systému.

Ak predchádzajúce kapitoly odpovedali na otázku “ako sme agenta vytvorili a natrénovali”, táto kapitola odpovedá na otázku “čo ten agent v skutočnosti dokáže”.
