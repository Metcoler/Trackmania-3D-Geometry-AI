# Kostra kapitoly Hľadanie policy a tréning agenta

Tento dokument je pracovná osnova kapitoly, ktorá tvorí jadro diplomovej práce. Predchádzajúca kapitola vysvetlila, ako sme z Trackmanie vytvorili pozorovateľné prostredie: máme stav auta, virtuálnu mapu, observation vektor, akčný priestor a uzavretý cyklus `observation -> policy -> action -> new observation`.

Táto kapitola začína presne v bode, kde už agent vie vnímať a konať, ale ešte nevie jazdiť. Máme teda pripravený svet, ale nepoznáme pravidlo, podľa ktorého sa v ňom má agent rozhodovať. Hľadáme policy.

Naratív kapitoly môže byť pokojne vedený ako detektívne vyšetrovanie. Nechceme hneď tvrdiť, že poznáme riešenie. Na začiatku vieme iba to, že potrebujeme funkciu, ktorá z pozorovania vyberie akciu. Postupne zisťujeme:

- či neurónová sieť vôbec dokáže mapovať observation na ľudskú akciu,
- aká veľká má byť,
- či stačí napodobňovať človeka,
- ako vyhodnotiť správanie agenta v prostredí,
- ako nastaviť genetický algoritmus,
- ktoré tréningové vylepšenia reálne pomohli,
- a čo sa z toho dá preniesť do výškových rozdielov a rôznych povrchov.

Kapitola by mala byť písaná ako logický reťazec rozhodnutí, nie ako náhodný zoznam experimentov. Každý experiment má odpovedať na otázku, ktorá vznikla v predchádzajúcej časti. Nemáme však vysvetlovať princíp ako niečo funguje a opakovať sa znovu v práci, potrebujeme sa na to odkázať do teoretickej časti.

Treba uviesť, že aby sme zjednodušili problém, budeme najskôr experimentovať v zjednodušenej verzií sveta trackmanie a to iba v jednej rovine, bez použitia rôznych povrchov. Neuvádzame že sa jendá o nejaký senbox, nerozoberáme,, vždy to uvedieme iba tymto smerom. V zjednodušenej verzii trackmanie alebo tak

## 1. Problém: hľadáme optimálnu policy

Na konci predchádzajúcej kapitoly máme všetky technické časti potrebné na to, aby agent mohol jazdiť:

- z hry získavame stav auta,
- vo virtuálnej mape vieme dopočítať observation,
- neurónovej sieti vieme observation odovzdať,
- výstup siete vieme preložiť na akciu,
- akciu vieme aplikovať späť do Trackmanie alebo do TM2D.

Tým však vzniká hlavná otázka celej práce: akú funkciu má policy vlastne reprezentovať?

V našom prípade policy chápeme ako funkciu:

```text
policy(observation) -> action
```

alebo formálnejšie:

```text
πθ(o) = a
```

kde `o` je observation, `a` je akcia a `θ` sú parametre policy. Ako reprezentáciu tejto policy volíme neurónovú sieť. Dôvod je prirodzený: neurónová sieť je parametrická funkcia, ktorá dokáže aproximovať zložité nelineárne vzťahy medzi vstupom a výstupom.

Tým sa však problém nevyrieši, iba presunie. Namiesto ručného písania pravidiel teraz hľadáme dobré parametre neurónovej siete. Vznikajú otázky:

- koľko vrstiev má sieť mať,
- koľko neurónov má byť v jednotlivých vrstvách,
- aké aktivačné funkcie použiť,
- aké aktivačné funkcie použiť na výstupe,
- a hlavne, ako nájsť parametre, ktoré budú v prostredí fungovať.

Tu sa dá pekne použiť detektívny tón: máme podozrivého, teda neurónovú sieť ako univerzálny aproximátor, ale ešte nevieme, akú podobu má mať a ako ju prinútiť jazdiť dobre.

## 2. Výber architektúry neurónovej siete

Prvý problém je veľkosť siete. V teórii neurónových sietí vieme, že viac parametrov môže znamenať väčšiu aproximačnú schopnosť. V našej úlohe však viac parametrov znamená aj väčší priestor, ktorý budeme neskôr prehľadávať genetickým algoritmom.

Preto architektúra nie je len implementačný detail. Je to kompromis medzi:

- kapacitou siete,
- rýchlosťou inferencie,
- počtom parametrov,
- veľkosťou vyhľadávacieho priestoru,
- a praktickou trénovateľnosťou.

Príliš malá sieť môže byť lacná, ale nemusí vedieť zachytiť vzťah medzi observation a akciou. Príliš veľká sieť môže mať kapacitu, ale pre GA bude znamenať dlhší genóm, viac dimenzií a náročnejšie hľadanie.

V texte treba vysvetliť, že naším cieľom nie je vybrať najväčšiu možnú sieť. Chceme nájsť sieť, ktorá je dosť veľká na to, aby vedela reprezentovať použiteľnú policy, ale dosť malá na to, aby sa dala prakticky trénovať v našich experimentoch.

Predpoklady neuronovej siete sú jasné, máme fixnú veľkosť vstupu aj výstupu, - výstupná vrstva musí rešpektovať zvolený action mode. teda steering tanh a gas, bereak sigmoid

Kandidáti, ktoré treba v texte predstaviť:

- všetky rôzne veľkosti na ktorom sme robili grid search
- väčšie siete ako referenciu alebo horný odhad kapacity,
- kombinácie aktivačných funkcií relu, tanh, sigmoid
- sigmoid v hidden vrstvách skôr ako negatívnejšiu alebo menej vhodnú kontrolu, ak sa naň odkazujeme z experimentov.

Treba tiež vysvetliť aktivačné funkcie:

- ReLU je jednoduchá, výpočtovo lacná a často používaná v hidden vrstvách,
- tanh má výstup v rozsahu `[-1, 1]`, čo intuitívne sedí najmä k riadeniu typu steering,


Pri finálnom texte bude treba citovať všeobecné zdroje k MLP, ReLU, tanh a univerzálnej aproximácii. V tejto osnove si stačí poznačiť, že architektúru nevyberáme iba pocitovo, ale ako kompromis medzi teóriou, výpočtovou praktickosťou a našimi experimentmi.

## 3. Supervised learning ako test kapacity siete

Keďže chceme, aby agent vykonával podobné akcie ako človek v rovnakých situáciách, vieme si pripraviť dáta:

```text
observation -> human action
```

Človek jazdí trať, systém ukladá observation a súčasne uloží aj akciu hráča. Tak vznikne dataset dvojíc vstup-výstup. Na týchto dátach vieme trénovať neurónovú sieť supervised spôsobom.

Táto časť je dôležitá, ale treba ju interpretovať opatrne. Supervised tréning tu nie je ešte dôkaz, že máme ideálneho autonómneho agenta. Je to najmä test kapacity siete:

- či sieť vie spracovať observation,
- či vie predikovať akcie podobné človeku,
- či loss počas tréningu klesá,
- či menšie architektúry nie sú príliš slabé,
- či väčšie architektúry prinášajú zmysluplný zisk.


Treba ukázať:

- ako vznikol dataset,
- aký bol vstup a výstup siete,
- akú loss funkciu sme použili,
- ako vyzeral priebeh loss,
- ako sa líšili architektúry,
- prečo výsledok podporuje výber `32x16` a `48x24`.

Najskôr treba ukázať všetky možné kombinácie ktoré sme skusili a potom sa zamerať na naše `32x16` a `48x24` relu tanh

Interpretácia:

- `32x16` je dobrý lacný experimentálny baseline,
- má menej parametrov,
- znamená menší vektorový priestor pre GA,
- umožňuje rýchlejšie porovnávať reward funkcie a hyperparametre,
- `48x24` je silnejší kandidát, keď nám už nejde iba o lacné testy, ale o kvalitnejší finálny tréning.

V texte treba zdôrazniť, že supervised výsledok odpovedá na otázku “vie sieť napodobniť dáta?”, nie na otázku “je to najlepšia jazdná policy?”.

## 4. Behavior cloning a butterfly efekt

Po supervised tréningu môžeme nechať natrénovaného agenta jazdiť v prostredí. Tu sa ukáže dôležitý rozdiel medzi predikciou na datasete a jazdou v closed-loop režime.

Na datasete sieť dostane observation, ktorú vytvoril človek počas svojej jazdy. Pri reálnej jazde však agent svojimi akciami mení budúce pozorovania. Ak spraví malú chybu, auto sa posunie trochu inak. O pár sekúnd môže byť v stave, ktorý sa v trénovacích dátach takmer nevyskytoval.

Toto je vhodné vysvetliť cez butterfly efekt:

- malá odchýlka v akcii na začiatku jazdy,
- mierne iná trajektória,
- iná poloha pred ďalšou zákrutou,
- iné lidarové vzdialenosti,
- ďalšie odlišné akcie,
- nakoniec úplne iný priebeh jazdy.

Pre grafy sa hodí:

- mapa s trajektóriami učiteľa,
- trajektória supervised agenta,
- miesto kontaktu alebo odchýlky,
- prípadne obrázok, kde krátky zásah do steeringu vytvorí výrazne inú trajektóriu.

Táto časť má byť úprimná. Supervised agent môže vyzerať sľubne a na jednoduchých mapách jazdiť celkom rozumne, ale zatiaľ neriešime robustnosť siete ako finálneho pretekárskeho agenta. Sledujeme, že sieť má schopnosť naučiť sa z dát, no zároveň vidíme, že samotné kopírovanie nie je celé riešenie.

## 5. Imitation learning ako korekcia behavior cloningu

Behavior cloning trpí tým, že agent je trénovaný najmä na stavoch, do ktorých sa dostal človek. Ak sa agent sám dostane do odlišného stavu, nemusí vedieť, čo robiť. Imitation learning tento problém zmierňuje tým, že počas zberu dát alebo tréningu miešame správanie človeka a agenta.

Myšlienka:

- agent začne konať podľa svojej policy,
- človek môže jeho správanie korigovať,
- ukladáme dáta aj zo stavov, ktoré vznikli v dôsledku agentových vlastných chýb,
- sieť sa učí nielen ideálnu jazdu, ale aj návrat z menej ideálnych situácií.

V texte treba vysvetliť váhu miešania akcií:

- na začiatku môže dominovať človek,
- postupne môže mať väčší vplyv agent,
- graf váhy ukáže, ako sa menil pomer medzi ľudskou a agentovou akciou.

Výstupy tejto časti:

- priebeh loss siete,
- graf miešania akcií,
- jazda agenta po imitácii,
- porovnanie s čistým behavior cloningom.

Interpretácia musí zostať opatrná. Imitation learning je lepší než čisté statické napodobňovanie, pretože rozširuje množinu stavov, ktoré agent videl. Stále však trénuje najmä podobnosť s človekom. Nie je to priamy tréning optimálnej policy podľa cieľa, ktorý nás zaujíma v pretekoch.

## 6. Limity napodobňovania

Tu treba kapitolu na chvíľu spomaliť a jasne pomenovať, čo sme sa naučili.

Supervised learning a imitation learning ukázali, že:

- observation obsahuje informácie, z ktorých sa dá vybrať rozumná akcia,
- neurónová sieť má dostatočnú kapacitu na mapovanie observation na action,
- architektúry ako `32x16` a `48x24` sú použiteľné,
- agent vie napodobniť niektoré časti ľudskej jazdy.

Zároveň však platí:

- agent sa učí podobať človeku, nie optimalizovať čas,
- bez veľmi kvalitných a variabilných dát sa ľahko dostane mimo distribúciu,
- ak človek nejazdí optimálne, agent nemá dôvod byť lepší,
- aj malá chyba môže spôsobiť inú trajektóriu a ďalšie chyby,
- supervised loss nemusí korelovať s tým, či agent dokončí trať.

Táto časť má byť most k ďalšej fáze. Ak chceme, aby agent nebol iba napodobňovač, musíme ho hodnotiť podľa toho, ako si počínal v prostredí. Potrebujeme reward, fitness alebo ranking.

Dobrá prechodová veta:

```text
Napodobňovanie nám ukázalo, že sieť vie konať rozumne. Neodpovedalo však na otázku, ako ju prinútiť jazdiť lepšie než dáta, z ktorých sa učila.
```

## 7. Od napodobňovania k hodnoteniu správania

Keď už máme policy reprezentovanú neurónovou sieťou, môžeme ju spustiť v prostredí a pozerať sa, ako jazdí. Potrebujeme však číselne alebo porovnávaco určiť, ktorá policy je lepšia.

Najjednoduchší cieľ v Trackmanii je:

- dostať sa do cieľa,
- a spraviť to čo najrýchlejšie.

Intuitívne by sa to dalo zapísať ako:

```text
(finished, -time)
```

To znamená: najprv nás zaujíma, či agent dokončil trať, a medzi dokončenými jazdami nás zaujíma čas.

Problém je, že náhodne inicializovaný neuroevolučný agent sa do cieľa zo začiatku takmer určite nedostane. Ak by sme hodnotili iba finish a čas, väčšina populácie by vyzerala rovnako zle. Algoritmus by nemal dostatočný signál, ktorým smerom sa zlepšovať.

Tu treba vysvetliť rozdiel medzi:

- reward ako jedným číslom,
- fitness ako hodnotením jedinca,
- lexikografickým rankingom ako porovnaním viacerých metrík v poradí priority.

Dôležitá myšlienka práce:

- nechceme všetko miešať do jedného čísla pomocou ľubovoľných konštánt,
- nechceme reward typu `1000 * finish + 37 * progress - 0.12 * time`, ak tieto váhy nevieme obhájiť,
- radšej používame lexikografické poradie, kde je jasné, ktorá metrika má prioritu.

Toto je dobré miesto pomenovať “woodoo konštanty”. Nie hanlivo voči iným prístupom, ale ako praktický problém: ak veľa cieľov zmiešame cez ručne vybrané váhy, ťažko sa vysvetľuje, prečo práve tieto váhy reprezentujú naše priority.

## 8. Postupné skladanie lexikografickej ranking funkcie

Túto časť treba napísať ako postupné skladanie prípadu. Každý nový člen ranking tuple rieši konkrétny problém, ktorý sa objavil pri predchádzajúcej verzii.

### 8.1 Iba finish a čas

Prvý nápad:

```text
(finished, -time)
```

Problém:

- v počiatočných generáciách takmer nikto nedokončí trať,
- agenti bez finishu sa od seba zle rozlišujú,
- GA nemá jemný signál, ktorý by posúval populáciu dopredu.

Interpretácia:

- cieľ je správny, ale signál je príliš riedky,
- potrebujeme rozdeliť úlohu na menšie časti.

### 8.2 Pridanie progressu

Ďalší krok:

```text
(finished, progress)
```

`progress` znamená plynulý geometrický postup po trati v percentách. Agent už nemusí dokončiť celú trať, aby bol odlíšiteľný od slabšieho jedinca. Ak jeden agent prejde 30 % trate a druhý 10 %, vieme ich porovnať.

Toto rieši problém riedkeho signálu:

- agent dostáva spätnú väzbu po častiach,
- GA vie preferovať jedincov, ktorí sa dostali ďalej,
- populácia sa vie postupne posúvať k finishu.

Problém:

- agent sa môže naučiť dostať do cieľa, ale nemusí sa ponáhľať,
- ak je čas až druhoradý alebo chýba, môže vzniknúť pomalý, opatrný agent.

### 8.3 Pridanie času

Ďalší krok:

```text
(finished, progress, -time)
```

Tým hovoríme:

- najprv finish,
- potom ako ďaleko sa agent dostal,
- potom preferujeme kratší čas.

Toto dáva zmysel, pretože Trackmania je time-attack hra. Nestačí trať prejsť. Chceme ju prejsť rýchlo.

Problém:

- pri rovnakom alebo podobnom progresse sa môže oplatiť skončiť rýchlo,
- agent môže preferovať kamikaze stratégiu,
- riskantná jazda môže vyzerať dobre, ak rýchlo dosiahne určitý progress a potom narazí.

### 8.4 Pridanie crash metriky

Preto pridáme informáciu o náraze:

```text
(finished, progress, -time, -crashes)
```

alebo alternatívne:

```text
(finished, progress, -crashes, -time)
```

Tu je dôležité vysvetliť rozdiel v poradí:

- ak dáme `-time` pred `-crashes`, agent pri rovnakom finish/progress najprv preferuje rýchlosť a až potom bezpečnosť,
- ak dáme `-crashes` pred `-time`, agent pri rovnakom finish/progress najprv preferuje bezpečnosť a až potom čas.

Tým vznikne presne konflikt, ktorý sme očakávali:

- rýchlejší variant môže byť riskantnejší,
- bezpečnejší variant môže byť pomalší,
- ani jedna priorita nie je absolútne “pravdivá”, je to rozhodnutie podľa cieľa práce.

Experimentálne výsledky ukazujú, že aktuálne najlepší praktický základ je:

```text
(finished, progress, -time, -crashes)
```

Táto verzia našla dobrý kompromis medzi dokončovaním trate, progresom, časom a penalizáciou nárazov. V texte treba ukázať grafy reward sweepu a jasne vysvetliť, prečo nevyberáme iba podľa jedného najlepšieho času, ale podľa celkového správania populácie.

### 8.5 Kontrolný návrat k reinforcement learningu

V tomto bode už máme oveľa lepšiu predstavu o tom, čo má hodnotenie jazdy obsahovať. To je dobré miesto na krátky návrat k bakalárskej práci a k reinforcement learningu.

V bakalárskej práci bol použitý PPO smer, ale výsledky neboli presvedčivé a reward nebol dostatočne dobre vysvetlený ani experimentálne obhájený. Spätne sa preto dá povedať, že problém nemusel byť iba v samotnom algoritme. Veľká časť problému mohla byť v tom, že agent nedostával vhodne formulovaný signál.

Tu môžeme ukázať kontrolný RL experiment:

- použijeme rovnakú logiku observation/action prostredia,
- použijeme reward postavený na zmysluplných metrikách auta,
- napríklad dokončenie trate, prejdená vzdialenosť alebo progress a čas,
- porovnáme PPO, SAC a TD3 ako moderné RL algoritmy,
- sledujeme, či sa agent pri rozumnejšom rewarde začne učiť.

Dôležitá interpretácia:

- PPO sa v našom reward-equivalent sweep-e vedelo naučiť dokončiť trať,
- SAC a TD3 v rovnakom nastavení zlyhali,
- tým sa ukazuje, že lepšia formulácia rewardu reálne mení správanie učenia,
- zároveň to neznamená, že RL je v tejto práci hlavná cesta.

Táto odbočka má v texte jasnú úlohu. Nevraciame sa k RL preto, aby sme prerušili GA líniu, ale aby sme ukázali poučenie z bakalárskej práce: ak je zle navrhnuté hodnotenie, aj vhodný algoritmus môže vyzerať neúspešne. Keď hodnotenie postavíme na jasnejších metrikách, PPO už dáva sľubný signál.

Prečo potom pokračujeme GA:

- GA prirodzene pracuje s vyhodnotením celej epizódy,
- lexikografické rankingy sa v GA používajú čitateľnejšie než v bežnom scalar RL rewarde,
- v našich experimentoch bol GA stabilnejší praktický smer,
- RL ostáva porovnávacia vetva a dôležitý návrat k bakalárskej práci, nie hlavný tréningový postup.

Do tejto časti patria grafy:

- RL progress cez epizódy,
- best finish time cez epizódy,
- porovnanie PPO, SAC a TD3,
- stručné porovnanie fixného a variabilného physics tick profilu.

Túto časť netreba písať príliš dlho. Má byť krátka, ale dôležitá. Ukazuje, že nový reward/ranking pohľad neopravuje iba GA experimenty, ale aj lepšie vysvetľuje, prečo starší RL prístup nefungoval a za akých podmienok už fungovať začne.

## 9. Konflikt cieľov a Pareto/NSGA-II

Po reward sweep-e už vidíme, že problém nie je len technický. Viaceré ciele si reálne protirečia.

Chceme:

- finish,
- vysoký progress,
- nízky čas,
- málo crashov,
- stabilné správanie,
- a ideálne aj rýchly tréning.

Niektoré z týchto cieľov idú proti sebe. Rýchla jazda často znamená viac rizika. Bezpečná jazda môže byť pomalšia. Agresívna explorácia môže rýchlejšie nájsť prvý finish, ale zhoršiť stabilitu.

Toto vedie k otázke: nemali by sme prestať nútiť všetky ciele do jedného poradia?

Tu vstupuje myšlienka Pareto fronty a NSGA-II:

- jedinec je lepší, ak dominuje iného jedinca vo viacerých cieľoch,
- nemusíme dopredu určiť jednu presnú kombináciu váh,
- môžeme udržiavať front riešení, ktoré reprezentujú rôzne kompromisy.

V texte treba vysvetliť, že toto bol logický pokus. Ak sa ciele bijú, Pareto optimalizácia vyzerá ako prirodzenejšie riešenie než pevný tuple.

Výsledok však treba formulovať pravdivo:

- pôvodné MOO/NSGA-II pokusy boli slabé alebo nestabilné,
- opravený variant `trackmania_racing` už mal signál,
- napriek tomu neprebil lexikografický baseline,
- preto Pareto vetvu berieme ako užitočný negatívny alebo diagnostický výsledok, nie ako finálnu metódu.

Táto časť je dôležitá pre obhajobu. Ukazuje, že sme nad konfliktom metrík nepremýšľali povrchne. Skúsili sme metódu, ktorá je teoreticky vhodná, ale experimentálne sa v našom nastavení neukázala ako lepšia.

## 10. Hyperparametre genetického algoritmu

Keď máme ranking funkciu, vzniká ďalšia otázka: ako nastaviť samotný genetický algoritmus?

Treba vysvetliť hlavné parametre:

- `population`: koľko jedincov vyhodnotíme v jednej generácii,
- `parents`: koľko najlepších jedincov použijeme ako základ ďalšej generácie,
- `elites`: koľko najlepších jedincov prenesieme bez mutácie,
- `mutation_prob`: pravdepodobnosť mutácie váhy,
- `mutation_sigma`: veľkosť mutačného šumu.

Tieto parametre priamo ovplyvňujú:

- evolučný tlak,
- diverzitu populácie,
- rýchlosť hľadania prvého riešenia,
- schopnosť fine-tuningu,
- riziko predčasnej konvergencie.

Výklad selection parametrov:

- príliš málo rodičov môže znamenať silný tlak, ale menšiu diverzitu,
- príliš veľa rodičov môže rozriediť selekciu,
- príliš veľa elít môže zamraziť populáciu,
- príliš málo elít môže stratiť dobrý smer.

Výklad mutácie:

- vyššia `mutation_prob` a vyššia `mutation_sigma` podporujú exploráciu,
- nižšie hodnoty sú vhodnejšie na fine-tuning,
- veľká mutácia môže zničiť už sľubnú policy,
- príliš malá mutácia nemusí prekonať lokálne chyby.

V experimentoch sa ako praktický baseline ustálilo:

```text
population = 48
parents = 14
elites = 2
mutation_prob = 0.10
mutation_sigma = 0.25
```

Treba zároveň korektne povedať, že refined grid papierovo najlepšie podporil `48/14/1`, ale pre ďalšie experimenty používame `48/14/2`, pretože dve elity dávajú prakticky väčší zmysel pre diverzitu zachovaných riešení. Toto je dobré miesto ukázať, že nie sme otrokom jednej tabuľky, ale interpretujeme výsledky v kontexte tréningu.

## 11. Tréningové vylepšenia

Po výbere ranking funkcie a základných hyperparametrov sa kapitola môže presunúť od otázky “čo funguje?” k otázke “ako to zlepšiť?”.

Túto časť treba uviesť ako sériu hypotéz:

- vieme tréning zrýchliť,
- vieme ho stabilizovať,
- vieme získať robustnejšieho agenta,
- vieme lepšie využiť už nájdené dobré jedince,
- vieme využiť supervised model ako počiatočný odhad.

### 11.1 Mirroring

Myšlienka mirroringu:

- niektoré informácie v observation sú symetrické vzhľadom na os auta,
- ak by bola trať zrkadlová, vhodná akcia by mala byť tiež zrkadlová,
- preto vieme observation zrkadliť, pustiť cez policy a potom zrkadliť steering späť.

Očakávanie:

- agent nebude pretrénovaný na preferovanú stranu zatáčania,
- mal by byť robustnejší,
- mohol by lepšie generalizovať.

Výsledok treba opísať opatrne:

- both-mirror eval sa vedel trénovať,
- ale každá generácia stála približne dvojnásobný eval čas,
- na tréningovej mape nebolo jasne viditeľné zlepšenie kvality,
- holdout test zatiaľ nepotvrdil generalizačný prínos.

Interpretácia:

- myšlienka je rozumná,
- experiment nie je pozitívny dôkaz robustnosti,
- treba ju uviesť ako diagnostický pokus.

### 11.2 Mirror probability

Both-mirror eval je drahý, preto vznikla lacnejšia verzia: mirror evaluáciu použiť iba s určitou pravdepodobnosťou.

Hypotéza:

- nezvýšiť cenu tréningu dvojnásobne,
- stále občas filtrovať agentov, ktorí sa správajú zle v zrkadlených situáciách.

Výsledok:

- v našich doterajších experimentoch sa mirror probability neukázala presvedčivo,
- nepriniesla jasný thesis-grade pozitívny efekt.

Použitie v texte:

- ako dobrý príklad rozumnej hypotézy,
- ale nie ako finálne odporúčané zlepšenie.

### 11.3 Elite cache

Elite cache vychádza z jednoduchej otázky: ak najlepšieho jedinca prenášame do ďalšej generácie, musíme ho vždy znovu vyhodnocovať?

Pri fixed100 deterministickom prostredí by opakovaný rollout toho istého jedinca mal dať rovnaký výsledok. Pri variabilnom fyzikálnom kroku však môže ten istý agent dopadnúť trochu inak. Preto vznikla aj obava:

- nechránime náhodou jedinca, ktorému sa raz zadaril riskantný run,
- nebude potom šíriť zlé gény ďalej,
- nestratíme tým férovosť vyhodnotenia?

Experimentálne výsledky sú tu dôležité:

- pri variable tick bez cache neboli finishery,
- pri variable tick s elite cache sa objavili finishery a dobrý najlepší čas,
- elite cache sa ukázal ako najsilnejší pozitívny tréningový vylepšovák.

Interpretácia:

- cache šetrí výpočtový čas,
- zároveň funguje ako ochrana dobrého riešenia v šumovejšom prostredí,
- v našich dátach sa obava nepotvrdila ako problém,
- toto je thesis-grade pozitívny výsledok.

### 11.4 Mutation decay

Z mutation gridu vyplýva, že rôzne hodnoty mutácie majú rôzne úlohy:

- vyššia mutácia je dobrá na exploráciu,
- nižšia mutácia je lepšia na fine-tuning,
- jedna fixná hodnota je kompromis.

Preto vzniká mutation decay:

- začneme s exploračnými hodnotami,
- postupne ich znižujeme k fine-tuning hodnotám.

Problém pôvodného decay:

- decay môže znížiť mutácie príliš skoro,
- agent ešte nemusí mať ani prvého finishera,
- tým sa explorácia obmedzí skôr, než existuje riešenie, ktoré sa oplatí dolaďovať.

Riešenie:

- first-finish triggered decay,
- pred prvým finishom držíme exploračné hodnoty,
- po prvom finishi dopočítame decay tak, aby sme ku koncu tréningu smerovali k cieľovým fine-tuning hodnotám.

Interpretácia výsledku:

- first-finish decay je rozumný a obhájiteľný nápad,
- v našich výsledkoch je skôr tradeoff než jednoznačný víťaz,
- môže zlepšiť best time, ale nemusí zlepšiť stabilitu populácie.

### 11.5 Viac dotykov a crash count

Pôvodná logika crashu je binárna: agent narazí a epizóda končí. To je prísne a nie vždy zodpovedá intuícii z jazdy. V skutočnej hre môže malý náraz znamenať chybu, ale nie nutne koniec celej jazdy.

Preto vznikol experiment:

- povoliť viac dotykov,
- po náraze auto spomaliť a odraziť,
- logovať počet crashov,
- sledovať, či tréning nebude menej krehký.

Interpretácia:

- max touches experiment bol diagnostický,
- nebol jasným finálnym zlepšením,
- pomohol však lepšie rozmýšľať o rozdiele medzi “náraz ako koniec” a “náraz ako chyba v jazde”.

### 11.6 Supervised-seeded GA

Po supervised a BC experimentoch vznikla prirodzená otázka: ak supervised model už vie jazdiť aspoň čiastočne rozumne, prečo GA začínať úplne náhodne?

Myšlienka:

- natrénovať supervised model,
- vložiť ho ako základ prvej populácie,
- nevložiť 48 rovnakých kópií,
- ale vytvoriť populáciu pomocou náhodných mutácií okolo tohto modelu.

Dôležitý rozdiel:

- sparse mutácia zachovala BC model príliš silno a bola negatívna kontrola,
- dense weight-noise mutácia vytvorila dostatočne rôznorodú populáciu a bola pozitívny hybridný výsledok.

Interpretácia:

- BC initialization sama o sebe nestačí,
- supervised model je dobrý štart, ale GA musí mať dosť priestoru ho zmeniť,
- dense-noise supervised seeding je thesis-grade pozitívny hybrid,
- treba jasne uviesť, že používa extra ľudské dáta, takže nie je férové prezentovať ho ako čistý GA trik.

## 12. Zloženie najlepšieho 2D asfaltového agenta

Po všetkých predchádzajúcich častiach máme sadu rozhodnutí:

- vieme, akú architektúru použiť ako lacný baseline a ako silnejší variant,
- vieme, že samotné napodobňovanie nestačí,
- máme crash-aware lexikografický ranking,
- máme praktické GA hyperparametre,
- vieme, ktoré vylepšenia sú pozitívne a ktoré iba diagnostické.

Táto časť má ukázať finálne zloženie 2D asphalt tréningu.

Treba vysvetliť:

- prečo nejde o jeden náhodný úspech,
- ako sa každé rozhodnutie opiera o predchádzajúci experiment,
- ktoré časti používame vo finálnej konfigurácii,
- ktoré pokusy nepoužívame a prečo.

Do tejto časti patria grafy:

- focus progress plot najlepšieho tréningu,
- best time vs generation,
- finish/crash/timeout vývoj,
- trajectory overview,
- path najlepšieho agenta,
- prípadne porovnanie s baseline alebo supervised-seeded variantom.

Toto je prirodzený vrchol 2D časti práce. Po ňom môžeme povedať: pre rovinnú asfaltovú trať máme dobrý postup. Ďalšia otázka je, čo sa stane, keď svet prestane byť rovinný alebo homogénny.

## 13. Prechod k výškovým rozdielom

Keď 2D asphalt riešenie funguje, môžeme pridať ďalší problém: trať nemusí byť rovná.

Výškové rozdiely menia situáciu:

- auto môže ísť do kopca alebo z kopca,
- cesta môže byť naklonená,
- lúč v 2D projekcii už nemusí stačiť,
- vzdialenosť k prekážke treba interpretovať vzhľadom na 3D geometriu.

Tu bude treba v práci podrobne vysvetliť:

- ako reprezentujeme výškové zmeny blokov,
- ako sa lúč premieta na novú rovinu bloku,
- ako riešime prechod medzi blokmi,
- ako šetríme výpočet, aby sa realtime systém nespomalil,
- ako sa observation space rozšíri o výškové informácie.

Experimentálne:

- väčšina zistení z 2D GA by mala zostať použiteľná,
- ale veľkosť siete treba znova overiť,
- výškový modul zvyšuje informačný obsah observation,
- preto je prirodzené otestovať aj väčšiu architektúru.

Táto časť by mala nadväzovať ako nový prípad v detektívke. Prípad rovnej trate sme vyriešili. Teraz sa objavuje nový dôkaz: trať má tretí rozmer.

## 14. Prechod k rôznym povrchom

Ďalší rozširujúci problém sú povrchy. Na rôznych povrchoch sa auto nespráva rovnako.

Treba vysvetliť:

- asfalt, tráva, hlina, plast alebo ľad môžu mať odlišnú trakciu,
- rovnaká akcia nemusí mať rovnaký efekt na každom povrchu,
- agent by preto mal vedieť nielen kde sa nachádza cesta, ale aj po čom ide.

Observation sa rozšíri o surface informácie:

- typ aktuálneho povrchu,
- prípadne informácie o povrchoch pred autom,
- alebo iné kompaktné kódovanie, ktoré policy umožní prispôsobiť akcie.

Experimentálne:

- základný ranking a GA logika by mali zostať použiteľné,
- no veľkosť siete a kapacitu treba znovu overiť,
- surface modul je podobný výškovému modulu v tom, že nepridáva novú tréningovú metódu, ale rozširuje stav sveta.

Táto časť uzatvára prechod od jednoduchého sveta k komplexnejšiemu prostrediu. Najprv sme riešili, ako agent jazdí na rovnej asfaltovej trati. Potom pridávame výšku a povrchy, teda vlastnosti, ktoré robia Trackmaniu bohatšou a náročnejšou.

## Obrázky a grafy pre kapitolu

Odporúčané obrázky a grafy:

- supervised architecture/loss comparison,
- počet parametrov jednotlivých architektúr,
- teacher vs agent trajectories,
- butterfly efekt pri malej zmene steeringu,
- graf miešania akcií pri imitation learningu,
- reward sweep pre `(finished, progress)`, `(finished, progress, -time)`, `(finished, progress, -time, -crashes)` a `(finished, progress, -crashes, -time)`,
- best finish time pri reward sweep-e,
- finish count alebo finish rate pri reward sweep-e,
- RL reward-equivalent grafy: PPO/SAC/TD3 progress, outcome a best finish time cez epizódy,
- MOO/Pareto graf ako diagnostický alebo negatívny výsledok,
- heatmapy selection pressure a mutation parametrov,
- focus progress plot pre baseline vs decay,
- variable tick elite cache vs no-cache,
- supervised dense seeding vs random baseline,
- trajectory/path najlepšieho 2D agenta,
- mapový podklad s agentovou trajektóriou,
- neskôr samostatné grafy pre výškový a surface modul.

Pri každom grafe treba v texte odpovedať:

- čo graf ukazuje,
- prečo bol experiment spustený,
- aký problém riešil,
- čo z neho berieme ďalej,
- a čo naopak nebudeme tvrdiť.

## Citácie a zdroje, ktoré bude treba dohľadať

Pri písaní finálneho textu bude treba citovať:

- neurónové siete ako aproximátory funkcií,
- MLP architektúry a aktivačné funkcie,
- ReLU a tanh,
- supervised learning,
- behavior cloning,
- imitation learning a DAgger alebo príbuzné riešenia distribučného posunu,
- genetické algoritmy,
- neuroevolúciu,
- lexikografickú optimalizáciu alebo lexikografické porovnávanie,
- Pareto dominanciu a NSGA-II,
- PPO, SAC a TD3 na úrovni princípu, ak ich budeme interpretovať v RL odbočke,
- reward shaping a problém návrhu reward funkcie,
- prípadne zdroje k autonomous racing alebo game AI, ak ich používame ako kontrast.

Nevymýšľať citácie. Pri každom netriviálnom tvrdení vo finálnom texte musí byť jasné, či je to všeobecná znalosť, naše experimentálne zistenie alebo tvrdenie zo zdroja.

## Čo do tejto kapitoly nepatrí

Do tejto kapitoly ešte nepatrí:

- úplne detailný opis implementácie súborov,
- dlhé výpisy konfigurácií bez interpretácie,
- finálna kapitola výsledkov ako tabuľkový katalóg,
- porovnanie s človekom ako záverečný výsledok,
- príliš hlboký matematický výklad NSGA-II alebo PPO,
- opis Trackmanie ako hry, ktorý už patrí do teórie alebo súvisiacich prác.

Táto kapitola má byť praktické vyšetrovanie tréningu policy. Má vysvetliť, prečo sme skúšali jednotlivé metódy, čo ukázali a ako nás posunuli k ďalšiemu kroku.

## Pracovný verdikt

Táto kapitola má byť jadrom celej práce. Nie je to iba technický popis trénera. Je to príbeh o tom, ako sme postupne zistili, že:

- neurónová sieť vie napodobniť ľudské akcie,
- napodobňovanie samo nestačí,
- agent potrebuje hodnotenie správania v prostredí,
- lexikografický ranking dáva čitateľnejší signál než ručne vážený reward,
- GA je pre túto úlohu praktickejší hlavný tréningový nástroj než testované RL varianty,
- niektoré vylepšenia pomohli, iné boli užitočné negatívne výsledky,
- a stabilný 2D agent je základ, na ktorom sa dá stavať výška a povrchy.

Ak predchádzajúca kapitola bola o tom, ako sme agentovi postavili svet, táto kapitola je o tom, ako sme ho naučili v tomto svete konať.
