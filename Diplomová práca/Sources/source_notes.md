# Source Notes

## Stabilne jadro citacii

Zaklad teoretickej casti by mali tvorit skor "bibliove" alebo standardne zdroje:

- Russell a Norvig pre definicie AI, agentov a racionalneho spravania.
- Goodfellow, Bengio a Courville pre neuronove siete a hlboke ucenie.
- Cybenko pre univerzalnu aproximaciu.
- Sutton a Barto pre reinforcement learning a MDP.
- Schulman, Haarnoja a Fujimoto pre PPO, SAC a TD3.
- Goldberg, Stanley a Deb pre geneticke algoritmy, neuroevoluciu a NSGA-II.
- Ng, Harada a Russell pre reward shaping.
- Yannakakis a Togelius, Millington a Funge pre AI v hrach.

## Trackmania a racing zdroje

- TMRL je najblizsi prakticky Trackmania RL projekt. Treba ho citovat ako softver/dokumentaciu, nie ako recenzovany clanok.
- Trackmania official/Ubisoft a Openplanet su prakticke zdroje pre opis hry a skriptovatelnosti prostredia.
- `A Driving Model in the Realistic 3D Game Trackmania...` je preprint a online verzia je problematicka, preto je vhodny najviac ako `context_only`.
- Gran Turismo Sophy a GT Sport DRL zdroje su silne porovnanie k racing RL, ale nejde o Trackmaniu.
- TORCS a Simulated Car Racing Championship su vhodne historicke racing benchmarky.

## Sucvisiace prace a kontrast k nasej praci

Pri pisani kapitoly suvisiacich prac bude uzitocne rozlisit:

- image-based end-to-end driving: prirodzene, ale drahe a spaja percepciu s riadenim;
- simulator-based RL/racing: dobre porovnanie k treningu v lacnom prostredi;
- game AI pravidlove/behavior-based metody: historicky zaklad AI v hrach;
- Trackmania community/RL projekty: blizky kontext, ale casto menej formalne alebo menej reprodukovatelne.

## Starostlivost o kvalitu

Niektore stare PDF su uzitocne len na orientaciu. Ak zdroj nema jasne autorstvo, publikaciu, DOI alebo stabilnu URL, nepouzivat ho ako hlavny dokaz. Pri kazdej citacii radsej skontrolovat originalny zdroj este raz pocas pisania konkretneho odseku.

## Audit stareho suboru `sources_old/High_priority`

Subor `Diplomova praca/sources_old/High_priority` nie je PDF ani prazdny marker, ale starsi BibTeX scratchpad. Bol skontrolovany 2026-05-06.

- Duplikaty uz pokryte v `references.bib`: Turing, Russell a Norvig, Sutton a Barto, Goodfellow, Mnih DQN, Bojarski, Fuchs GT Sport DRL, TMRL, Sophy a zaklady pocitacovej grafiky.
- Opravene: Game AI Pro kapitola `Representing and Driving a Race Track for AI Controlled Vehicles` ma autorov Simon Tomlinson a Nic Melder, nie Dave Graham.
- Doplnene do `references.bib`: prehlad deep RL, real-time RL, standardne evolucne zdroje, neuroevolucia, multi-objective neuroevolution, WRC6 end-to-end racing DRL prace a kontextovy Trackmania/Sophy studentsky zdroj.
- Neprevziate ako hlavne zdroje: YouTube kanaly, blogove zdroje, GBX kniznice a vseobecne popularizacne odkazy. Pouzit iba ako implementacny alebo komunitny kontext, ak to bude v texte vyslovene potrebne.
