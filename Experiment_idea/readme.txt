Dobrý deň,
chcel by som Vám poďakovať za krásne cvičenie.
Implementoval som bonus, pričom je jednoduché pridať ďalšie vrstvy aj dodatočne.
Stačí pridať dimenziu do self.dimensions a aktivačnú funkciu do activation_functions vo funkcií MyCar.__init__

Paradoxne, pri použití viac vrstiev sa zdalo, že agent vôbec nefunguje lepšie. 
Pri použití 4 vrstiev (3 váhové matice) sa trénovanie zdá byť pomalšie a často sa jednotlivci pri mutácií pokazia. 
Teda aj v poslednej generácií autá búrali. 

Tento efekt trochu zmiernilo zníženie parametru mutation_probability a deviation, tým sa však o to viac predĺžilo trénovanie.
Je možné, že pri oveľa viac generáciach, pri malej mutation probability a deviation by agent fungoval lepšie, lebo by sa jednalo o akoby jemnejší tréning.
Osobne sa mi zdalo, že kvôli zníženiu mutation_probability a deviation agent ľahko padol do lokálneho minimá, odkiaľ sa už v rozumnom počte generácií nevyhrabal.

Aj napriek číslam sa zdá, že trénovanie pri použití 4 vrstiev (bez zníženia mutation_probability a deviation) išlo v poriadku, 
avšak nevidno ako sa správali jednotlivci. Pri generácií 86 vštci jedinci prešli celú trať bez nabúrania.
no v zápätí v naslédujúcej generácií väčšina nabúrala v prvej zákrute.
Pri 3 vrstvovej neurónke sa až takéto rozdieli nediali, alebo som si to nevšimol


Neural design:   ||3 layer|4 layers| 4 layers, lower mutation_probability & deviation
-------------------------------------------------------------------------------------------
Generation: 1    || 0.47  |  0.83  |  0.45  
Generation: 2    || 0.52  |  0.76  |  0.81  
Generation: 3    || 0.59  |  0.69  |  0.87  
Generation: 4    || 2.08  |  0.82  |  0.80  
Generation: 5    || 5.40  |  0.91  |  2.02  
Generation: 6    || 5.43  |  0.90  |  1.13  
Generation: 7    || 10.38 |  0.88  |  1.43  
Generation: 8    || 15.81 |  1.45  |  2.80  
Generation: 9    || 16.24 |  6.02  |  1.87  
Generation: 10   || 16.25 |  2.31  |  1.83  
Generation: 11   || 16.25 |  7.14  |  1.73
Generation: 12   || 16.29 |  7.09  |  2.32
Generation: 13   || 16.30 |  7.09  |  2.06
Generation: 14   || 16.35 |  7.77  |  2.32
Generation: 15   || 16.34 |  7.94  |  7.25
Generation: 16   || 16.39 |  9.73  |  7.23
Generation: 17   || 16.37 |  9.14  |  7.67
Generation: 18   || 16.39 |  9.66  |  7.61
Generation: 19   || 16.42 |  9.14  |  7.71
Generation: 20   || 16.39 |  8.23  |  7.83
Generation: 21   || 16.44 |  8.22  |  7.79
Generation: 22   || 16.44 |  8.10  |  7.83
Generation: 23   || 16.44 |  9.84  |  7.88
Generation: 24   || 16.44 |  9.74  |  7.78
Generation: 25   || 16.44 |  9.77  |  7.86
Generation: 26   || 16.43 |  9.77  |  7.88
Generation: 27   || 16.41 |  9.79  |  7.88
Generation: 28   || 16.44 |  9.81  |  7.86
Generation: 29   || 16.44 |  9.79  |  7.86
Generation: 30   || 16.44 |  9.90  |  7.90
Generation: 31   || 16.44 |  9.86  |  7.90
Generation: 32   || 16.44 |  9.86  |  7.90
Generation: 33   || 16.41 |  9.89  |  7.97
Generation: 34   || 16.50 |  9.84  |  7.96
Generation: 35   || 16.51 |  9.81  |  7.96
Generation: 36   || 16.50 |  10.23 |  7.96
Generation: 37   || 16.50 |  10.33 |  7.97
Generation: 38   || 16.50 |  16.05 |  8.00
Generation: 39   || 16.50 |  16.08 |  8.00
Generation: 40   || 16.50 |  16.11 |  8.01
Generation: 41   || 16.50 |  16.09 |  8.02
Generation: 42   || 16.50 |  16.11 |  8.01
Generation: 43   || 16.51 |  16.11 |  8.00
Generation: 44   || 16.50 |  16.12 |  8.08
Generation: 45   || 16.50 |  16.16 |  8.01
Generation: 46   || 16.51 |  16.13 |  8.00
Generation: 47   || 16.51 |  16.15 |  8.00
Generation: 48   || 16.48 |  16.17 |  8.01
Generation: 49   || 16.51 |  16.15 |  8.01
Generation: 50   || 16.51 |  16.21 |  8.01
Generation: 51   || 16.50 |  16.18 |  8.00
Generation: 52   || 16.48 |  16.19 |  8.01
Generation: 53   || 16.50 |  16.18 |  8.01
Generation: 54   || 16.50 |  16.23 |  8.02
Generation: 55   || 16.51 |  16.29 |  8.02
Generation: 56   || 16.50 |  16.28 |  8.02
Generation: 57   || 16.50 |  16.29 |  8.04
Generation: 58   || 16.51 |  16.34 |  8.03
Generation: 59   || 16.50 |  16.36 |  8.03
Generation: 60   || 16.50 |  16.36 |  8.03
Generation: 61   || 16.51 |  16.33 |  8.04
Generation: 62   || 16.50 |  16.27 |  8.08
Generation: 63   || 16.50 |  16.35 |  8.08
Generation: 64   || 16.50 |  16.34 |  8.12
Generation: 65   || 16.50 |  16.34 |  8.06
Generation: 66   || 16.50 |  16.37 |  8.08
Generation: 67   || 16.51 |  16.39 |  8.08
Generation: 68   || 16.50 |  16.39 |  8.14
Generation: 69   || 16.51 |  16.35 |  8.11
Generation: 70   || 16.51 |  16.41 |  8.12
Generation: 71   || 16.51 |  16.42 |  8.12
Generation: 72   || 16.51 |  16.39 |  8.13
Generation: 73   || 16.50 |  16.40 |  8.13
Generation: 74   || 16.51 |  16.41 |  8.13
Generation: 75   || 16.51 |  16.42 |  8.13
Generation: 76   || 16.50 |  16.43 |  8.13
Generation: 77   || 16.51 |  16.40 |  8.13
Generation: 78   || 16.52 |  16.40 |  8.14
Generation: 79   || 16.52 |  16.44 |  8.13
Generation: 80   || 16.53 |  16.42 |  8.07
Generation: 81   || 16.50 |  16.41 |  8.09
Generation: 82   || 16.50 |  16.43 |  8.12
Generation: 83   || 16.52 |  16.45 |  8.07
Generation: 84   || 16.52 |  16.44 |  8.11
Generation: 85   || 16.50 |  16.45 |  8.12
Generation: 86   || 16.51 |  16.45 |  8.10
Generation: 87   || 16.52 |  16.45 |  8.07
Generation: 88   || 16.52 |  16.46 |  8.18
Generation: 89   || 16.52 |  16.46 |  8.15
Generation: 90   || 16.52 |  16.46 |  8.17
Generation: 91   || 16.52 |  16.45 |  8.14
Generation: 92   || 16.52 |  16.46 |  8.12
Generation: 93   || 16.52 |  16.46 |  8.10
Generation: 94   || 16.53 |  16.47 |  8.16
Generation: 95   || 16.52 |  16.47 |  8.10
Generation: 96   || 16.55 |  16.49 |  8.11
Generation: 97   || 16.52 |  16.47 |  8.12
Generation: 98   || 16.52 |  16.46 |  8.09
Generation: 99   || 16.54 |  16.51 |  8.10
Generation: 100  || 16.54 |  16.51 |  8.09