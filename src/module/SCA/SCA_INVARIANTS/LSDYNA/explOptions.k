$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$                             SECTION DEFINITIONS                              $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
$$$$$$ Papierdicke, mm: Zeile 2 Spalten 1-4
$*SECTION_SHELL
$         1        16    0.8333       5.0       0.0       0.0         0
$ 0.870E-01 0.870E-01 0.870E-01 0.870E-01      0.00
$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$                             MATERIAL DEFINITIONS                             $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
$$$$$ Materialkonstanten des Papiers:
$$$$$ Dichte, g/mm^3: Zeile 1 Spalte 2
$$$$$ Elastizitaetsmodul, N/mm^2: Zeile 1 Spalte 3
$$$$$ Querkontraktion: Zeile 1 Spalte 4
$$$$$ Fliessspannung, N/mm^2: Zeile 1 Spalte 5
$$$$$ Verfestigungsmodul, N/mm^2: Zeile 1 Spalte 6
*MAT_PLASTIC_KINEMATIC
         2 0.700E-03      21.8  0.300000     0.322      8.01      1.00
      0.00      0.00      0.00
$$$$$ Materialkonstanten des Gummis:
$$$$$ Dichte, g/mm^3: Zeile 1 Spalte 2
$$$$$ Querkontraktion: Zeile 1 Spalte 3
$$$$$ C10, N/mm^2: Zeile 1 Spalte 4
$$$$$ C01, N/mm^2: Zeile 1 Spalte 5
*MAT_MOONEY-RIVLIN_RUBBER
         3 0.150E-02  0.499000     0.137     0.175
    0.0000    0.0000    0.0000    0.0000
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$                                RIGID BOUNDRIES                               $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$                              CONTACT DEFINITIONS                             $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
$$$$$ Reibung Papier-Stempel: Zeile 2 Spalte 1
*CONTACT_FORMING_SURFACE_TO_SURFACE
         2         1         3         3         0         0         0         0
    0.2500    0.0000    0.0000    0.0000   20.0000         0     0.0000.1000E+08
    1.0000    1.0000    0.0000    0.0000    1.0000    1.0000    1.0000    1.0000
         1
$$$$$ Reibung Papier-Gummi: Zeile 2 Spalte 1
$$$$$ Papierdicke, mm: Zeile 5 Spalte 7
*CONTACT_FORMING_SURFACE_TO_SURFACE
         2         3         3         3         0         0         0         0
    0.9500    0.0000    0.0000    0.0000   20.0000         0     0.0000.1000E+08
    1.0000    1.0000    0.0000    0.0000    1.0000    1.0000    1.0000    1.0000
         1
                                                             0.870E-01
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$                              CONTROL OPTIONS                                 $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
$$$$$ Beendigungzeit, ms: Zeile 1 Spalte 1
*CONTROL_TERMINATION
     1.999         0   0.00000   0.00000   0.00000
$
*END
