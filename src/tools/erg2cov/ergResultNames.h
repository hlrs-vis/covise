/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

const char *resultNames[] = {
    "Wanddicke [mm]", // 0
    "Layer", // 1
    "Temperatur der Fließfront [°C]", // 2
    "Druckverlust [bar]", // 3
    "Schubspannung an Fließfront [Pa]", // 4
    "Geschwindigkeit [mm/s]", // 5
    "Temperatur Ende Formfüllung [°C]", // 6
    "Druck Ende Formfüllung [bar]", // 7
    "Füllung - Temperatur [°C]", // 8
    "Füllung - Druck [bar]", // 9
    "Füllung - Schubspannung [Pa]", // 10
    "Füllung - Geschwindigkeit [mm/s]", // 11
    "Nicht definiert", // 12
    "Nicht definiert", // 13
    "Erstarrte Randschicht während Füllung [mm]", // 14
    "Verbleibende Wanddicke [%]", // 15
    "Kern [mm]", // 16
    "Kern [%]", // 17
    "Nicht definiert", // 18
    "Nicht definiert", // 19
    "Nachdruck - Druck [bar]", // 20
    "Nachdruck - Temperatur [°C]", // 21
    "Nachdruck - Schmelzetemperatur [°C]", // 22
    "Nachdruck - Fließgeschwindigkeit [mm^3/s]", // 23
    "Nachdruck - Freier Querschnitt [mm]", // 24
    "Nachdruck - Volumetrische Schwindung [%]", // 25
    "Nicht definiert", // 26
    "Nicht definiert", // 27
    "Nicht definiert", // 28
    "Füllung - Max. Temperatur [°C]", // 29
    "Füllung - Max. Schergeschwindigkeit [1/s]", // 30
    "Nicht definiert", // 31
    "Faserorientierung in Layer Nr. #", // 32
    "Füllung - Wandtemp. pos. Seite [°C]", // 33
    "Füllung - Wandtemp. neg. Seite [°C]", // 34
    "Kühlung - Wandtemp. pos. Seite [°C]", // 35
    "Kühlung - Wandtemp. neg. Seite [°C]", // 36
    "Nicht definiert", // 37
    "Nicht definiert", // 38
    "Füllug [%]", // 39
    "Füllzeit [s]", // 40
    "Füllpfad", // 41
    "Verzug bei Entformung [mm]", // 42
    "Gesamtverzug [mm]", // 43
    "Bindenähte", // 44
    "Temperatur in Layer Nr. # [°C]", // 45
    "Geschwindigkeit in Layer Nr. # [mm/s]", // 46
    "Temperatur in Werkzeug in Layer Nr. # [°C]", // 47
    "Temperatur aus Werkzeug in Layer Nr. # [°C]", // 48
    "Temperatur [°C] - V Nr. #", // 49
    "Druck [bar] - V Nr. #", // 50
    "Schergeschwindigkeit [Pa] - V Nr. #", // 51
    "Geschwindigkeit [mm/s] - V Nr. #", // 52
    "Temperatur bei 100% Füllung [°C] - V Nr. #", // 53
    "Druck bei 100% Füllung [bar] - V Nr. #", // 54
    "Füllung [%] - V Nr. #", // 55
    "Füllzeit [s] - V Nr. #", // 56
    "Füllpfad - V Nr. #", // 57
    "Temperatur [°C]", // 58
    "Druck [bar]", // 59
    "Schubspannung [Pa]", // 60
    "Geschwindigkeit [mm/s]", // 61
    "Temperatur [°C]", // 62
    "Druck [bar]", // 63
    "Schubspannung [Mpa]", // 64
    "Geschwindigkeit [mm/s]", // 65
    "Erstarrte Randschicht [%]", // 66
    "Maximale Temperatur [°C]", // 67
    "Max. Schergeschwindigkeit [1/s]", // 68
    "Kühlzeit [s]", // 69
    "Wanddicke [mm]", // 70
    "Zusatzinformation", // 71
    "Verzug in X-Richtung [mm]", // 72
    "Verzug in Y-Richtung [mm]", // 73
    "Verzug in Z-Richtung [mm]", // 74
    "Gesamtverzug X-Richtung [mm]", // 75
    "Gesamtverzug Y-Richtung [mm]", // 76
    "Gesamtverzug Z-Richtung [mm]", // 77
    "Einfrierzeit [s]", // 78
    "Siegelzeit [s]", // 79
    "Mögliche Einfallstellen", // 80
    "Füllprobleme", // 81
    "Entformungszeit [s]", // 82
    "Kraft in X-Richtung [N]", // 83
    "Kraft in Y-Richtung [N]", // 84
    "Kraft in Z-Richtung [N]", // 85
    "Verschiebung X-Richtung [mm]", // 86
    "Verschiebung Y-Richtung [mm]", // 87
    "Verschiebung Z-Richtung [mm]", // 88
    "Knoten - Wandstärke [mm]", // 89
    "Sigma XX [N/mm²]", // 90
    "Sigma YY [N/mm²]", // 91
    "Sigma ZZ [N/mm²]", // 92
    "Sigma XY [N/mm²]", // 93
    "Sigma YZ [N/mm²]", // 94
    "Sigma XZ [N/mm²]", // 95
    "Normalspannung [N/mm²]", // 96
    "Tangentiale Spannungen [N/mm²]", // 97
    "V. Mises Spannung [N/mm²]", // 98
    "Fülldifferenz", // 99
    "Kühldifferenz", // 100
    "Hydrostatische Spannung [MPa]", // 101
    "V. Mises Spannungen [MPa]", // 102
    "Randschicht [%]", // 103
    "Freier Querschnitt [%]", // 104
    "Knoten - Wanddicke [mm]", // 105
    "Füllung - Kern [%]", // 106
    "Füllung Kern [s]", // 107
    "Nachdruck - Kern [%]", // 108
    "Nachdruck - Kern [s]", // 109
    "Freier Querschnitt [mm]", // 110
    "Vol. Schwindung [%]", // 111
    "Spritzprägen: Wanddicke", // 112
    "Tensor Spannung X-Wert in Layer #", // 113
    "Tensor Spannung Y-Wert in Layer #", // 114
    "Tensor Spannung X-Y in Layer #", // 115
    "Faserorientierung am Ende in Layer #", // 116
    "Rapid Varianten", // 117
    "Schließkraft (t)", // 118
    "Fließ- und Bindenähte", // 119
    "Not Defined 121", // 120
    "Temperatur", // 121
    "Scorch Index", // 122
    "Not Defined 124", // 123
    "Not Defined 125", // 124
    "Curing Rate", // 125
    "Lokale Schwindung", // 126
    "Verzug", // 127
    "Schmelze Ursprung", // 128
    " ", //129
    " ", //130
    " ", //131
    " ", //132
    " ", //133
    " ", //134
    " ", //135
    " ", //136
    " ", //137
    " ", //138
    " ", //139
    " ", //140
    " ", //141
    " ", //142
    " ", //143
    " ", //144
    " ", //145
    " ", //146
    " ", //147
    " ", //148
    " ", //149
    " ", //150
    " ", //151
    " ", //152
    " ", //153
    " ", //154
    " ", //155
    " ", //156
    " ", //157
    " ", //158
    " ", //159
    " ", //160
    " ", //161
    " ", //162
    " ", //163
    " ", //164
    " ", //165
    " ", //166
    " ", //167
    " ", //168
    " ", //169
    " ", //170
    " ", //171
    " ", //172
    " ", //173
    " ", //174
    " ", //175
    " ", //176
    " ", //177
    " ", //178
    " ", //179
    " ", //180
    " ", //181
    " ", //182
    " ", //183
    " ", //184
    " ", //185
    " ", //186
    " ", //187
    " ", //188
    " ", //189
    " ", //190
    " ", //191
    " ", //192
    " ", //193
    " ", //194
    " ", //195
    " ", //196
    " ", //197
    " ", //198
    " ", //199
    " ", //200
    " ", //201
    " ", //202
    " ", //203
    " ", //204
    " ", //205
    " ", //206
    " ", //207
    " ", //208
    " ", //209
    " ", //210
    " ", //211
    " ", //212
    " ", //213
    " ", //214
    " ", //215
    " ", //216
    " ", //217
    " ", //218
    " ", //219
    " ", //220
    " ", //221
    " ", //222
    " ", //223
    " ", //224
    " ", //225
    " ", //226
    " ", //227
    " ", //228
    " ", //229
    " ", //230
    " ", //231
    " ", //232
    " ", //233
    " ", //234
    " ", //235
    " ", //236
    " ", //237
    " ", //238
    " ", //239
    " ", //240
    " ", //241
    " ", //242
    " ", //243
    " ", //244
    " ", //245
    " ", //246
    " ", //247
    " ", //248
    " ", //249
    " ", //250
    " ", //251
    " ", //252
    " ", //253
    " ", //254
    " ", //255
    " ", //256
    " ", //257
    " ", //258
    " ", //259
    " ", //260
    " ", //261
    " ", //262
    " ", //263
    " ", //264
    " ", //265
    " ", //266
    " ", //267
    " ", //268
    " ", //269
    " ", //270
    " ", //271
    " ", //272
    " ", //273
    " ", //274
    " ", //275
    " ", //276
    " ", //277
    " ", //278
    " ", //279
    " ", //280
    " ", //281
    " ", //282
    " ", //283
    " ", //284
    " ", //285
    " ", //286
    " ", //287
    " ", //288
    " ", //289
    " ", //290
    " ", //291
    " ", //292
    " ", //293
    " ", //294
    " ", //295
    " ", //296
    " ", //297
    " ", //298
    " ", //299
    "Flow Direction", //300
    "Filling - Delta In / Out (°C)", //301
    "Cooling - Delta In / Out (°C)", //302
    "Average Temperature (°C)", //303
    "Average Velocity (mm/s)", //304
    "Average Temperature inside Mold (°C)", //305
    "Average Temperature ouside Mold (°C)", //306
    "wall thickness [mm]", //307
    "Layer", //308
    "Schliesskraft [t]", //309
    "pressure for clamp force calculation [MPa]", //310
    "Temperatur [bar]", //311
    "Restwanddicke [%]", //312
    "Volumenstrom [mm^3/s]", //313
    "Freier Querschnitt [mm]", //314
    "Volumetrische Schwindung [%]", //315
    "Minimum Scorch Index", //316
    "Maximum Scorch Index", //317
    "Average Scorch Index", //318
    "Minimum Curing Rate", //319
    "Maximum Curing Rate", //320
    "Average Curing Rate", //321
    "Curvature Difference - E (°)", //322
    "Curvature Difference - N (°)", //323
    "Reynoldszahl", //324
    "Wärmeübergangskoeff. [W/K/m^2]", //325
    "Globale Entfernung", //326
    "Abstand X", //327
    "Abstand Y", //328
    "Abstand Z", //329
    "Ergebnisbericht", //330
    "Tensor Stress Diag X Value at ejection in Layer #", //331
    "Tensor Stress Diag Y Value at ejection in Layer #", //332
    "Tensor Stress X-Y Value at ejection in Layer #", //333
    " ", //334=
    "Wallthickness [mm]", //335
    "Layers", //336
    "Clamp Force", //337
    "Pressure Clamp Force Calculation [MPa]", //338
    " ", //339
    " ", //340
    "- #", //341
    "- #", //342
    "- #", //343
    "- #", //344
    "- #", //345
    "- #", //346
    NULL
};
