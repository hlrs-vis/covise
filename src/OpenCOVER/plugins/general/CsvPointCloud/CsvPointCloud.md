
CsvPointCloud
------------------

This plugin converts a csv data table to a point cloud using configurable expressions for x,y and z coordinats as well as for the color.
Also the points are played as animation. To improve loading times the csv file is converted to a binary .oct file when loaded. Using the .oct file afterwards is 
way faster. The colormap can be adjusted in the covise/oct menu.

|Parameters|Description|
|----------|-----------|
|PointSize | size of the points|
|NumPoints | number of points that are reneres with full resolution (see PointReductionCriterium)|
|mapChoice | the color map to apply|
|ScalarData | the scalar data fields to color the points, generated from 'Color' parameter|
|Scale | factor that is applied to X, Y, Z, Right, Forward and Up|
|X,Y,Z | the coordinates of the points|
|Right, Forward, Up | the coordinates to move the tool machine (from external vrml file)|
|HeaderOffset | number of lines to ignore before the actual headline starts in the csv file|
|Delimiter | the delimiter that seperates columns in the csv file|
|Color | ';' seperated list of scalar name an a term to calculate it. The names are added to SalarData to color the points with the corresponding term|
|TimeScaleIndicator | The headline of time steps, multiple occurences are supported, rougher scales values are interpolated to lowest time scale. For each clolumn Rows are expected to be filled if the previous occurence of TimeScaleIndicator is filled in this row. Also finer timescales must be a multiple in lenght of the roughest timescale  |


Example how this could be configured in the config file (.txt file with same name as data file(.csv(.oct))):

```txt
    Scale "1/10^6"
    X "CurrPosX_PCS / 10 + cos((-AbsolutWinkel * 1.8 * 0.0174533)- 4.01426) * 1500"
    Y "CurrPosY_PCS / 10 + sin((-AbsolutWinkel * 1.8 * 0.0174533)- 4.01426) * 1500"
    Z "CurrPosZ_PCS / 10 + OCT1_fast_LREAL"
    Right "CurrPosX_PCS / 10"
    Forward "CurrPosY_PCS / 10"
    Up "CurrPosZ_PCS / 10"
    Color "Leistung;Leistung;Oct Messwert;OCT1_fast_LREAL;Fitywerte;Fitywerte;Winkel;-AbsolutWinkel * 1.8"
    TimeScaleIndicator "Name"
    Delimiter ";"
    HeaderOffset "2"
    PointReductionCriteria "abs((cos((-AbsolutWinkel * 1.8 * 0.0174533) - 4.01426) * dx + sin((-AbsolutWinkel * 1.8 * 0.0174533) - 4.01426) * dy)/sqrt(dx^2 + dy^2)) < 0.1"
    PointSize "1.5"
    NumPoints "8"
    AnimationSkip "827"
```

The corresponding csv file could look like this:

```csv
    Name;Matlab_4erkreuzung;

    Name;CurrPosX_PCS;CurrPosY_PCS;CurrPosZ_PCS;Name;Fitoffset;OCT1_fast_LREAL;OCTMax;OCTMin;Name;Fitywerte;Name;OCTArraymittelwert;AbsolutWinkel;OCT2_fast;Name;Leistung;Medianwert;T_4;Draht_Sollgeschwindigkeit_mm_min;Leistung2;Drahtmotor_Sollgeschwindigkeit_mm_min

    0;-140665;100077;-17;0;-63.568203124999989;-72.5050048828125;-47.5770721;-79.55933;0;-73.261620624312243;0;-63.740612;75;509.8117;0;2500;-67.68317;5623;0;2500;0
    1;-140665;100077;-17;0.05;-63.569729003906239;-73.05810546875;-47.5769;-79.56256;1;-69.942723581465856;0.05;-63.740612;76;509.8117;1;2500;-67.68317;5623;0;2500;0
```


