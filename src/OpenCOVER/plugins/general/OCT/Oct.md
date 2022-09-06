
CsvPointCloud
------------------

This plugin converts a csv data table to a point cloud using configurable expressions for x,y and z coordinats as well as for the color.
Also the points are played as animation.

|Parameters|Description|
|----------|-----------|
|HeaderOffset | number of lines to ignore before the actual headline starts in the csv file|
|Delimiter | the delimiter that seperates columns in the csv file|
|X,Y,Z | the coordinates of the points|
|Color | their color value that is converted to rgb values by the chosen color map|
|TimeScaleIndicator | The headline of time steps, multiple occurences are supported, rougher scales values are interpolated to lowest time scale. For each clolumn Rows are expected to be filled if the previous occurence of TimeScaleIndicator is filled in this row. Also finer timescales must be a multiple in lenght of the roughest timescale  |



Example how this could be configured in the config file:

```xml
    <X value="CurrPosX_PCS *10 + cos(Winkel * 1.8 * 0.0174533) * 1500" />
    <Y value="CurrPosY_PCS *10 + sin(Winkel * 1.8 * 0.0174533) * 1500" />
    <Z value="CurrPosZ_PCS * 10 + OCTArraymittelwert" />
    <Color value="Leistung" />
    <HeaderOffset value="0" />
    <TimeScaleIndicator value="Name" />
    <Delimiter value="," />
```

The corresponding csv file could look like this:

```csv
    Name,CurrPosX_PCS,CurrPosY_PCS,CurrPosZ_PCS,Name,Leistung,Winkel
    ,,,,,,
    0,0,0,0,0,0,0
    1,0,0,0,0.5,1,45
    2,0,0,0,1,2,90
    3,0,0,0,1.5,3,135
    ,,,,2.5,5,180
    ,,,,3,6,225
    ,,,,3.5,7,270
    ,,,,4,8,315
```

This example creates a ring of 8 points.
