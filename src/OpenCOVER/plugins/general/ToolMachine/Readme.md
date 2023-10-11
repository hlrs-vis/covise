Toolmachine with Opc-Ua connetion
----------------------------------

This plugin provides utility to synchronize virtual tool machines with their rea counterparts via Opc-Ua

The configuration of the digital twin is done in VRML via the ToolMachine node:

```c++
DEF machine ToolMachine{
    MachineName "LaserTec" //This should be the name of the Opc-Ua server
    AxisNames ["X","Y","Z","A","C"] //first 3 axcees are expect eto be linear, other rotational
    OPCUANames ["ENC2_POS|X","ENC2_POS|Y","ENC2_POS|Z","ENC2_POS|A","ENC2_POS|C"] //the name of the variable containing axis position on the Opc-Ua server
    AxisOrientations [1 0 0, 0 0 1, 0 1 0, -1 0 0, 0 0 1] //in which direction each axis is moved or around which it is rotated
    Offsets [-406.401596,324.97962, 280.54943,0,0,0,0,0,0] //offset to configure base position
    AxisNodes [USE X, USE Y, USE Z, USE A, USE C] //the VRML nodes that represents an axis
    TableNode USE C //the VRML node where the workpiece will be placed
    ToolHeadNode USE ToolHeadNode //the VRML node where the tool interacts with the workpiece
    OpcUaToVrml 0.001 //scale factor
    VisualizationType "Currents" //optional visualization, e.g. currents in actuators
}
```

