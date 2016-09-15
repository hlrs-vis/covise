package de.hlrs.starplugin.python.scriptcreation.modules;

import de.hlrs.starplugin.configuration.Configuration_Module;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Module_MinMax extends Module {

//    private Vec NumBucks;

    public Module_MinMax(String Name, int param_pos_x, int param_pos_y) {
        super(Configuration_Module.Typ_MinMax, Name, param_pos_x, param_pos_y);
//        this.NumBucks = new Vec(5, 100, 20);
    }

    @Override
    public String[] addtoscript() {
        String[] ExportStringLines = new String[6];
        ExportStringLines[0] = "#";
        ExportStringLines[1] = "# MODULE: MinMax";
        ExportStringLines[2] = "#";
        ExportStringLines[3] = this.Name + "=MinMax()";
        ExportStringLines[4] = "network.add(" + this.Name + ")";
        ExportStringLines[5] = this.Name + ".setPos(" + Integer.toString(this.param_pos_x) + "," + Integer.
                toString(this.param_pos_y) + ")";

//        ExportStringLines[6] = "#";
//        ExportStringLines[7] = "# set parameter values";
//        ExportStringLines[8] = "#";
//        ExportStringLines[9] = this.Name + ".set_NumBuck(" + this.NumBucks.x + "," + this.NumBucks.y + "," + this.NumBucks.z + ")";
        return ExportStringLines;
    }
}
