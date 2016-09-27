package de.hlrs.starplugin.python.scriptcreation.modules;

import de.hlrs.starplugin.configuration.Configuration_Module;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Module_Collect extends Module {



    public Module_Collect(String Name, int param_pos_x, int param_pos_y) {
        super(Configuration_Module.Typ_Collect, Name, param_pos_x, param_pos_y);
    }

    @Override
    public String[] addtoscript() {
        String[] ExportStringLines = new String[6];
        ExportStringLines[0] = "#";
        ExportStringLines[1] = "# MODULE: Collect";
        ExportStringLines[2] = "#";
        ExportStringLines[3] = this.Name + "=Collect()";
        ExportStringLines[4] = "network.add(" + this.Name + ")";
        ExportStringLines[5] = this.Name + ".setPos(" + Integer.toString(this.param_pos_x) + "," + Integer.
                toString(this.param_pos_y) + ")";

//        ExportStringLines[6] = "#";
//        ExportStringLines[7] = "# set parameter values";
//        ExportStringLines[8] = "#";
//        ExportStringLines[9] = this.Name + ".set_varName(\"" + this.varName + "\" )";
//        ExportStringLines[10] = this.Name + ".set_attribute(\"" + this.attribute + "\")";
//        ExportStringLines[11] = this.Name + ".set_minBound(" + minBound.x + "," + minBound.y + "," + minBound.z + ")";
//        ExportStringLines[12] = this.Name + ".set_maxBound(" + maxBound.x + "," + maxBound.y + "," + maxBound.z + ")";
        return ExportStringLines;
    }
}
