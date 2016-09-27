package de.hlrs.starplugin.python.scriptcreation.modules;

import de.hlrs.starplugin.configuration.Configuration_Module;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Module_GetSubset extends Module {

    private String selection;

    public Module_GetSubset(String Name, int param_pos_x, int param_pos_y, String Slection) {
        super(Configuration_Module.Typ_GetSubset, Name, param_pos_x, param_pos_y);
        this.selection = Slection;
    }

    @Override
    public String[] addtoscript() {
        String[] ExportStringLines = new String[10];
        ExportStringLines[0] = "#";
        ExportStringLines[1] = "# MODULE: GetSubset";
        ExportStringLines[2] = "#";
        ExportStringLines[3] = this.Name + "=GetSubset()";
        ExportStringLines[4] = "network.add(" + this.Name + ")";
        ExportStringLines[5] = this.Name + ".setPos(" + Integer.toString(this.param_pos_x) + "," + Integer.
                toString(this.param_pos_y) + ")";

        ExportStringLines[6] = "#";
        ExportStringLines[7] = "# set parameter values";
        ExportStringLines[8] = "#";
        ExportStringLines[9] = this.Name + ".set_selection(\"" + this.selection + "\" )";
        return ExportStringLines;
    }
}
