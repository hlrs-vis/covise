package de.hlrs.starplugin.python.scriptcreation.modules;

import de.hlrs.starplugin.configuration.Configuration_Module;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Module_OpenCover extends Module_Renderer {

    public Module_OpenCover(String Name, int param_pos_x, int param_pos_y) {
        super(Configuration_Module.Typ_OpenCover, Name, param_pos_x, param_pos_y);
    }

    @Override
    public String[] addtoscript() {
        String[] ExportStringLines = new String[6];
        ExportStringLines[0] = "#";
        ExportStringLines[1] = "# MODULE: OpenCOVER";
        ExportStringLines[2] = "#";
        ExportStringLines[3] = this.Name + "=OpenCOVER()";
        ExportStringLines[4] = "network.add(" + this.Name + ")";
        ExportStringLines[5] = this.Name + ".setPos(" + Integer.toString(this.param_pos_x) + "," + Integer.
                toString(this.param_pos_y) + ")";
//        ExportStringLines[6] = "#";
//        ExportStringLines[7] = "# set parameter values";
//        ExportStringLines[8] = "#";
//OpenCOVER_1.set_Viewpoints( "./default.vwp" )
//OpenCOVER_1.set_Viewpoints___filter( "Viewpoints *.vwp/* " )
//OpenCOVER_1.set_Plugins( "" )
//OpenCOVER_1.set_WindowID( 0 )
        return ExportStringLines;
    }
}
