package de.hlrs.starplugin.python.scriptcreation.modules;

import de.hlrs.starplugin.configuration.Configuration_Module;
import de.hlrs.starplugin.util.Vec;

/**
 *
 *@author Weiss HLRS Stuttgart
 */
public class Module_CuttingSurface extends Module {

    private Vec vertex;
    private Vec point;
    private float scalar;
//    private boolean skew;
//    private int option;
//    private boolean gennormals;
//    private boolean genstrips;
//    private boolean genDummyS;
//    private float vertex_ratio;


    public Module_CuttingSurface(String Name, int param_pos_x, int param_pos_y, Vec vertex, Vec point, float scalar) {
        super(Configuration_Module.Typ_CuttingSurface, Name, param_pos_x, param_pos_y);
        this.vertex = vertex;
        this.point = point;
        this.scalar = scalar;
//        this.skew = false;
//        this.option = 1;
//        this.gennormals = false;
//        this.genstrips = true;
//        this.genDummyS = true;
//        this.vertex_ratio = 4;
    }

    @Override
    public String[] addtoscript() {
        String[] ExportStringLines = new String[12];
        ExportStringLines[0] = "#";
        ExportStringLines[1] = "# MODULE: CuttingSurface";
        ExportStringLines[2] = "#";
        ExportStringLines[3] = this.Name + "=CuttingSurface()";
        ExportStringLines[4] = "network.add(" + this.Name + ")";
        ExportStringLines[5] = this.Name + ".setPos(" + Integer.toString(this.param_pos_x) + "," + Integer.
                toString(this.param_pos_y) + ")";
        ExportStringLines[6] = "#";
        ExportStringLines[7] = "# set parameter values";
        ExportStringLines[8] = "#";
        ExportStringLines[9] = this.Name + ".set_vertex(" + this.vertex.x + "," + this.vertex.y + "," + this.vertex.z + ")";
        ExportStringLines[10] = this.Name + ".set_point(" + this.point.x + "," + this.point.y + "," + this.point.z + ")";
        ExportStringLines[11] = this.Name + ".set_scalar(" + this.scalar + ")";
//        ExportStringLines[12] = this.Name + ".set_skew(\"" + this.skew + "\")";
//        ExportStringLines[13] = this.Name + ".set_option(" + this.option + ")";
//        ExportStringLines[14] = this.Name + ".set_gennormals(\"" + this.gennormals + "\")";
//        ExportStringLines[15] = this.Name + ".set_genstrips(\"" + this.genstrips + "\")";
//        ExportStringLines[16] = this.Name + ".set_genDummyS(\"" + this.genDummyS + "\")";
//        ExportStringLines[17] = this.Name + ".set_vertex_ratio(" + this.vertex_ratio + ")";
        return ExportStringLines;
    }
}
