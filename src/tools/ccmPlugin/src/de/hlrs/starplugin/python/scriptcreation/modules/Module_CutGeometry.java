//Not Used
package de.hlrs.starplugin.python.scriptcreation.modules;

import de.hlrs.starplugin.configuration.Configuration_Module;
import de.hlrs.starplugin.util.Vec;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Module_CutGeometry extends Module {

    private int method;
    private int geoMethod;
    private float distance;
    private Vec normal;
    private Vec bottom;
    private float data_min;
    private float data_max;
    private boolean invert_cut;
    private boolean strict_selection;

    public Module_CutGeometry(String Name, int param_pos_x, int param_pos_y, int method, int geoMethod, float distance, Vec normal, Vec bottom, float data_min, float data_max, boolean invert_cut, boolean strict_selection) {
        super(Configuration_Module.Typ_CutGeometry, Name, param_pos_x, param_pos_y);
        this.method = method;
        this.geoMethod = geoMethod;
        this.distance = distance;
        this.normal = normal;
        this.bottom = bottom;
        this.data_min = data_min;
        this.data_max = data_max;
        this.invert_cut = invert_cut;
        this.strict_selection = strict_selection;
    }

    @Override
    public String[] addtoscript() {
        String[] ExportStringLines = new String[18];
        ExportStringLines[0] = "#";
        ExportStringLines[1] = "# MODULE: CutGeometry";
        ExportStringLines[2] = "#";
        ExportStringLines[3] = this.Name + "=CutGeometry()";
        ExportStringLines[4] = "network.add(" + this.Name + ")";
        ExportStringLines[5] = this.Name + ".setPos(" + Integer.toString(this.param_pos_x) + "," + Integer.
                toString(this.param_pos_y) + ")";

        ExportStringLines[6] = "#";
        ExportStringLines[7] = "# set parameter values";
        ExportStringLines[8] = "#";
        ExportStringLines[9] = this.Name + ".set_method(" + this.method + ")";
        ExportStringLines[10] = this.Name + ".set_geoMethod(" + this.geoMethod + ")";
        ExportStringLines[11] = this.Name + ".set_distance(" + this.distance + ")";
        ExportStringLines[12] = this.Name + ".set_normal(" + normal.x + "," + normal.y + "," + normal.z + ")";
        ExportStringLines[13] = this.Name + ".set_bottom(" + bottom.x + "," + bottom.y + "," + bottom.z + ")";
        ExportStringLines[14] = this.Name + ".set_data_min(" + this.data_min + ")";
        ExportStringLines[15] = this.Name + ".set_data_max(" + this.data_max + ")";
        ExportStringLines[16] = this.Name + ".set_invert_cut(\"" + this.invert_cut + "\")";
        ExportStringLines[17] = this.Name + ".set_strict_selection(\"" + this.strict_selection + "\")";

        return ExportStringLines;
    }
}
