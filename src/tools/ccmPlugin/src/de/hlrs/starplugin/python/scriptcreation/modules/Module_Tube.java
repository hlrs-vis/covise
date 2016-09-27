package de.hlrs.starplugin.python.scriptcreation.modules;

import de.hlrs.starplugin.configuration.Configuration_Module;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Module_Tube extends Module {

    private float Radius;
    private int Parts;
    private int Option;
//  private boolean LimitRadius;
//    private float max_Radius;

    public Module_Tube(String Name, int param_pos_x, int param_pos_y) {
        super(Configuration_Module.Typ_Tube, Name, param_pos_x, param_pos_y);
        this.Radius = 1;
        this.Parts = 10;
        this.Option = 1;
//        this.LimitRadius = false;
//        this.max_Radius = 1;

    }

    @Override
    public String[] addtoscript() {
        String[] ExportStringLines = new String[12];

        ExportStringLines[0] = "#";
        ExportStringLines[1] = "# MODULE: Tube";
        ExportStringLines[2] = "#";
        ExportStringLines[3] = this.Name + "=Tube()";
        ExportStringLines[4] = "network.add(" + this.Name + ")";
        ExportStringLines[5] = this.Name + ".setPos(" + Integer.toString(this.param_pos_x) + "," + Integer.
                toString(this.param_pos_y) + ")";
        ExportStringLines[6] = "#";
        ExportStringLines[7] = "# set parameter values";
        ExportStringLines[8] = "#";
        ExportStringLines[9] = this.Name + ".set_Radius(" + this.Radius + ")";
        ExportStringLines[10] = this.Name + ".set_Parts(" + this.Parts + ")";
        ExportStringLines[11] = this.Name + ".set_Option(" + this.Option + ")";
//        ExportStringLines[12] = this.Name + ".set_LimitRadius(\"" + this.LimitRadius + "\")";
//        ExportStringLines[13] = this.Name + ".set_max_Radius(" + this.max_Radius + ")";

        return ExportStringLines;
    }

    public float getRadius() {
        return Radius;
    }

    public void setRadius(float Radius) {
        this.Radius = Radius;
    }
}
