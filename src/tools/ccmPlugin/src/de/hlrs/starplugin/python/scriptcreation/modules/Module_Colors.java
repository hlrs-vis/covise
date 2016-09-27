package de.hlrs.starplugin.python.scriptcreation.modules;

import de.hlrs.starplugin.configuration.Configuration_Module;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Module_Colors extends Module {

    private float min;
    private float max;
    private String EditableColormap;
    private int Colormap;
    private int numSteps;
    private boolean autoScales;
    private boolean scaleNow;
    private float opacityFactor;
    private String annotation;
//    private int SpikeAlgo;
//    private float SpikeLowFrac;
//    private float SpikeTopFract;

    public Module_Colors(String Name, int param_pos_x, int param_pos_y) {
        super(Configuration_Module.Typ_Colors, Name, param_pos_x, param_pos_y);
        this.min = 0;
        this.max = 0;
        this.EditableColormap = "0 0 RGBAX 21"
                + " 0 0 1 1 0"
                + " 0 0.2 1 1 0.05"
                + " 0 0.4 1 1 0.1"
                + " 0 0.6 1 1 0.15"
                + " 0 0.8 1 1 0.2"
                + " 0 0.917647 0.917647 1 0.25"
                + " 0 1 0.8 1 0.3"
                + " 0 1 0.615686 1 0.35"
                + " 0 1 0.4 1 0.4"
                + " 0 1 0.2 1 0.45"
                + " 0 1 0 1 0.5"
                + " 0.184314 1 0 1 0.55"
                + " 0.4 1 0 1 0.6"
                + " 0.6 1 0 1 0.65"
                + " 0.8 1 0 1 0.7"
                + " 0.913725 0.913725 0 1 0.75"
                + " 1 0.8 0 1 0.8"
                + " 1 0.6 0 1 0.85"
                + " 1 0.4 0 1 0.9"
                + " 1 0.2 0 1 0.95"
                + " 1 0 0 1 1 ";
        this.Colormap = 11;
        this.numSteps = 256;
        this.autoScales = true;
        this.scaleNow = true;
        this.opacityFactor = 1;
        this.annotation = "Colors";
//        this.SpikeAlgo = 1;
//        this.SpikeLowFrac = 0.05f;
//        this.SpikeTopFract = 0.05f;

    }

    @Override
    public String[] addtoscript() {
        String[] ExportStringLines = new String[17];
        ExportStringLines[0] = "#";
        ExportStringLines[1] = "# MODULE: Colors";
        ExportStringLines[2] = "#";
        ExportStringLines[3] = this.Name + "=Colors()";
        ExportStringLines[4] = "network.add(" + this.Name + ")";
        ExportStringLines[5] = this.Name + ".setPos(" + Integer.toString(this.param_pos_x) + "," + Integer.
                toString(this.param_pos_y) + ")";
        ExportStringLines[6] = "#";
        ExportStringLines[7] = "# set parameter values";
        ExportStringLines[8] = "#";
        ExportStringLines[9] = this.Name + ".set_MinMax(" + this.min + "," + this.max + ")";

        ExportStringLines[10] = "#" + this.Name + ".set_EditableColormap(\"" + this.EditableColormap + "\")";
        ExportStringLines[11] = this.Name + ".set_Colormap(" + this.Colormap + ")";
        ExportStringLines[12] = this.Name + ".set_numSteps(" + this.numSteps + ")";
        ExportStringLines[13] = this.Name + ".set_autoScales(\"" + this.autoScales + "\")";
        ExportStringLines[14] = this.Name + ".set_scaleNow(\"" + this.scaleNow + "\")";
        ExportStringLines[15] = this.Name + ".set_opacityFactor(" + this.opacityFactor + ")";
        ExportStringLines[16] = this.Name + ".set_annotation(\"" + this.annotation + "\")";
//        ExportStringLines[17] = this.Name + ".set_SpikeAlgo(" + this.SpikeAlgo + ")";
//        ExportStringLines[18] = this.Name + ".set_SpikeLowFract(" + this.SpikeLowFrac + ")";
//        ExportStringLines[19] = this.Name + ".set_SpikeTopFract(" + this.SpikeTopFract + ")";
        return ExportStringLines;
    }

    public String getAnnotation() {
        return annotation;
    }

    public void setAnnotation(String annotation) {
        this.annotation = annotation;
    }

}
