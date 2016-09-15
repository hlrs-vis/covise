package de.hlrs.starplugin.python.scriptcreation.modules;

import de.hlrs.starplugin.configuration.Configuration_Module;
import de.hlrs.starplugin.util.Vec;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Module_IsoSurface extends Module {

//    private boolean gennormals;
//    private boolean genstrips;
//    private int Interactor;
//    private Vec IsoPoint;
    private Vec IsoValue = new Vec(0, 0.5f, 1);
    private boolean autominmax;

    public Module_IsoSurface(String Name, int param_pos_x, int param_pos_y) {
        super(Configuration_Module.Typ_IsoSurface, Name, param_pos_x, param_pos_y);
//        this.gennormals = true;
//        this.genstrips = true;
//        this.Interactor = 2;
//        this.IsoPoint = new Vec(0, 0, 0);
        this.IsoValue = new Vec(0, 0.5f, 1);
        this.autominmax = true;

    }

    @Override
    public String[] addtoscript() {
        String[] ExportStringLines = new String[11];
        ExportStringLines[0] = "#";
        ExportStringLines[1] = "# MODULE: IsoSurface";
        ExportStringLines[2] = "#";
        ExportStringLines[3] = this.Name + "=IsoSurface()";
        ExportStringLines[4] = "network.add(" + this.Name + ")";
        ExportStringLines[5] = this.Name + ".setPos(" + Integer.toString(this.param_pos_x) + "," + Integer.
                toString(this.param_pos_y) + ")";

        ExportStringLines[6] = "#";
        ExportStringLines[7] = "# set parameter values";
        ExportStringLines[8] = "#";
        ExportStringLines[9] = this.Name + ".set_autominmax(\"" + this.autominmax + "\")";
        ExportStringLines[10] = this.Name + ".set_isovalue(" + this.IsoValue.x + "," + this.IsoValue.y + "," + this.IsoValue.z + ")";
//        ExportStringLines[11] = this.Name + ".set_gennormals(\"" + this.gennormals + "\" )";
//        ExportStringLines[12] = this.Name + ".set_genstrips(\"" + this.genstrips + "\")";
//        ExportStringLines[13] = this.Name + ".set_Interactor(" + this.Interactor + ")";
//        ExportStringLines[14] = this.Name + ".set_isopoint(" + this.IsoPoint.x + "," + this.IsoPoint.y + "," + this.IsoPoint.z + ")";


        return ExportStringLines;
    }

    public Vec getIsoValue() {
        return IsoValue;
    }

    public void setIsoValue(Vec IsoValue) {
        this.IsoValue = IsoValue;
    }
}
