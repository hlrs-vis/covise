package de.hlrs.starplugin.python.scriptcreation.modules;

import de.hlrs.starplugin.configuration.Configuration_Module;
import de.hlrs.starplugin.configuration.Configuration_Tool;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Module_Material extends Module {

    private String Material;    //Material_1.set_Material( Standard )
//    private String varName;     //Material_1.set_varName( "" )
//    private String attribute;   //Material_1.set_attribute( "" )
//    private Vec minBound;       //Material_1.set_minBound( 0, 0, 0 )
//    private Vec maxBound;       //Material_1.set_maxBound( 0, 0, 0 )

    public Module_Material(String Name, int param_pos_x, int param_pos_y) {
        super(Configuration_Module.Typ_Material, Name, param_pos_x, param_pos_y);
        this.Material = "Standard";
//        this.varName = "";
//        this.attribute = "";
//        this.minBound = new Vec(0, 0, 0);
//        this.maxBound = new Vec(0, 0, 0);


    }

    @Override
    public String[] addtoscript() {

        String[] ExportStringLines = new String[10];
        ExportStringLines[0] = "#";
        ExportStringLines[1] = "# MODULE: Material";
        ExportStringLines[2] = "#";
        ExportStringLines[3] = this.Name + "=Material()";
        ExportStringLines[4] = "network.add(" + this.Name + ")";
        ExportStringLines[5] = this.Name + ".setPos(" + Integer.toString(this.param_pos_x) + "," + Integer.
                toString(this.param_pos_y) + ")";

        ExportStringLines[6] = "#";
        ExportStringLines[7] = "# set parameter values";
        ExportStringLines[8] = "#";
        ExportStringLines[9] = this.Name + ".set_Material(\"" + this.Material + "\" )";
//        ExportStringLines[10] = this.Name + ".set_varName(\"" + this.varName + "\" )";
//        ExportStringLines[11] = this.Name + ".set_attribute(\"" + this.attribute + "\" )";

//        ExportStringLines[12] = this.Name + ".set_minBound(" + this.minBound.x + "," + this.minBound.y + "," + this.minBound.z + ")";
//        ExportStringLines[13] = this.Name + ".set_maxBound(" + this.maxBound.x + "," + this.maxBound.y + "," + this.maxBound.z + ")";
        return ExportStringLines;
    }

    public void setMaterial(String Color, float Transparency) {
        String newMaterial = "";
        if (Color.equals(Configuration_Tool.Color_grey)) {
            newMaterial = newMaterial.concat(
                    "ModuleDefined 0.140541 0.140541 0.140541 0.610811 0.610811 0.610811 0.767568 0.756757 0.764964 0 0 0 0.897297 ");
            newMaterial = newMaterial.concat(String.valueOf(Transparency));
        }
        if (Color.equals(Configuration_Tool.Color_green)) {
            newMaterial = newMaterial.concat(
                    "ModuleDefined 0.092753 0.161054 0.024138 0.371013 0.644214 0.096552 0.087474 0.207208 0.576531 0 0 0 0.892857 ");
            newMaterial = newMaterial.concat(String.valueOf(Transparency));
        }
        if (Color.equals(Configuration_Tool.Color_red)) {
            newMaterial = newMaterial.concat(
                    "ModuleDefined 0.135135 0 0 0.540541 0 0 0.592903 0 0.129656 0 0 0 0.064865 ");
            newMaterial = newMaterial.concat(String.valueOf(Transparency));
        }
        this.Material = newMaterial;
    }
}


