package de.hlrs.starplugin.python.scriptcreation.modules;

import de.hlrs.starplugin.configuration.Configuration_Module;
import de.hlrs.starplugin.util.Vec;

/**
 *
 * @author hpcweiss
 */
public class Module_SimplifySurface extends Module {

    private int method;                         //SimplifySurface_1.set_method( 2 )
//    private float percent;                      //SimplifySurface_1.set_percent( 30.000000 )
//    private float max_normaldeviation;         //SimplifySurface_1.set_max_normaldeviation( 3.000000 )
//    private Vec max_domaindeviation;           //SimplifySurface_1.set_max_domaindeviation( 0.3, 4, 2 )
//    private float data_relative_weight;         //SimplifySurface_1.set_data_relative_weight( 0.050000 )
//    private boolean ignore_data;                //SimplifySurface_1.set_ignore_data( "FALSE" )
    private Vec divisions;                      //SimplifySurface_1.set_divisions( 5, 5, 5 )
    private boolean divisions_are_absolute;     //SimplifySurface_1.set_divisons_are_absolute( "FALSE" )
//    private boolean smooth_surface;             //SimplifySurface_1.set_smooth_surface( "FALSE" )
//    private boolean preserve_topology;          //SimplifySurface_1.set_preserve_topology( "TRUE" )
//    private boolean mesh_splitting;             //SimplifySurface_1.set_mesh_splitting( "TRUE" )
//    private float split_angle;                  //SimplifySurface_1.set_split_angle( 75.000000 )
//    private float feature_angle;                //SimplifySurface_1.set_feature_angle( 15.000000 )
//    private boolean boundary_vertex_deletion;   //SimplifySurface_1.set_boundary_vertex_deletion( "TRUE" )
//    private float maximum_error;                //SimplifySurface_1.set_maximum_error( 0.030000 )

    public Module_SimplifySurface(String Name, int param_pos_x, int param_pos_y) {
        super(Configuration_Module.Typ_SimplifySurface, Name, param_pos_x, param_pos_y);
        this.method = 2;
//        this.percent = 30;
//        this.max_normaldeviation = 3;
//        this.max_domaindeviation = new Vec(0.3f, 4, 2);
//        this.data_relative_weight = 0.05f;
//        this.ignore_data = false;
        this.divisions = new Vec(5, 5, 5);
        this.divisions_are_absolute = false;
//        this.smooth_surface = false;
//        this.preserve_topology = true;
//        this.mesh_splitting = true;
//        this.split_angle = 75f;
//        this.feature_angle = 15f;
//        this.boundary_vertex_deletion = true;
//        this.maximum_error = 0.03f;
    }

    @Override
    public String[] addtoscript() {
        String[] ExportStringLines = new String[12];

        ExportStringLines[0] = "#";
        ExportStringLines[1] = "# MODULE: SimplifySurface";
        ExportStringLines[2] = "#";
        ExportStringLines[3] = this.Name + "=SimplifySurface()";
        ExportStringLines[4] = "network.add(" + this.Name + ")";
        ExportStringLines[5] = this.Name + ".setPos(" + Integer.toString(this.param_pos_x) + "," + Integer.
                toString(this.param_pos_y) + ")";
        ExportStringLines[6] = "#";
        ExportStringLines[7] = "# set parameter values";
        ExportStringLines[8] = "#";
        ExportStringLines[9] = this.Name + ".set_method(" + this.method + ")";
        ExportStringLines[10] = this.Name + ".set_divisions(" + this.divisions.x + "," + this.divisions.y + "," + this.divisions.z + ")";
        ExportStringLines[11] = this.Name + ".set_divisons_are_absolute(\"" + this.divisions_are_absolute + "\")";
//        ExportStringLines[12] = this.Name + ".set_percent(" + this.percent + ")";
//        ExportStringLines[13] = this.Name + ".set_max_normaldeviation(" + this.max_normaldeviation + ")";
//        ExportStringLines[14] = this.Name + ".set_max_domaindeviation(" + this.max_domaindeviation.x + "," + this.max_domaindeviation.y + "," + this.max_domaindeviation.z + ")";
//        ExportStringLines[15] = this.Name + ".set_data_relative_weight(" + this.data_relative_weight + ")";
//        ExportStringLines[16] = this.Name + ".set_ignore_data(\"" + this.ignore_data + "\")";

//        ExportStringLines[17] = this.Name + ".set_smooth_surface(\"" + this.smooth_surface + "\")";
//        ExportStringLines[18] = this.Name + ".set_preserve_topology(\"" + this.preserve_topology + "\")";
//        ExportStringLines[19] = this.Name + ".set_mesh_splitting(\"" + this.mesh_splitting + "\")";
//        ExportStringLines[20] = this.Name + ".set_split_angle(" + this.split_angle + ")";
//        ExportStringLines[21] = this.Name + ".set_feature_angle(" + this.feature_angle + ")";
//        ExportStringLines[22] = this.Name + ".set_boundary_vertex_deletion(\"" + this.boundary_vertex_deletion + "\")";
//        ExportStringLines[23] = this.Name + ".set_maximum_error(" + this.maximum_error + ")";


        return ExportStringLines;
    }

    public Vec getDivisions() {
        return divisions;
    }

    public void setDivisions(Vec divisions) {
        this.divisions = divisions;
    }
}
