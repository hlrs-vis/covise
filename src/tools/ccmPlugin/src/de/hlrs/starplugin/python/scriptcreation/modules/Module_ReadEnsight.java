package de.hlrs.starplugin.python.scriptcreation.modules;

import de.hlrs.starplugin.configuration.Configuration_Module;
import de.hlrs.starplugin.python.scriptcreation.module_metadata.ModuleMetaData_ReadEnsight;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Module_ReadEnsight extends Module {

    private String path;
    private String case_file__filter;
    private int sdata_3D_1;
    private int sdata_3D_2;
    private int sdata_3D_3;
    private int vdata_3D_1;
    private int vdata_3D_2;
    private int sdata_2D_1;
    private int sdata_2D_2;
    private int sdata_2D_3;
    private int vdata_2D_1;
    private int vdata_2D_2;
    private int sdata_1D_1;
    private int sdata_1D_2;
    private int sdata_1D_3;
    private int vdata_1D_1;
    private int vdata_1D_2;
//    private boolean data_byte_swap;
    private String choose_parts;
//    private boolean repair_connectivity;
//    private boolean enable_autocoloring;
//    private boolean store_covgrp;
//    private boolean include_polyhedra;
    private ModuleMetaData_ReadEnsight MetaData;

    public Module_ReadEnsight(String Name, int param_pos_x, int param_pos_y, String path) {
        super(Configuration_Module.Typ_ReadEnsight, Name, param_pos_x, param_pos_y);
        this.path = path.replace("\\", "/");
        this.case_file__filter = "case_file *.case;*.CASE;*.encas ";
        this.sdata_3D_1 = 1;
        this.sdata_3D_2 = 1;
        this.sdata_3D_3 = 1;
        this.vdata_3D_1 = 1;
        this.vdata_3D_2 = 1;
        this.sdata_2D_1 = 1;
        this.sdata_2D_2 = 1;
        this.sdata_2D_3 = 1;
        this.vdata_2D_1 = 1;
        this.vdata_2D_2 = 1;
        this.sdata_1D_1 = 1;
        this.sdata_1D_2 = 1;
        this.sdata_1D_3 = 1;
        this.vdata_1D_1 = 1;
        this.vdata_1D_2 = 1;
//        this.data_byte_swap = true;
        this.choose_parts = "all";
//        this.repair_connectivity = false;
//        this.enable_autocoloring = true;
//        this.store_covgrp = false;
//        this.include_polyhedra = true;
        this.MetaData = new ModuleMetaData_ReadEnsight(this);

    }


    @Override
    public String[] addtoscript() {
        String[] ExportStringLines = new String[27];
        ExportStringLines[0] = "#";
        ExportStringLines[1] = "# MODULE: ReadEnsight";
        ExportStringLines[2] = "#";
        ExportStringLines[3] = this.Name + "=ReadEnsight()";
        ExportStringLines[4] = "network.add(" + this.Name + ")";
        ExportStringLines[5] = this.Name + ".setPos(" + Integer.toString(this.param_pos_x) + "," + Integer.
                toString(this.param_pos_y) + ")";

        ExportStringLines[6] = "#";
        ExportStringLines[7] = "# set parameter values";
        ExportStringLines[8] = "#";
        ExportStringLines[9] = this.Name + ".set_case_file(\"" + this.path + "\")";
        ExportStringLines[10] = this.Name + ".set_case_file___filter(\"" + this.case_file__filter + "\")";
        ExportStringLines[11] = this.Name + ".set_data_for_sdata1_3D(" + this.sdata_3D_1 + ")";
        ExportStringLines[12] = this.Name + ".set_data_for_sdata2_3D(" + this.sdata_3D_2 + ")";
        ExportStringLines[13] = this.Name + ".set_data_for_sdata3_3D(" + this.sdata_3D_3 + ")";
        ExportStringLines[14] = this.Name + ".set_data_for_vdata1_3D(" + this.vdata_3D_1 + ")";
        ExportStringLines[15] = this.Name + ".set_data_for_vdata2_3D(" + this.vdata_3D_2 + ")";
        ExportStringLines[16] = this.Name + ".set_data_for_sdata1_2D(" + this.sdata_2D_1 + ")";
        ExportStringLines[17] = this.Name + ".set_data_for_sdata2_2D(" + this.sdata_2D_2 + ")";
        ExportStringLines[18] = this.Name + ".set_data_for_sdata3_2D(" + this.sdata_2D_3 + ")";
        ExportStringLines[19] = this.Name + ".set_data_for_vdata1_2D(" + this.vdata_2D_1 + ")";
        ExportStringLines[20] = this.Name + ".set_data_for_vdata2_2D(" + this.vdata_2D_2 + ")";
        ExportStringLines[21] = this.Name + ".set_data_for_sdata1_1D(" + this.sdata_1D_1 + ")";
        ExportStringLines[22] = this.Name + ".set_data_for_sdata2_1D(" + this.sdata_1D_2 + ")";
        ExportStringLines[23] = this.Name + ".set_data_for_sdata3_1D(" + this.sdata_1D_3 + ")";
        ExportStringLines[24] = this.Name + ".set_data_for_vdata1_1D(" + this.vdata_1D_1 + ")";
        ExportStringLines[25] = this.Name + ".set_data_for_vdata2_1D(" + this.vdata_1D_2 + ")";
        ExportStringLines[26] = this.Name + ".set_choose_parts(\"" + this.choose_parts + "\")";
//        ExportStringLines[27] = this.Name + ".set_data_byte_swap(\"" + this.data_byte_swap + "\")";
        
//        ExportStringLines[28] = this.Name + ".set_repair_connectivity(\"" + this.repair_connectivity + "\")";
//        ExportStringLines[29] = this.Name + ".set_enable_autocoloring(\"" + this.enable_autocoloring + "\")";
//        ExportStringLines[30] = this.Name + ".set_store_covgrp(\"" + this.store_covgrp + "\")";
//        ExportStringLines[31] = this.Name + ".set_include_polyhedra(\"" + this.include_polyhedra + "\")";
        return ExportStringLines;
    }

    public void setChoose_parts(String choose_parts) {
        this.choose_parts = choose_parts;
    }

    public ModuleMetaData_ReadEnsight getMetaData() {
        return MetaData;
    }

    public void setMetaData(ModuleMetaData_ReadEnsight MetaData) {
        this.MetaData = MetaData;
    }

    public void setSdata_2D_1(int sdata_2D_1) {
        this.sdata_2D_1 = sdata_2D_1;
    }

    public void setSdata_2D_2(int sdata_2D_2) {
        this.sdata_2D_2 = sdata_2D_2;
    }

    public void setSdata_2D_3(int sdata_2D_3) {
        this.sdata_2D_3 = sdata_2D_3;
    }

    public void setSdata_3D_1(int sdata_3D_1) {
        this.sdata_3D_1 = sdata_3D_1;
    }

    public void setSdata_3D_2(int sdata_3D_2) {
        this.sdata_3D_2 = sdata_3D_2;
    }

    public void setSdata_3D_3(int sdata_3D_3) {
        this.sdata_3D_3 = sdata_3D_3;
    }


    public void setVdata_2D_1(int vdata_2D_1) {
        this.vdata_2D_1 = vdata_2D_1;
    }

    public void setVdata_2D_2(int vdata_2D_2) {
        this.vdata_2D_2 = vdata_2D_2;
    }

    public void setVdata_3D_1(int vdata_3D_1) {
        this.vdata_3D_1 = vdata_3D_1;
    }

    public void setVdata_3D_2(int vdata_3D_2) {
        this.vdata_3D_2 = vdata_3D_2;
    }
    
}
