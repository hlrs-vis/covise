package de.hlrs.starplugin.python.scriptcreation.module_metadata;

import de.hlrs.starplugin.python.scriptcreation.modules.Module_ReadEnsight;
import java.util.Arrays;
import star.common.FieldFunction;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class ModuleMetaData_ReadEnsight extends ModuleMetaData {

    private Integer[] Parts;
    private FieldFunction[] ScalarFieldFunctionList = new FieldFunction[3];
    private FieldFunction[] VectorFieldFunctionList = new FieldFunction[2];
    private int ScalarFieldFunctionCount = 0;
    private int VectorFieldFunctionCount = 0;
    private Module_ReadEnsight Module_RE;

    public ModuleMetaData_ReadEnsight(Module_ReadEnsight Module_RE) {
        this.Module_RE = Module_RE;
    }

    public Integer[] getParts() {
        return Parts;
    }

    public void setParts(Integer[] Parts) {
        this.Parts = Parts;
        this.Module_RE.setChoose_parts(Arrays.toString(Parts));
    }

    public void addScalarFieldFunction(FieldFunction FF, int FF_Number, int FF_Index) {
        ScalarFieldFunctionList[FF_Number] = FF;
        ScalarFieldFunctionCount++;
        switch (FF_Number) {

            case 0:

                Module_RE.setSdata_2D_1(FF_Index);
                Module_RE.setSdata_3D_1(FF_Index);
                break;
            case 1:

                Module_RE.setSdata_2D_2(FF_Index);
                Module_RE.setSdata_3D_2(FF_Index);
                break;
            case 2:

                Module_RE.setSdata_2D_3(FF_Index);
                Module_RE.setSdata_3D_3(FF_Index);
                break;

        }



    }

    public void addVectorFieldFunction(FieldFunction FF, int FF_Number, int FF_Index) {
        VectorFieldFunctionList[FF_Number] = FF;
        VectorFieldFunctionCount++;
        switch (FF_Number) {

            case 0:

                Module_RE.setVdata_2D_1(FF_Index);
                Module_RE.setVdata_3D_1(FF_Index);
                break;
            case 1:

                Module_RE.setVdata_2D_2(FF_Index);
                Module_RE.setVdata_3D_2(FF_Index);
                break;
        }
    }

    public int getScalarFieldFunctionCount() {
        return ScalarFieldFunctionCount;
    }

    public void setScalarFieldFunctionCount(int ScalarFieldFunctionCount) {
        this.ScalarFieldFunctionCount = ScalarFieldFunctionCount;
    }

    public FieldFunction[] getScalarFieldFunctionList() {
        return ScalarFieldFunctionList;
    }

    public void setScalarFieldFunctionList(FieldFunction[] ScalarFieldFunctionList) {
        this.ScalarFieldFunctionList = ScalarFieldFunctionList;
    }

    public int getVectorFieldFunctionCount() {
        return VectorFieldFunctionCount;
    }

    public void setVectorFieldFunctionCount(int VectorFieldFunctionCount) {
        this.VectorFieldFunctionCount = VectorFieldFunctionCount;
    }

    public FieldFunction[] getVectorFieldFunctionList() {
        return VectorFieldFunctionList;
    }

    public void setVectorFieldFunctionList(FieldFunction[] VectorFieldFunctionList) {
        this.VectorFieldFunctionList = VectorFieldFunctionList;
    }

    public Module_ReadEnsight getModule_RE() {
        return Module_RE;
    }
}
