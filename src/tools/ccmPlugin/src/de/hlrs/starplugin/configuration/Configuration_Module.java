/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package de.hlrs.starplugin.configuration;

/**
 *Contais Configuration Data regarding the Modules
 * (Connection to Covise)
 * @author Weiss HLRS Stuttgart
 */
public final class Configuration_Module {

    public static final String Typ_DomainSurface = "DomainSurface";
    public static final String Typ_ReadEnsight = "ReadEnsight";
    public static final String Typ_CutGeometry = "CutGeometry";
    public static final String Typ_IvRenderer = "IvRenderer";
    public static final String Typ_Collect = "Collect";
    public static final String Typ_Colors = "Colors";
    public static final String Typ_CuttingSurface = "CuttingSurface";
    public static final String Typ_GetSubset = "GetSubset";
    public static final String Typ_Tracer = "Tracer";
    public static final String Typ_Tube = "Tube";
    public static final String Typ_SimplifySurface = "SimplifySurface";
    public static final String Typ_IsoSurface = "IsoSurface";
    public static final String Typ_MinMax = "MinMax";
    public static final String Typ_Material = "Material";
    public static final String Typ_OpenCover="OpenCOVER";
    public static final String Underscore = "_";
    //Portnames from COVISE
    public static final String[] OutputPorts_ReadEnsight = new String[12];

    static {

        OutputPorts_ReadEnsight[0] = "geoOut_3D";
        OutputPorts_ReadEnsight[1] = "sdata1_3D";
        OutputPorts_ReadEnsight[2] = "sdata2_3D";
        OutputPorts_ReadEnsight[3] = "sdata3_3D";
        OutputPorts_ReadEnsight[4] = "vdata1_3D";
        OutputPorts_ReadEnsight[5] = "vdata2_3D";
        OutputPorts_ReadEnsight[6] = "geoOut_2D";
        OutputPorts_ReadEnsight[7] = "sdata1_2D";
        OutputPorts_ReadEnsight[8] = "sdata2_2D";
        OutputPorts_ReadEnsight[9] = "sdata3_2D";
        OutputPorts_ReadEnsight[10] = "vdata1_2D";
        OutputPorts_ReadEnsight[11] = "vdata2_2D";
    }
    public final static String[] InputPorts_CutGeometry = new String[6];

    static {
        InputPorts_CutGeometry[0] = "geo_in";
        InputPorts_CutGeometry[1] = "data_in1";
        InputPorts_CutGeometry[2] = "data_in2";
        InputPorts_CutGeometry[3] = "data_in3";
        InputPorts_CutGeometry[4] = "data_in4";
        InputPorts_CutGeometry[5] = "adjustParams";
    }
    public final static String[] OutputPorts_CutGeometry = new String[5];

    static {
        OutputPorts_CutGeometry[0] = "geo_out";
        OutputPorts_CutGeometry[1] = "data_out1";
        OutputPorts_CutGeometry[2] = "data_out2";
        OutputPorts_CutGeometry[3] = "data_out3";
        OutputPorts_CutGeometry[4] = "data_out4";

    }
    public static final String[] InputPorts_DomainSurface = new String[2];

    static {
        InputPorts_DomainSurface[0] = "requiredMesh";
        InputPorts_DomainSurface[1] = "optionalData";
    }
    public static final String[] OutputPorts_DomainSurface = new String[4];

    static {
        OutputPorts_DomainSurface[0] = "outputSurface";
        OutputPorts_DomainSurface[1] = "dependSurfDat";
        OutputPorts_DomainSurface[2] = "outputLines";
        OutputPorts_DomainSurface[3] = "dependLinesDat";
    }
    public final static String[] InputPorts_IvRenderer = new String[1];

    static {
        InputPorts_IvRenderer[0] = "RenderData";

    }
    public final static String[] InputPorts_OpenCOVER = new String[1];

    static {
        InputPorts_OpenCOVER[0] = "RenderData";

    }
    public static final String[] InputPorts_Collect = new String[5];

    static {
        InputPorts_Collect[0] = "GridIn0";
        InputPorts_Collect[1] = "DataIn0";
        InputPorts_Collect[2] = "DataIn1";
        InputPorts_Collect[3] = "TextureIn0";
        InputPorts_Collect[4] = "VertexAttribIn0";
    }
    public static final String[] OutputPorts_Collect = new String[1];

    static {
        OutputPorts_Collect[0] = "GeometryOut0";
    }
    public static final String[] InputPorts_Colors = new String[4];

    static {
        InputPorts_Colors[0] = "DataIn0";
        InputPorts_Colors[1] = "DataIn1";
        InputPorts_Colors[2] = "DataIn2";
        InputPorts_Colors[3] = "ColormapIn0";

    }
    public static final String[] OutputPorts_Colors = new String[3];

    static {
        OutputPorts_Colors[0] = "DataOut0";
        OutputPorts_Colors[1] = "TextureOut0";
        OutputPorts_Colors[2] = "ColormapOut0";


    }
    public static final String[] InputPorts_GetSubset = new String[8];

    static {
        InputPorts_GetSubset[0] = "DataIn0";
        InputPorts_GetSubset[1] = "DataIn1";
        InputPorts_GetSubset[2] = "DataIn2";
        InputPorts_GetSubset[3] = "DataIn3";
        InputPorts_GetSubset[4] = "DataIn4";
        InputPorts_GetSubset[5] = "DataIn5";
        InputPorts_GetSubset[6] = "DataIn6";
        InputPorts_GetSubset[7] = "DataIn7";
    }
    public static final String[] OutputPorts_GetSubset = new String[8];

    static {
        OutputPorts_GetSubset[0] = "DataOut0";
        OutputPorts_GetSubset[1] = "DataOut1";
        OutputPorts_GetSubset[2] = "DataOut2";
        OutputPorts_GetSubset[3] = "DataOut3";
        OutputPorts_GetSubset[4] = "DataOut4";
        OutputPorts_GetSubset[5] = "DataOut5";
        OutputPorts_GetSubset[6] = "DataOut6";
        OutputPorts_GetSubset[7] = "DataOut7";
    }
    public static final String[] InputPorts_CuttingSurface = new String[3];

    static {
        InputPorts_CuttingSurface[0] = "GridIn0";
        InputPorts_CuttingSurface[1] = "DataIn0";
        InputPorts_CuttingSurface[2] = "DataIn3";

    }
    public static final String[] OutputPorts_CuttingSurface = new String[3];

    static {
        OutputPorts_CuttingSurface[0] = "GridOut0";
        OutputPorts_CuttingSurface[1] = "DataOut0";
        OutputPorts_CuttingSurface[2] = "DataOut1";

    }
    public static final String[] InputPorts_Tracer = new String[5];

    static {
        InputPorts_Tracer[0] = "meshIn";
        InputPorts_Tracer[1] = "dataIn";
        InputPorts_Tracer[2] = "pointsIn";
        InputPorts_Tracer[3] = "octtreesIn";
        InputPorts_Tracer[4] = "fieldIn";
    }
    public static final String[] OutputPorts_Tracer = new String[3];

    static {
        OutputPorts_Tracer[0] = "lines";
        OutputPorts_Tracer[1] = "dataOut";
        OutputPorts_Tracer[2] = "startingPoints";
    }
    public static final String[] InputPorts_Tube = new String[3];

    static {
        InputPorts_Tube[0] = "Lines";
        InputPorts_Tube[1] = "Data";
        InputPorts_Tube[2] = "Diameter";
    }
    public static final String[] OutputPorts_Tube = new String[3];

    static {
        OutputPorts_Tube[0] = "Tubes";
        OutputPorts_Tube[1] = "DataOut";
        OutputPorts_Tube[2] = "Normals";
    }
    public static final String[] InputPorts_SimplifySurface = new String[3];

    static {
        InputPorts_SimplifySurface[0] = "meshIn";
        InputPorts_SimplifySurface[1] = "dataIn_0";
        InputPorts_SimplifySurface[2] = "normalsIn";
    }
    public static final String[] OutputPorts_SimplifySurface = new String[3];

    static {
        OutputPorts_SimplifySurface[0] = "meshOut";
        OutputPorts_SimplifySurface[1] = "dataOut_0";
        OutputPorts_SimplifySurface[2] = "normalsOut";
    }
    public static final String[] InputPorts_IsoSurface = new String[4];

    static {
        InputPorts_IsoSurface[0] = "GridIn0";
        InputPorts_IsoSurface[1] = "DataIn0";
        InputPorts_IsoSurface[2] = "DataIn1";
        InputPorts_IsoSurface[3] = "DataIn2";
    }
    public static final String[] OutputPorts_IsoSurface = new String[3];

    static {
        OutputPorts_IsoSurface[0] = "GridOut0";
        OutputPorts_IsoSurface[1] = "DataOut0";
        OutputPorts_IsoSurface[2] = "DataOut1";
    }
    public static final String[] InputPorts_MinMax = new String[1];

    static {
        InputPorts_MinMax[0] = "Data";

    }
    public static final String[] OutputPorts_MinMax = new String[3];

    static {
        OutputPorts_MinMax[0] = "plot2d";
        OutputPorts_MinMax[1] = "DataOut1";
        OutputPorts_MinMax[2] = "minmax";
    }
    public static final String[] InputPorts_Material = new String[5];

    static {
        InputPorts_Material[0] = "GridIn0";
        InputPorts_Material[1] = "DataIn0";
        InputPorts_Material[2] = "DataIn1";
        InputPorts_Material[3] = "TextureIn0";
        InputPorts_Material[4] = "VertexAttribIn0";

    }
    public static final String[] OutputPorts_Material = new String[1];

    static {
        OutputPorts_Material[0] = "GeometryOut0";

    }
}
