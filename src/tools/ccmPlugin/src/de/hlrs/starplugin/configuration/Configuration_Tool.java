package de.hlrs.starplugin.configuration;

/**
 *Contains important static Objects used from the Classes of the Plugin
 * @author Weiss HLRS Stuttgart
 */
public class Configuration_Tool {
    //Main Frame Status

    public static final Integer STATUS_ENSIGHTEXPORT = 0;
    public static final Integer STATUS_NETGENERATION = 1;
    //Main Card Layout
    public static final String CardLayoutKey_EnsightExportCard = "EnsightExport";
    public static final String CardLayoutKey_CoviseNetGenerationCard = "CoviseNetGenaration";
    //ComboBox Visualization Types
    public static final String VisualizationType_CuttingSurface = "CuttingSurface";
    public static final String VisualizationType_CuttingSurface_Series = "CuttingSurface Series";
    public static final String VisualizationType_Streamline = "Streamline";
    public static final String VisualizationType_Geometry = "Geometry";
//    public static final String VisualizationType_Geometry_Cut = "Geometry Cut";
    public static final String VisualizationType_IsoSurface = "IsoSurface";
    public static final String VisualizationType_ChooseType = "Choose Type";
    public static final String[] VisualizationTypes = {VisualizationType_ChooseType, VisualizationType_Geometry, VisualizationType_CuttingSurface,
        VisualizationType_CuttingSurface_Series, VisualizationType_Streamline, VisualizationType_IsoSurface};
    //Combobox DataTypes
    public static final String DataType_none = "None";
    public static final String DataType_scalar = "Scalar";
    public static final String DataType_vector = "Vector";
    //CoviseNetGeneration CardLayoutStatus
    public static final String STATUS_ChooseType = "ChooseType";
    public static final String STATUS_CuttingSurface = "CuttingSurface";
    public static final String STATUS_CuttingSurface_Series = "CuttingSurfaceSeries";
    public static final String STATUS_Geometry = "Geometry";
    public static final String STATUS_Streamline = "Stremaline";
    public static final String STATUS_IsoSurface = "IsoSurface";
    //Radio Buttons Action Commands
    public static final String RadioButtonActionCommand_X_Direction = "X";
    public static final String RadioButtonActionCommand_Y_Direction = "Y";
    public static final String RadioButtonActionCommand_Z_Direction = "Z";
    public static final String RadioButtonActionCommand_notKart_Direction = "notKart";
    //Velocity Function Name
    public final static String FunctionName_Velocity = "Velocity";
    //GeometryCard Material Vis ColorType
    public final static String Color_red = "red";
    public final static String Color_grey = "grey";
    public final static String Color_green = "green";
    //StreamlineCard Tracer Tdirection
    public final static String forward = "1";
    public final static String back = "2";
    public final static String both = "3";


    //Strings for on ChangeMethods
    public final static String onChange_add="add";
    public final static String onChange_delete="delete";
    public final static String onChange_Selection="Selection";
    public final static String onChange_ExportPath="ExportPath";
    public final static String onChange_AppendToFile="AppendToFile";
    public final static String onChange_ExportOnVertices="ExportOnVertices";
    public final static String onChange_Boundary="Boundary";
    public final static String onChange_Region="Region";
    public final static String onChange_Vector="Vectors";
    public final static String onChange_Scalar="Scalars";

}
