package de.hlrs.starplugin.configuration;

/**
 *Contains Configuration for the ModuleManager to place the Modules
 * in the Covise Map_Editor
 * @author Weiss HLRS Stuttgart
 */
public class Configuration_ModuleManager {

    public static final int YDistance = 60;
    public static final int ReadEnsight_XPosition = 350;
    public static final int XPosition_Distance = 200;
    public static final int YPosition_ReadEnsight = 0;
    public static final int XPosition_MinMax = -600;
    public static final int YPosition_MinMax = 100;
    public static final int YPosition_GetSubset = YPosition_ReadEnsight + YDistance;
    public static final int YPosition_Material = YPosition_GetSubset + YDistance;
    public static final int YPosition_Tracer = YPosition_GetSubset + YDistance;
    public static final int YPosition_SimplifySurface = YPosition_GetSubset + YDistance;
    public static final int YPosition_Tube = YPosition_Tracer + YDistance;
    public static final int YPosition_CuttingSurface = YPosition_GetSubset + YDistance;
    public static final int YPosition_IsoSurface = YPosition_CuttingSurface;
    public static final int YPosition_Colors = YPosition_Tube + YDistance;
    public static final int YPosition_Collect = YPosition_Colors + YDistance;
    public static final int YPosition_Renderer = YPosition_Collect + YDistance * 2;
    public static final int D_3D = 0;
    public static final int D_2D = 1;
}
