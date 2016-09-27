package de.hlrs.starplugin.covise_net_generation.constructs;

import Main.PluginContainer;
import de.hlrs.starplugin.configuration.Configuration_GUI_Strings;
import de.hlrs.starplugin.configuration.Configuration_Tool;

/**
 * Creates Constructs
 * @author Weiss HLRS Stuttgart
 */
public class Construct_Creator {

    public static Construct_GeometryVisualization createGeometryConstruct(PluginContainer PC) {
        Construct_GeometryVisualization Con = new Construct_GeometryVisualization();
        Con.setTyp(Configuration_Tool.VisualizationType_Geometry);
        Con.setName(Configuration_Tool.VisualizationType_Geometry + Configuration_GUI_Strings.Underscore + PC.getCNGDMan().
                getConMan().getGeometryCount());
        return Con;

    }

    public static Construct_CuttingSurface createCuttingSurfaceConstruct(PluginContainer PC) {
        Construct_CuttingSurface Con = new Construct_CuttingSurface();
        Con.setTyp(Configuration_Tool.VisualizationType_CuttingSurface);
        Con.setName(Configuration_Tool.VisualizationType_CuttingSurface + Configuration_GUI_Strings.Underscore + PC.
                getCNGDMan().getConMan().
                getCuttingSurfaceCount());
        return Con;
    }

    public static Construct_CuttingSurfaceSeries createCuttingSurfaceSeriesConstruct(PluginContainer PC) {
        Construct_CuttingSurfaceSeries Con = new Construct_CuttingSurfaceSeries();
        Con.setTyp(Configuration_Tool.VisualizationType_CuttingSurface_Series);
        Con.setName(Configuration_Tool.VisualizationType_CuttingSurface_Series + Configuration_GUI_Strings.Underscore + PC.
                getCNGDMan().getConMan().
                getCuttingSurfaceSeriesCount());
        return Con;
    }

    public static Construct_Streamline createStreamlineConstruct(PluginContainer PC) {
        Construct_Streamline Con = new Construct_Streamline();
        Con.setTyp(Configuration_Tool.VisualizationType_Streamline);
        Con.setName(Configuration_Tool.VisualizationType_Streamline + Configuration_GUI_Strings.Underscore + PC.
                getCNGDMan().getConMan().
                getStreamlineCount());

        return Con;
    }

    public static Construct_IsoSurface createIsoSurfaceConstruct(PluginContainer PC) {
        Construct_IsoSurface Con = new Construct_IsoSurface();
        Con.setTyp(Configuration_Tool.VisualizationType_IsoSurface);
        Con.setName(Configuration_Tool.VisualizationType_IsoSurface + Configuration_GUI_Strings.Underscore + PC.
                getCNGDMan().getConMan().
                getIsoSurfaceCount());

        return Con;
    }
}
