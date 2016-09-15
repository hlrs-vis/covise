package de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator;

import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards.JPanel_BlankCard;
import de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards.JPanel_CuttingSurfaceCard;
import de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards.JPanel_CuttingSurfaceSeriesCard;
import de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards.JPanel_GeometryCard;
import de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards.JPanel_IsoSurfaceCard;
import de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards.JPanel_StreamlineCard;
import java.awt.CardLayout;
import javax.swing.JPanel;

/**
 *
 *  @author Weiss HLRS Stuttgart
 */
public class JPanel_SettingsCardLayout extends JPanel {

    private CardLayout layout;
    private JPanel_GeometryCard GeometryCard;
    private JPanel_CuttingSurfaceCard CuttingSurfaceCard;
    private JPanel_CuttingSurfaceSeriesCard CuttingSurfaceSeriesCard;
    private JPanel_StreamlineCard StreamlineCard;
    private JPanel_IsoSurfaceCard IsoSurfaceCard;
    private String Status;

    public JPanel_SettingsCardLayout() {
        layout = new CardLayout();
        setLayout(layout);


        JPanel_BlankCard BlankCard = new JPanel_BlankCard();
        this.add(Configuration_Tool.VisualizationType_ChooseType, BlankCard);

        this.GeometryCard = new JPanel_GeometryCard();
        this.add(Configuration_Tool.VisualizationType_Geometry, GeometryCard);


        this.CuttingSurfaceCard = new JPanel_CuttingSurfaceCard();
        this.add(Configuration_Tool.VisualizationType_CuttingSurface, CuttingSurfaceCard);

        this.CuttingSurfaceSeriesCard = new JPanel_CuttingSurfaceSeriesCard();
        this.add(Configuration_Tool.VisualizationType_CuttingSurface_Series, CuttingSurfaceSeriesCard);

        StreamlineCard = new JPanel_StreamlineCard();
        this.add(Configuration_Tool.VisualizationType_Streamline, StreamlineCard);

        IsoSurfaceCard = new JPanel_IsoSurfaceCard();
        this.add(Configuration_Tool.VisualizationType_IsoSurface, IsoSurfaceCard);

        this.Status = Configuration_Tool.STATUS_ChooseType;


    }

    public JPanel_GeometryCard getGeometryCard() {
        return GeometryCard;
    }

    public JPanel_CuttingSurfaceCard getCuttingSurfaceCard() {
        return CuttingSurfaceCard;
    }

    public JPanel_CuttingSurfaceSeriesCard getCuttingSurfaceSeriesCard() {
        return CuttingSurfaceSeriesCard;
    }

    public JPanel_StreamlineCard getStreamlineCard() {
        return StreamlineCard;
    }

    public JPanel_IsoSurfaceCard getIsoSurfaceCard() {
        return IsoSurfaceCard;
    }

    public void statusChanged() {
        layout.show(this, Configuration_Tool.VisualizationType_ChooseType);

        if (Status.equals(Configuration_Tool.STATUS_Geometry)) {
            layout.show(this, Configuration_Tool.VisualizationType_Geometry);
        }
        if (Status.equals(Configuration_Tool.STATUS_CuttingSurface)) {
            layout.show(this, Configuration_Tool.VisualizationType_CuttingSurface);
        }

        if (Status.equals(Configuration_Tool.STATUS_CuttingSurface_Series)) {
            layout.show(this, Configuration_Tool.VisualizationType_CuttingSurface_Series);
        }

        if (Status.equals(Configuration_Tool.STATUS_Streamline)) {
            layout.show(this, Configuration_Tool.VisualizationType_Streamline);
        }
        if (Status.equals(Configuration_Tool.STATUS_IsoSurface)) {
            layout.show(this, Configuration_Tool.VisualizationType_IsoSurface);
        }

    }

    public void setStatus(String Status) {
        this.Status = Status;
        statusChanged();
    }
}
