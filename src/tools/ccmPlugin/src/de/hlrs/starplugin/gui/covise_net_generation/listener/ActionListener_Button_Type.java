package de.hlrs.starplugin.gui.covise_net_generation.listener;

import Main.PluginContainer;
import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_Creator;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_CuttingSurface;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_CuttingSurfaceSeries;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_GeometryVisualization;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_IsoSurface;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_Streamline;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.JButton;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class ActionListener_Button_Type implements ActionListener {

    private PluginContainer PC;

    public ActionListener_Button_Type(PluginContainer PC) {
        super();
        this.PC = PC;
    }

    public void actionPerformed(ActionEvent e) {
        String Selection = (String) ((JButton) e.getSource()).getActionCommand();

        if (Selection.equals(Configuration_Tool.VisualizationType_Geometry)) {
            Construct_GeometryVisualization Con = Construct_Creator.createGeometryConstruct(PC);
            PC.getCNGDMan().getConMan().addConsturct(Con);
            PC.getCNGDMan().getConMan().setSelectedConstruct(Con);
        }
        if (Selection.equals(Configuration_Tool.VisualizationType_CuttingSurface)) {
            Construct_CuttingSurface Con = Construct_Creator.createCuttingSurfaceConstruct(PC);
            PC.getCNGDMan().getConMan().addConsturct(Con);
            PC.getCNGDMan().getConMan().setSelectedConstruct(Con);
        }
        if (Selection.equals(Configuration_Tool.VisualizationType_CuttingSurface_Series)) {
            Construct_CuttingSurfaceSeries Con = Construct_Creator.createCuttingSurfaceSeriesConstruct(PC);
            PC.getCNGDMan().getConMan().addConsturct(Con);
            PC.getCNGDMan().getConMan().setSelectedConstruct(Con);
        }
        if (Selection.equals(Configuration_Tool.VisualizationType_Streamline)) {
            Construct_Streamline Con = Construct_Creator.createStreamlineConstruct(PC);
            PC.getCNGDMan().getConMan().addConsturct(Con);
            PC.getCNGDMan().getConMan().setSelectedConstruct(Con);

        }
        if (Selection.equals(Configuration_Tool.VisualizationType_IsoSurface)) {
            Construct_IsoSurface Con = Construct_Creator.createIsoSurfaceConstruct(PC);
            PC.getCNGDMan().getConMan().addConsturct(Con);
            PC.getCNGDMan().getConMan().setSelectedConstruct(Con);
        }
    }
}
