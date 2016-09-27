package de.hlrs.starplugin.gui.covise_net_generation.listener;

import Main.PluginContainer;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_Creator;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_CuttingSurface;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_CuttingSurfaceSeries;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_GeometryVisualization;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_IsoSurface;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_Streamline;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class ActionListener_Button_Clone implements ActionListener {

    private PluginContainer PC;

    public ActionListener_Button_Clone(PluginContainer pC) {
        super();
        this.PC = pC;
    }

    public void actionPerformed(ActionEvent e) {
        Construct ConSel = PC.getCNGDMan().getConMan().getSelectedConstruct();

        if (ConSel != null) {

            if (ConSel instanceof Construct_GeometryVisualization) {
                Construct_GeometryVisualization Con = Construct_Creator.createGeometryConstruct(PC);
                Con.modify((Construct_GeometryVisualization) ConSel);
                PC.getCNGDMan().getConMan().addConsturct(Con);
                PC.getCNGDMan().getConMan().setSelectedConstruct(Con);
            }
            if (ConSel instanceof Construct_CuttingSurface && !(ConSel instanceof Construct_CuttingSurfaceSeries)) {
                Construct_CuttingSurface Con = Construct_Creator.createCuttingSurfaceConstruct(PC);
                Con.modify((Construct_CuttingSurface) ConSel);
                PC.getCNGDMan().getConMan().addConsturct(Con);
                PC.getCNGDMan().getConMan().setSelectedConstruct(Con);
            }
            if (ConSel instanceof Construct_CuttingSurfaceSeries) {
                Construct_CuttingSurfaceSeries Con = Construct_Creator.createCuttingSurfaceSeriesConstruct(PC);
                Con.modify((Construct_CuttingSurfaceSeries) ConSel);
                PC.getCNGDMan().getConMan().addConsturct(Con);
                PC.getCNGDMan().getConMan().setSelectedConstruct(Con);
            }
            if (ConSel instanceof Construct_Streamline) {
                Construct_Streamline Con = Construct_Creator.createStreamlineConstruct(PC);
                Con.modify((Construct_Streamline) ConSel);
                PC.getCNGDMan().getConMan().addConsturct(Con);
                PC.getCNGDMan().getConMan().setSelectedConstruct(Con);

            }
            if (ConSel instanceof Construct_IsoSurface) {
                Construct_IsoSurface Con = Construct_Creator.createIsoSurfaceConstruct(PC);
                Con.modify((Construct_IsoSurface) ConSel);
                PC.getCNGDMan().getConMan().addConsturct(Con);
                PC.getCNGDMan().getConMan().setSelectedConstruct(Con);
            }
        }
    }
}
