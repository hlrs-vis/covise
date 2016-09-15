package de.hlrs.starplugin.gui.ensight_export;

import de.hlrs.starplugin.gui.ensight_export.tabbed_pane.JPanel_Boundaries;
import de.hlrs.starplugin.gui.ensight_export.tabbed_pane.JPanel_Regions;
import de.hlrs.starplugin.gui.ensight_export.tabbed_pane.JPanel_Scalars;
import de.hlrs.starplugin.gui.ensight_export.tabbed_pane.JPanel_Vectors;
import java.awt.Dimension;
import javax.swing.JTabbedPane;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class JTabbedPane_EnsightExport extends JTabbedPane {

    private JPanel_Regions JPanelRegions;
    private JPanel_Boundaries JPanelBoundaries;
    private JPanel_Scalars JPanelScalars;
    private JPanel_Vectors JPanelVectors;


    public JTabbedPane_EnsightExport() {
        super();
        this.setMinimumSize(new Dimension(400, 400));
        this.setPreferredSize(new Dimension(400, 400));

        //Panel Regions
        JPanelRegions = new JPanel_Regions();
        this.add("Regions", JPanelRegions);

        //Panel Boundaries
        JPanelBoundaries = new JPanel_Boundaries();
        this.add("Boundaries", JPanelBoundaries);


        //Panel Scalar Funcitons
        JPanelScalars = new JPanel_Scalars();
        this.add("Scalars", JPanelScalars);


        //Panel Vector Funcitons
        JPanelVectors = new JPanel_Vectors();
        this.add("Vectors", JPanelVectors);

    }

    public JPanel_Regions getJPanelRegions() {
        return JPanelRegions;
    }

    public JPanel_Boundaries getJPanelBoundaries() {
        return JPanelBoundaries;
    }

    public JPanel_Scalars getJPanelScalars() {
        return JPanelScalars;
    }

    public JPanel_Vectors getJPanelVectors() {
        return JPanelVectors;
    }
    
}
