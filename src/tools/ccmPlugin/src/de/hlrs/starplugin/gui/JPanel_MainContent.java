/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package de.hlrs.starplugin.gui;

import de.hlrs.starplugin.interfaces.Interface_MainFrame_StatusChangedListener;
import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.gui.covise_net_generation.JPanel_CoviseNetGeneration;
import de.hlrs.starplugin.gui.ensight_export.JPanel_EnsightExport;
import java.awt.CardLayout;
import javax.swing.JPanel;

/**
 *
 *  @author Weiss HLRS Stuttgart
 */
public class JPanel_MainContent extends JPanel implements Interface_MainFrame_StatusChangedListener {

    private final CardLayout layout;
    private JPanel_EnsightExport ensightexportjpanel;
    private JPanel_CoviseNetGeneration jPanelCoviseNetGenration;
    private JFrame_MainFrame Mainframe;

    public JPanel_MainContent(JFrame_MainFrame Main) {
        this.Mainframe = Main;
        layout = new CardLayout();
        this.setLayout(layout);
        //CardLayout einstellung (Cards: Ensight ExportPanel, COVIES NET GEN Panel)
        //Ensight Export Panel Einstellung
        ensightexportjpanel = new JPanel_EnsightExport();
        this.add(ensightexportjpanel, Configuration_Tool.CardLayoutKey_EnsightExportCard);
        //COVISE NET GEN Panel
        jPanelCoviseNetGenration = new JPanel_CoviseNetGeneration();
        this.add(jPanelCoviseNetGenration, Configuration_Tool.CardLayoutKey_CoviseNetGenerationCard);
    }

    public JPanel_EnsightExport getEnsightExportJpanel() {
        return ensightexportjpanel;
    }

    public JFrame_MainFrame getMainFrame() {
        return Mainframe;
    }

    public JPanel_CoviseNetGeneration getJPanelCoviseNetGeneration() {
        return jPanelCoviseNetGenration;
    }


    @Override
    public CardLayout getLayout() {
        return layout;
    }

    public void onChange(int Status) {
        if (Configuration_Tool.STATUS_ENSIGHTEXPORT.equals(Status)) {
            layout.show(this, Configuration_Tool.CardLayoutKey_EnsightExportCard);
        }
        if (Configuration_Tool.STATUS_NETGENERATION.equals(Status)) {
            layout.show(this, Configuration_Tool.CardLayoutKey_CoviseNetGenerationCard);
        }
    }

}
