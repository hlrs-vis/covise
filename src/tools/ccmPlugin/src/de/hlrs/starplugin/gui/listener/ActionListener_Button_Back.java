package de.hlrs.starplugin.gui.listener;

import Main.PluginContainer;
import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.gui.JPanel_MainContent;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

/**
 *
 *@author Weiss HLRS Stuttgart
 */
public class ActionListener_Button_Back implements ActionListener {

    private PluginContainer PC;
    private JPanel_MainContent Container;

    public ActionListener_Button_Back(PluginContainer PC) {
        super();
        this.PC = PC;
        Container = PC.getGUI().getJPanelMainContent();

    }

    public void actionPerformed(ActionEvent e) {
        Container.getJPanelCoviseNetGeneration().getjPanel_CreatedConstructsList().
                getJScrollPane_JTree_CreatedVisualizationConstructs().getJTree_createdConstructsList().
                setSelectionPath(null);

        Container.getMainFrame().changeStatus(Configuration_Tool.STATUS_ENSIGHTEXPORT);

    }
}
