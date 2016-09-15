package de.hlrs.starplugin.gui.listener;

import Main.PluginContainer;
import de.hlrs.starplugin.configuration.Configuration_Tool;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class MouseListener_Click_StatusBar_Ensight implements MouseListener {

    private PluginContainer PC;

    public MouseListener_Click_StatusBar_Ensight(PluginContainer PC) {
        this.PC = PC;
    }

    public void mouseClicked(MouseEvent e) {
        this.PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanel_CreatedConstructsList().
                getJScrollPane_JTree_CreatedVisualizationConstructs().getJTree_createdConstructsList().
                setSelectionPath(null);
        this.PC.getGUI().getJPanelMainContent().getMainFrame().setStatus(
                Configuration_Tool.STATUS_ENSIGHTEXPORT);

    }

    public void mousePressed(MouseEvent e) {
    }

    public void mouseReleased(MouseEvent e) {
    }

    public void mouseEntered(MouseEvent e) {
    }

    public void mouseExited(MouseEvent e) {
    }
}
