package de.hlrs.starplugin.gui.covise_net_generation.listener;

import Main.PluginContainer;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreePath;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class ActionListener_Button_Delete implements ActionListener {

    private PluginContainer PC;

    public ActionListener_Button_Delete(PluginContainer PC) {
        super();
        this.PC = PC;
    }

    public void actionPerformed(ActionEvent e) {
        TreePath a = PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().
                getjPanel_CreatedConstructsList().
                getJScrollPane_JTree_CreatedVisualizationConstructs().getJTree_createdConstructsList().
                getSelectionPath();
        if (a != null) {
            PC.getCNGDMan().getConMan().deleteConstruct(((String) ((DefaultMutableTreeNode) a.getLastPathComponent()).
                    getUserObject()));
        }
        if (((DefaultMutableTreeNode) PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().
                getjPanel_CreatedConstructsList().
                getJScrollPane_JTree_CreatedVisualizationConstructs().getJTree_createdConstructsList().
                getModel().getRoot()).getChildCount() > 0) {
            PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().
                    getjPanel_CreatedConstructsList().
                    getJScrollPane_JTree_CreatedVisualizationConstructs().getJTree_createdConstructsList().
                    setSelectionRow(0);
        }
    }
}
