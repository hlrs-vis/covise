package de.hlrs.starplugin.gui.covise_net_generation.jpanel_created_visualization_constructs;

import java.awt.Color;
import javax.swing.BorderFactory;
import javax.swing.JScrollPane;
import javax.swing.JTree;
import javax.swing.border.EtchedBorder;
import javax.swing.tree.DefaultTreeCellRenderer;
import javax.swing.tree.TreeSelectionModel;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class JScrollPane_CreatedConstructsList extends JScrollPane {

    private JTree JTree_createdPartsList;

    public JScrollPane_CreatedConstructsList() {
        setBackground(Color.white);
        setBorder(BorderFactory.createEtchedBorder(
                EtchedBorder.RAISED));
        this.setViewportView(this.ChosenGeometryTree());
    }

    private JTree ChosenGeometryTree() {


        //Baumerzeugen
        JTree_createdPartsList = new JTree();

        //Baumeinstellungen
        JTree_createdPartsList.setShowsRootHandles(true); //
        JTree_createdPartsList.getSelectionModel().setSelectionMode(TreeSelectionModel.SINGLE_TREE_SELECTION);//Selectionmodell einstzen einfache Selection ermöglichen

        //Tree Renderer einstellen
        DefaultTreeCellRenderer renderer = (DefaultTreeCellRenderer) JTree_createdPartsList.getCellRenderer();
        renderer.setLeafIcon(null);//keine Icons
        renderer.setClosedIcon(null);
        renderer.setOpenIcon(null);
        renderer.setBackground(Color.white);//Hintergurnd
        JTree_createdPartsList.setRootVisible(false);//rootknoten ausblenden
        JTree_createdPartsList.setVisibleRowCount(JTree_createdPartsList.getRowCount()); //sichtbarkein Dynamisch


        return JTree_createdPartsList;
    }

    public JTree getJTree_createdConstructsList() {
        return JTree_createdPartsList;
    }
}
