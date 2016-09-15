package de.hlrs.starplugin.gui.ensight_export.tabbed_pane;

import de.hlrs.starplugin.util.GetGridBagConstraints;

import java.awt.Color;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import javax.swing.BorderFactory;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTree;
import javax.swing.border.EtchedBorder;
import javax.swing.tree.DefaultTreeCellRenderer;
import javax.swing.tree.TreeSelectionModel;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class JPanel_Boundaries extends JPanel {

    private final GridBagLayout layout;
    private JScrollPane RegionPanelScrollPane;
    private JTree BoundaryTree;

    public JPanel_Boundaries() {
        layout = new GridBagLayout();
        setLayout(layout);


        BoundaryTree = new JTree();

        //Baumeinstellungen
        BoundaryTree.setShowsRootHandles(true); //
        BoundaryTree.getSelectionModel().setSelectionMode(TreeSelectionModel.DISCONTIGUOUS_TREE_SELECTION);//Selectionmodell einstzen beliebige Selection ermöglichen

        //Tree Renderer einstellen
        DefaultTreeCellRenderer renderer = (DefaultTreeCellRenderer) BoundaryTree.getCellRenderer();
        renderer.setLeafIcon(null);//keine Icons
        renderer.setClosedIcon(null);
        renderer.setOpenIcon(null);
        renderer.setBackground(Color.white);//Hintergurnd
        BoundaryTree.setRootVisible(false);//rootknoten ausblenden
        BoundaryTree.setVisibleRowCount(BoundaryTree.getRowCount()); //sichtbarkein Dynamisch


        //Tree in ScrollPanePacken
        RegionPanelScrollPane = new JScrollPane(BoundaryTree);
        GridBagConstraints GBCon = GetGridBagConstraints.get(0, 1, 1, 1, new Insets(
                0, 0, 0, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        RegionPanelScrollPane.setBackground(Color.white);
        RegionPanelScrollPane.setBorder(BorderFactory.createEtchedBorder(
                EtchedBorder.RAISED));
        this.add(RegionPanelScrollPane, GBCon);

    }

    public JTree getBoundaryTree() {
        return BoundaryTree;
    }
}
