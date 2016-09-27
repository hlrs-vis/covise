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
public class JPanel_Vectors extends JPanel {

    private final GridBagLayout layout;
    private JScrollPane VectorFieldFunctionPanelScrollPane;
    private JTree VectorFieldFunctionTree;

    public JPanel_Vectors() {
        layout = new GridBagLayout();
        setLayout(layout);


        VectorFieldFunctionTree = new JTree();

        //Baumeinstellungen
        VectorFieldFunctionTree.setShowsRootHandles(true); //
        VectorFieldFunctionTree.getSelectionModel().setSelectionMode(
                TreeSelectionModel.DISCONTIGUOUS_TREE_SELECTION);//Selectionmodell einstzen beliebige Selection ermöglichen

        //Tree Renderer einstellen
        DefaultTreeCellRenderer renderer = (DefaultTreeCellRenderer) VectorFieldFunctionTree.getCellRenderer();
        renderer.setLeafIcon(null);//keine Icons
        renderer.setClosedIcon(null);
        renderer.setOpenIcon(null);
        renderer.setBackground(Color.white);//Hintergurnd
        VectorFieldFunctionTree.setRootVisible(false);//rootknoten ausblenden
        VectorFieldFunctionTree.setVisibleRowCount(VectorFieldFunctionTree.getRowCount()); //sichtbarkein Dynamisch


        //Tree in ScrollPanePacken
        VectorFieldFunctionPanelScrollPane = new JScrollPane(VectorFieldFunctionTree);
        GridBagConstraints GBCon = GetGridBagConstraints.get(0, 1, 1, 1, new Insets(
                0, 0, 0, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        VectorFieldFunctionPanelScrollPane.setBackground(Color.white);
        VectorFieldFunctionPanelScrollPane.setBorder(BorderFactory.createEtchedBorder(
                EtchedBorder.RAISED));
        this.add(VectorFieldFunctionPanelScrollPane, GBCon);

    }

    public JTree getVectorFieldFunctionTree() {
        return VectorFieldFunctionTree;
    }
}
