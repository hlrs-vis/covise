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
public class JPanel_Scalars extends JPanel {

    private final GridBagLayout layout;
    private JScrollPane ScalarFieldFunctionPanelScrollPane;
    private JTree ScalarFieldFunctionTree;

    public JPanel_Scalars() {
        layout = new GridBagLayout();
        setLayout(layout);


        ScalarFieldFunctionTree = new JTree();

        //Baumeinstellungen
        ScalarFieldFunctionTree.setShowsRootHandles(true); //
        ScalarFieldFunctionTree.getSelectionModel().setSelectionMode(
                TreeSelectionModel.DISCONTIGUOUS_TREE_SELECTION);//Selectionmodell einstzen beliebige Selection ermöglichen

        //Tree Renderer einstellen
        DefaultTreeCellRenderer renderer = (DefaultTreeCellRenderer) ScalarFieldFunctionTree.getCellRenderer();
        renderer.setLeafIcon(null);//keine Icons
        renderer.setClosedIcon(null);
        renderer.setOpenIcon(null);
        renderer.setBackground(Color.white);//Hintergurnd
        ScalarFieldFunctionTree.setRootVisible(false);//rootknoten ausblenden
        ScalarFieldFunctionTree.setVisibleRowCount(ScalarFieldFunctionTree.getRowCount()); //sichtbarkein Dynamisch


        //Tree in ScrollPanePacken
        ScalarFieldFunctionPanelScrollPane = new JScrollPane(ScalarFieldFunctionTree);
        GridBagConstraints GBCon = GetGridBagConstraints.get(0, 1, 1, 1, new Insets(
                0, 0, 0, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        ScalarFieldFunctionPanelScrollPane.setBackground(Color.white);
        ScalarFieldFunctionPanelScrollPane.setBorder(BorderFactory.createEtchedBorder(
                EtchedBorder.RAISED));
        this.add(ScalarFieldFunctionPanelScrollPane, GBCon);

    }

    public JTree getScalarFieldFunctionTree() {
        return ScalarFieldFunctionTree;
    }
}
