package de.hlrs.starplugin.gui.ensight_export.chosen_jpanels.scrollpanes;

import java.awt.Color;
import javax.swing.BorderFactory;
import javax.swing.JScrollPane;
import javax.swing.JTree;
import javax.swing.border.EtchedBorder;
import javax.swing.tree.DefaultTreeCellRenderer;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class JScrollPane_ChosenGeomtryPanel extends JScrollPane {

    private JTree BaumChosenGeometry;


    public JScrollPane_ChosenGeomtryPanel() {
        setBackground(Color.white);
        setBorder(BorderFactory.createEtchedBorder(
                EtchedBorder.RAISED));
        this.setViewportView(this.ChosenGeometryTree());
    }

    private JTree ChosenGeometryTree() {


        //Baumerzeugen
        BaumChosenGeometry = new JTree();

        //Baumeinstellungen
        BaumChosenGeometry.setShowsRootHandles(true); //
        BaumChosenGeometry.setSelectionModel(null);//Selectionmodell einstzen keine Selection ermöglichen

        //Tree Renderer einstellen
        DefaultTreeCellRenderer renderer = (DefaultTreeCellRenderer) BaumChosenGeometry.getCellRenderer();
        renderer.setLeafIcon(null);//keine Icons
        renderer.setClosedIcon(null);
        renderer.setOpenIcon(null);
        renderer.setBackground(Color.white);//Hintergurnd
        BaumChosenGeometry.setRootVisible(false);//rootknoten ausblenden
        BaumChosenGeometry.setVisibleRowCount(BaumChosenGeometry.getRowCount()); //sichtbarkein Dynamisch

        return BaumChosenGeometry;
    }

    public JTree getBaumChosenGeometry() {
        return BaumChosenGeometry;
    }
}
