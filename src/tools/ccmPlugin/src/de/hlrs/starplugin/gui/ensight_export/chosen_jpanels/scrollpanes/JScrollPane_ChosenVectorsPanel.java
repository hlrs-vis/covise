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
public class JScrollPane_ChosenVectorsPanel extends JScrollPane {

    private JTree BaumChosenVectors;

    public JScrollPane_ChosenVectorsPanel() {
        setBackground(Color.white);
        setBorder(BorderFactory.createEtchedBorder(
                EtchedBorder.RAISED));
        this.setViewportView(this.ChosenVectorsTree());
    }

    private JTree ChosenVectorsTree() {
        //Baumerzeugen
        BaumChosenVectors = new JTree();

        //Baumeinstellungen
        BaumChosenVectors.setShowsRootHandles(true); //
        BaumChosenVectors.setSelectionModel(null);//Selectionmodell einstzen keine Selection ermöglichen

        //Tree Renderer einstellen
        DefaultTreeCellRenderer renderer = (DefaultTreeCellRenderer) BaumChosenVectors.getCellRenderer();
        renderer.setLeafIcon(null);//keine Icons
        renderer.setClosedIcon(null);
        renderer.setOpenIcon(null);
        renderer.setBackground(Color.white);//Hintergurnd
        BaumChosenVectors.setRootVisible(false);//rootknoten ausblenden
        BaumChosenVectors.setVisibleRowCount(BaumChosenVectors.getRowCount()); //sichtbarkein Dynamisch

        return BaumChosenVectors;

    }

    public JTree getBaumChosenVectors() {
        return BaumChosenVectors;
    }

}
