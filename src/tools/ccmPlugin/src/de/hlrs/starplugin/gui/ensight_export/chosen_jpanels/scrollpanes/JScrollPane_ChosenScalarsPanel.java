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
public class JScrollPane_ChosenScalarsPanel extends JScrollPane {

    private JTree BaumChosenScalars;

    public JScrollPane_ChosenScalarsPanel() {
        setBackground(Color.white);
        setBorder(BorderFactory.createEtchedBorder(
                EtchedBorder.RAISED));
        this.setViewportView(this.ChosenScalarsTree());
    }
    private JTree ChosenScalarsTree(){
                //Baumerzeugen
        BaumChosenScalars = new JTree();

        //Baumeinstellungen
        BaumChosenScalars.setShowsRootHandles(true); //
        BaumChosenScalars.setSelectionModel(null);//Selectionmodell einstzen keine Selection ermöglichen

        //Tree Renderer einstellen
        DefaultTreeCellRenderer renderer = (DefaultTreeCellRenderer) BaumChosenScalars.getCellRenderer();
        renderer.setLeafIcon(null);//keine Icons
        renderer.setClosedIcon(null);
        renderer.setOpenIcon(null);
        renderer.setBackground(Color.white);//Hintergurnd
        BaumChosenScalars.setRootVisible(false);//rootknoten ausblenden
        BaumChosenScalars.setVisibleRowCount(BaumChosenScalars.getRowCount()); //sichtbarkein Dynamisch

        return BaumChosenScalars;

    }

    public JTree getBaumChosenScalars() {
        return BaumChosenScalars;
    }
    
}
