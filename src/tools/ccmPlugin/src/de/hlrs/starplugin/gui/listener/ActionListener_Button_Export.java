package de.hlrs.starplugin.gui.listener;

import Main.EnsightExportManager;
import de.hlrs.starplugin.gui.dialogs.Message_Dialog;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class ActionListener_Button_Export implements ActionListener {

    private EnsightExportManager EEMan;

    public ActionListener_Button_Export(EnsightExportManager EEMan) {
        super();
        this.EEMan = EEMan;

    }

    public void actionPerformed(ActionEvent e) {
        EEMan.ExportSimulation();
        new Message_Dialog("Ensight Export Done!");
    }
}
