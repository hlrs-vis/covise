package de.hlrs.starplugin.gui.listener;

import Main.PluginContainer;
import de.hlrs.starplugin.configuration.Configuration_GUI_Strings;
import de.hlrs.starplugin.load_save.Saver;
import de.hlrs.starplugin.gui.dialogs.Error_Dialog;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.PrintWriter;
import java.io.StringWriter;

import javax.swing.JFileChooser;
import javax.swing.filechooser.FileFilter;
import javax.swing.filechooser.FileNameExtensionFilter;

/**
 *
 *@author Weiss HLRS Stuttgart
 */
public class ActionListener_MenuItem_Save implements ActionListener {

    private PluginContainer PC;

    public ActionListener_MenuItem_Save(PluginContainer PC) {
        super();
        this.PC = PC;


    }

    public void actionPerformed(ActionEvent e) {
        try {
            JFileChooser FileSaver = new JFileChooser();
            FileSaver.setCurrentDirectory(PC.getEEDMan().getExportPath());

            FileSaver.setApproveButtonText("save");
            FileSaver.setDialogTitle("Save Export Configuration");
            FileSaver.setDialogType(FileSaver.SAVE_DIALOG);
            FileFilter filter = new FileNameExtensionFilter("COVISE Export Configurationfile (*.covsav)",
                    new String[]{"covsav"});
            FileSaver.setFileFilter(filter);
            int result = FileSaver.showDialog(PC.getGUI(), "Save");
            if (result == JFileChooser.APPROVE_OPTION) {

                File test = FileSaver.getSelectedFile();
                if (test.getAbsolutePath().endsWith(".covsav")) {
                    Saver.SaveState(PC, FileSaver.getSelectedFile());
                    PC.getSim().println("Saved Configuration under: " + FileSaver.getSelectedFile().toString());
                } else {
                    File FilewithSuffix = new File(test.getAbsolutePath() + ".covsav");
                    Saver.SaveState(PC, FilewithSuffix);
                    PC.getSim().println("Saved Configuration under: " + FilewithSuffix.toString());
                }
            }
            if (result == JFileChooser.ERROR_OPTION) {
                throw new Exception("Saving failed");
            }

        } catch (Exception Ex) {
                StringWriter sw = new StringWriter();
                Ex.printStackTrace(new PrintWriter(sw));
                new Error_Dialog(Configuration_GUI_Strings.Occourence + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.ErrMass + Ex.getMessage() + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.StackTrace + sw.toString());
        }

    }
}
