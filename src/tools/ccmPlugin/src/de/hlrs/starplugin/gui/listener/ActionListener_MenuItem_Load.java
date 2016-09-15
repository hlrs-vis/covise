package de.hlrs.starplugin.gui.listener;

import Main.PluginContainer;
import de.hlrs.starplugin.configuration.Configuration_GUI_Strings;
import de.hlrs.starplugin.load_save.Loader;
import de.hlrs.starplugin.load_save.Message_Load;
import de.hlrs.starplugin.gui.dialogs.Error_Dialog;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.PrintWriter;
import java.io.StringWriter;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.filechooser.FileFilter;
import javax.swing.filechooser.FileNameExtensionFilter;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class ActionListener_MenuItem_Load implements ActionListener {

    private PluginContainer PC;

    public ActionListener_MenuItem_Load(PluginContainer PC) {
        super();
        this.PC = PC;


    }

    public void actionPerformed(ActionEvent e) {
        try {
            PC.getCNGDMan().getConMan().setSelectedConstruct(null);
            JFileChooser FileLoader = new JFileChooser();
            FileLoader.setCurrentDirectory(PC.getEEDMan().getExportPath());

            //File Loader Dialog
            FileLoader.setApproveButtonText("load");
            FileLoader.setDialogTitle("Load Export Configuration");
            FileLoader.setDialogType(FileLoader.OPEN_DIALOG);
            FileFilter filter = new FileNameExtensionFilter("COVISE Export Configurationfile (*.covsav)",
                    new String[]{"covsav"});
            FileLoader.setFileFilter(filter);
            int result = FileLoader.showDialog(PC.getGUI(), "Load");
            if (result == JFileChooser.APPROVE_OPTION) {
                File test = FileLoader.getSelectedFile();
                if (test.getAbsolutePath().endsWith(".covsav")) {
                    Message_Load Load_Result_Message = Loader.LoadState(PC, FileLoader.getSelectedFile());
                    PC.getSim().println("Loaded Configuration: " + FileLoader.getSelectedFile().toString());
                    if (Load_Result_Message.isTrouble()) {
                        JOptionPane.showMessageDialog(PC.getGUI(), "The following amount of Objects could not be found in this Simualtion!" + Configuration_GUI_Strings.eol + Load_Result_Message.getResultMessage(), "Load Result", JOptionPane.ERROR_MESSAGE);

                    }
                } else {
                    throw new Exception("Loading failed, no .covsav File");
                }
            }
            if (result == JFileChooser.ERROR_OPTION) {
                throw new Exception("Loading failed");
            }

        } catch (Exception Ex) {


            StringWriter sw = new StringWriter();
            Ex.printStackTrace(new PrintWriter(sw));
            new Error_Dialog(Configuration_GUI_Strings.Occourence + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.ErrMass + Ex.getMessage() + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.StackTrace + sw.toString());
            this.PC.getContr_EE().setEnabled(true);
            this.PC.getContr_CNG().setEnabled(true);
        }

    }
}
