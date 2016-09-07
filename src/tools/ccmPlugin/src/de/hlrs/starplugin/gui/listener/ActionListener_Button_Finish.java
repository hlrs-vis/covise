package de.hlrs.starplugin.gui.listener;

import Main.PluginContainer;
import de.hlrs.starplugin.configuration.Configuration_GUI_Strings;
import de.hlrs.starplugin.gui.dialogs.Error_Dialog;
import de.hlrs.starplugin.gui.dialogs.Message_Dialog;
import de.hlrs.starplugin.python.scriptcreation.PythonScript_Creator;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.PrintWriter;
import java.io.StringWriter;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class ActionListener_Button_Finish implements ActionListener {

    private PluginContainer PC;

    public ActionListener_Button_Finish(PluginContainer PC) {
        super();
        this.PC = PC;
    }

    public void actionPerformed(ActionEvent e) {
        this.PC.getEEMan().ExportSimulation();
        try {
            new PythonScript_Creator(this.PC).createScript();
            PC.getSim().print("Net Generation successful");
            new Message_Dialog("Pythonscript for COVISE Created!");
        } catch (Exception Ex) {
            StringWriter sw = new StringWriter();
            Ex.printStackTrace(new PrintWriter(sw));
            new Error_Dialog(Configuration_GUI_Strings.Occourence + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.ErrMass + Ex.getMessage() + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.StackTrace + sw.toString());
        }
    }
}
