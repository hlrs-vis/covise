package de.hlrs.starplugin.gui.dialogs;


import javax.swing.JOptionPane;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Error_Dialog extends JOptionPane {

    public Error_Dialog(Object message) {
        super(message);
        this.showMessageDialog(null, message);
    }
}
