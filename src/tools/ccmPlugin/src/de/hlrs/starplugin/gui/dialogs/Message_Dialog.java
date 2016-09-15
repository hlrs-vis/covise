package de.hlrs.starplugin.gui.dialogs;

import javax.swing.JOptionPane;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Message_Dialog extends JOptionPane {

    public Message_Dialog(Object message) {
        super(message);
        this.showMessageDialog(null, message, "Approve", JOptionPane.INFORMATION_MESSAGE);
    }
}
