package de.hlrs.starplugin.gui.listener;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.JFrame;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class ActionListener_Button_Exit implements ActionListener {

    private JFrame main;

    public ActionListener_Button_Exit(JFrame J) {
        this.main = J;
    }

    public void actionPerformed(ActionEvent e) {
        main.dispose();
    }
}
