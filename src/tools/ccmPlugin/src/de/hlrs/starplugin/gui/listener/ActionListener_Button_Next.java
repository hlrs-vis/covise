package de.hlrs.starplugin.gui.listener;

import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.gui.JPanel_MainContent;
import java.awt.CardLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class ActionListener_Button_Next implements ActionListener {

    private CardLayout cardLayout;
    private JPanel_MainContent Container;

    public ActionListener_Button_Next(JPanel_MainContent Con) {
        super();
        cardLayout = ((CardLayout) Con.getLayout());
        Container = Con;

    }

    public void actionPerformed(ActionEvent e) {
        Container.getMainFrame().changeStatus(Configuration_Tool.STATUS_NETGENERATION);


    }
}
