package de.hlrs.starplugin.gui.listener;

import Main.PluginContainer;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Manager_Controller {

    public static void add(PluginContainer PC) {
        PC.getGUI().getButtonbar().getButtonExit().addActionListener(new ActionListener_Button_Exit(PC.getGUI()));
        PC.getGUI().getButtonbar().getButtonExport().addActionListener(new ActionListener_Button_Export(
                PC.getEEMan()));
        PC.getGUI().getButtonbar().getButtonNext().addActionListener(new ActionListener_Button_Next(PC.getGUI().
                getJPanelMainContent()));
        PC.getGUI().getButtonbar().getButtonBack().addActionListener(new ActionListener_Button_Back(PC));
        PC.getGUI().getButtonbar().getButtonNetGeneration().addActionListener(new ActionListener_Button_CreateNet(
                PC));

        PC.getGUI().getJMenuItem_Load().addActionListener(new ActionListener_MenuItem_Load(PC));
        PC.getGUI().getJMenuItem_Save().addActionListener(new ActionListener_MenuItem_Save(PC));

        PC.getGUI().getButtonbar().getButtonFinish().addActionListener(new ActionListener_Button_Finish(PC));
        PC.getGUI().getJPanelMainContent().getMainFrame().getStatusbar().getJLabel_Ensight().addMouseListener(
                new MouseListener_Click_StatusBar_Ensight(PC));
        PC.getGUI().getJPanelMainContent().getMainFrame().getStatusbar().getJLabel_CoviseNetGen().
                addMouseListener(new MouseListener_Click_StatusBar_NetGen(PC));

    }
}
