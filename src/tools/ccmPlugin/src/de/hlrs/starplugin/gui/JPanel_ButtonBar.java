/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package de.hlrs.starplugin.gui;

import de.hlrs.starplugin.configuration.Configuration_GUI_Strings;
import de.hlrs.starplugin.interfaces.Interface_MainFrame_StatusChangedListener;
import de.hlrs.starplugin.configuration.Configuration_Tool;

import de.hlrs.starplugin.util.GetGridBagConstraints;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import javax.swing.JButton;
import javax.swing.JPanel;

/**
 *
 *  @author Weiss HLRS Stuttgart
 */
public class JPanel_ButtonBar extends JPanel implements Interface_MainFrame_StatusChangedListener {
    //GUI Bausteine für JPanel_ButtonBar

    private JButton ButtonNext;
    private JButton ButtonBack;
    private JButton ButtonExport;
    private JButton ButtonExit;
    private JButton ButtonFinish;
    private JButton ButtonNetGeneration;
    private final GridBagLayout layout;
    private JFrame_MainFrame Mainframe;

    public JPanel_ButtonBar(JFrame_MainFrame Main) {
        Mainframe = Main;
        layout = new GridBagLayout();
        ButtonNext = new JButton();
        ButtonBack = new JButton();
        ButtonExport = new JButton();
        ButtonExit = new JButton();

        ButtonFinish = new JButton();
        ButtonNetGeneration = new JButton();



        ButtonNext.setText(Configuration_GUI_Strings.Next);
        ButtonBack.setText(Configuration_GUI_Strings.Back);
        ButtonExport.setText(Configuration_GUI_Strings.Export);

        ButtonFinish.setText(Configuration_GUI_Strings.Finish);
        ButtonExit.setText(Configuration_GUI_Strings.Exit);


        ButtonNetGeneration.setText(Configuration_GUI_Strings.CreateNet);

        setLayout(layout);

        GridBagConstraints GBCon = GetGridBagConstraints.get(0, 0, 0, 0, new Insets(
                0, 0, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_START);

        this.add(ButtonBack, GBCon);

        GBCon = GetGridBagConstraints.get(1, 0, 0, 0, new Insets(0, 0, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_START);
        this.add(ButtonNext, GBCon);

        GBCon = GetGridBagConstraints.get(2, 0, 1, 1, new Insets(0, 0, 0, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        this.add(new JPanel(), GBCon);

        GBCon = GetGridBagConstraints.get(3, 0, 0, 0, new Insets(0, 0, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_START);
        this.add(ButtonExport, GBCon);

        GBCon = GetGridBagConstraints.get(4, 0, 0, 0, new Insets(0, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_START);
        this.add(ButtonNetGeneration, GBCon);

        GBCon = GetGridBagConstraints.get(5, 0, 1, 1, new Insets(0, 0, 0, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        this.add(new JPanel(), GBCon);




        GBCon = GetGridBagConstraints.get(8, 0, 0, 0, new Insets(0, 0, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_END);
        this.add(ButtonFinish, GBCon);

        GBCon = GetGridBagConstraints.get(9, 0, 0, 0, new Insets(0, 0, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_END);
        this.add(ButtonExit, GBCon);
    }

    public JButton getButtonBack() {
        return ButtonBack;
    }

    public JButton getButtonExit() {
        return ButtonExit;
    }

    public JButton getButtonExport() {
        return ButtonExport;
    }

    public JButton getButtonNext() {
        return ButtonNext;
    }

    public JButton getButtonNetGeneration() {
        return ButtonNetGeneration;
    }

    public JButton getButtonFinish() {
        return ButtonFinish;
    }

    public void onChange(int Status) {
        if (Configuration_Tool.STATUS_ENSIGHTEXPORT.equals(Status)) {
            this.getButtonBack().setEnabled(false);
            this.getButtonNext().setEnabled(true);
            this.getButtonExport().setEnabled(true);
            this.getButtonNetGeneration().setEnabled(false);
            this.getButtonFinish().setEnabled(false);
        }
        if (Configuration_Tool.STATUS_NETGENERATION.equals(Status)) {
            this.getButtonBack().setEnabled(true);
            this.getButtonNext().setEnabled(false);
            this.getButtonExport().setEnabled(false);
            this.getButtonNetGeneration().setEnabled(true);
            this.getButtonFinish().setEnabled(true);

        }
    }
}
