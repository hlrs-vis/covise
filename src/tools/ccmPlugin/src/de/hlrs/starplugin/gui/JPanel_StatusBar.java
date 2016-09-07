/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package de.hlrs.starplugin.gui;

import de.hlrs.starplugin.configuration.Configuration_GUI_Strings;
import de.hlrs.starplugin.interfaces.Interface_MainFrame_StatusChangedListener;
import de.hlrs.starplugin.configuration.Configuration_Tool;

import de.hlrs.starplugin.util.GetGridBagConstraints;
import java.awt.Color;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import javax.swing.BorderFactory;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.border.EmptyBorder;
import javax.swing.border.LineBorder;

/**
 *
 *  @author Weiss HLRS Stuttgart
 */
public class JPanel_StatusBar extends JPanel implements Interface_MainFrame_StatusChangedListener {

    //GUI Bausteine für Statusbar
    private JLabel JLabel_Ensight;
    private JLabel JLabel_CoviseNetGen;
    private JFrame Parent;
    private GridBagLayout layout;
    private JLabel Headline;
    private EmptyBorder Border_E;
    private LineBorder Border_L_Black;
    private LineBorder Border_L_Yellow;

    public JPanel_StatusBar(JFrame Main) {
        super();
        this.Parent = Main;
        layout = new GridBagLayout();
        this.setLayout(layout);


        Border_E = new EmptyBorder(2, 2, 2, 2);
        Border_L_Black = new LineBorder(Color.black);
        Border_L_Yellow = new LineBorder(Color.yellow);

        JPanel status = new JPanel();
        JLabel_Ensight = new JLabel(Configuration_GUI_Strings.EnsightExport);

        JLabel_CoviseNetGen = new JLabel(Configuration_GUI_Strings.CoviseNetGeneration);
        JLabel_Ensight.setBorder(BorderFactory.createCompoundBorder(Border_L_Black, Border_E));
        JLabel_CoviseNetGen.setBorder(BorderFactory.createCompoundBorder(Border_L_Black, Border_E));
        JLabel_Ensight.setFont(new Font("Dialog", 1, 15));
        JLabel_CoviseNetGen.setFont(new Font("Dialog", 1, 15));

        status.setLayout(new FlowLayout());
        status.add(JLabel_Ensight);
        status.add(JLabel_CoviseNetGen);
        GridBagConstraints GBCon = GetGridBagConstraints.get(0, 0, 0, 0, new Insets(0, -5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.LINE_START);

        this.add(status, GBCon);

        //Filler

        GBCon = GetGridBagConstraints.get(1, 0, 1, 1, new Insets(0, 0, 0, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridwidth = 1;
        this.add(new JPanel(), GBCon);
    }

    public void onChange(int Status) {

        if (Configuration_Tool.STATUS_ENSIGHTEXPORT.equals(Status)) {


            JLabel_Ensight.setBorder(BorderFactory.createCompoundBorder(Border_L_Black, Border_E));
            JLabel_CoviseNetGen.setBorder(null);

        }
        if (Configuration_Tool.STATUS_NETGENERATION.equals(Status)) {

            JLabel_Ensight.setBorder(null);
            JLabel_CoviseNetGen.setBorder(BorderFactory.createCompoundBorder(Border_L_Black, Border_E));

        }

    }

    public JLabel getJLabel_CoviseNetGen() {
        return JLabel_CoviseNetGen;
    }

    public JLabel getJLabel_Ensight() {
        return JLabel_Ensight;
    }

}


