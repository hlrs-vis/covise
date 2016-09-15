/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package de.hlrs.starplugin.gui.covise_net_generation;

import de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.JPanel_Creator;
import de.hlrs.starplugin.gui.covise_net_generation.jpanel_created_visualization_constructs.JPanel_CreatedConstructsList;
import de.hlrs.starplugin.util.GetGridBagConstraints;
import java.awt.Font;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import javax.swing.JLabel;
import javax.swing.JPanel;

/**
 *
 *  @author Weiss HLRS Stuttgart
 */
public class JPanel_CoviseNetGeneration extends JPanel {

    private final GridBagLayout layout;
    private JLabel JLabelCreator;
    private JPanel_CoviseNetGenerationFolder CoviseNetGenerationFolderPanel;
    private JPanel_Creator jPanelCreator;
    private JPanel_CreatedConstructsList jPanel_CreatedConstructsList;

    public JPanel_CoviseNetGeneration() {



        layout = new GridBagLayout();
        setLayout(layout);


        //File Destination
        CoviseNetGenerationFolderPanel = new JPanel_CoviseNetGenerationFolder();
        GridBagConstraints GBCon = GetGridBagConstraints.get(0, 0, 1, 0, new Insets(10, 0, 0, 0),
                GridBagConstraints.HORIZONTAL,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = 1;
        GBCon.gridwidth = 6;
        this.add(CoviseNetGenerationFolderPanel, GBCon);

        //Headline (jLabel) erzeugen und hinzufügen
        JLabelCreator = new JLabel("<html>Construct Creator</html>");
        JLabelCreator.setFont(new Font("Dialog", 1, 13));
        GBCon = GetGridBagConstraints.get(0, 1, 0, 0, new Insets(
                10, 0, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = 1;
        GBCon.gridwidth = 5;
        this.add(JLabelCreator, GBCon);

        //Creator
        jPanelCreator = new JPanel_Creator();
        GBCon = GetGridBagConstraints.get(0, 2, 1, 1, new Insets(0, 0, 5, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = 3;
        GBCon.gridwidth = 5;
        this.add(jPanelCreator, GBCon);



        //Part List
        jPanel_CreatedConstructsList = new JPanel_CreatedConstructsList();
        GBCon = GetGridBagConstraints.get(5, 2, 0.1f, 1, new Insets(0, 0, 5, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = 3;
        GBCon.gridwidth = 1;
        this.add(jPanel_CreatedConstructsList, GBCon);

    }

    public JPanel_CoviseNetGenerationFolder getCoviseNetGenerationFolderPanel() {
        return this.CoviseNetGenerationFolderPanel;
    }

    public JPanel_Creator getjPanelCreator() {
        return jPanelCreator;
    }

    public JPanel_CreatedConstructsList getjPanel_CreatedConstructsList() {
        return jPanel_CreatedConstructsList;
    }
}


