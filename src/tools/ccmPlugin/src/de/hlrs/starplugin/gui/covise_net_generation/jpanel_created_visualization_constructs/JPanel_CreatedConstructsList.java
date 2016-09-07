package de.hlrs.starplugin.gui.covise_net_generation.jpanel_created_visualization_constructs;

import de.hlrs.starplugin.util.GetGridBagConstraints;
import java.awt.Font;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.border.EtchedBorder;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class JPanel_CreatedConstructsList extends JPanel {

    private JLabel JLabel_PartListHeadline;
    final GridBagLayout layout;
    private JScrollPane_CreatedConstructsList JScrollPane_JTree_VisualizationParts;
    private JButton JButton_DeleteVisualizationConstruct;
    private JButton JButton_CloneVisualizationConstruct;
    private final JButton JButton_Test;

    public JPanel_CreatedConstructsList() {
        //Rahmen
        setBorder(BorderFactory.createEtchedBorder(
                EtchedBorder.RAISED));
        //Layout
        layout = new GridBagLayout();
        setLayout(layout);

        //Headline (jLabel) erzeugen und hinzufügen
        JLabel_PartListHeadline = new JLabel("<html>Created Visualization Constructs</html>");
        JLabel_PartListHeadline.setFont(new Font("Dialog", 1, 15));
        GridBagConstraints GBCon = GetGridBagConstraints.get(0, 0, 0, 0, new Insets(
                0, 0, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridwidth = GridBagConstraints.REMAINDER;
        this.add(JLabel_PartListHeadline, GBCon);

        //JTree Visualization Part List
        JScrollPane_JTree_VisualizationParts = new JScrollPane_CreatedConstructsList();
        GBCon = GetGridBagConstraints.get(0, 1, 1, 1, new Insets(
                0, 0, 2, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridwidth = GridBagConstraints.REMAINDER;
        GBCon.gridheight = 1;
        this.add(JScrollPane_JTree_VisualizationParts, GBCon);

        //ButtonBar
        JButton_CloneVisualizationConstruct = new JButton("Clone");
        GBCon = GetGridBagConstraints.get(0, 3, 0, 0, new Insets(0, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = 1;
        GBCon.gridwidth = 1;
        this.add(JButton_CloneVisualizationConstruct, GBCon);

        JButton_DeleteVisualizationConstruct = new JButton("Delete");
        GBCon = GetGridBagConstraints.get(1, 3, 0, 0, new Insets(0, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = 1;
        GBCon.gridwidth = 1;
        this.add(JButton_DeleteVisualizationConstruct, GBCon);

        //ButtonTest
        JButton_Test = new JButton("StarCCM+ Vis");
        GBCon = GetGridBagConstraints.get(2, 3, 0, 0, new Insets(0, 0, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_END);
        GBCon.gridheight = 1;
        GBCon.gridwidth = GridBagConstraints.REMAINDER;
        this.add(JButton_Test, GBCon);


    }

    public JScrollPane_CreatedConstructsList getJScrollPane_JTree_CreatedVisualizationConstructs() {
        return JScrollPane_JTree_VisualizationParts;
    }

    public JButton getJButton_DeleteVisualizationConstruct() {
        return JButton_DeleteVisualizationConstruct;
    }

    public JButton getJButton_CloneVisualizationConstruct() {
        return JButton_CloneVisualizationConstruct;
    }

    public JButton getJButton_Test() {
        return JButton_Test;
    }
}
