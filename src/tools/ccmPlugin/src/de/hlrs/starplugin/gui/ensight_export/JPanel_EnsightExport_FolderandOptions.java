package de.hlrs.starplugin.gui.ensight_export;

import de.hlrs.starplugin.util.GetGridBagConstraints;
import java.awt.Color;
import java.awt.Font;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import javax.swing.BorderFactory;
import javax.swing.GroupLayout;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JPanel;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class JPanel_EnsightExport_FolderandOptions extends JPanel {

    final GridBagLayout layout;
    private JLabel LabelExportDestinationHeadline;
    private JLabel LabelDestinationFile;
    private JCheckBox CheckBoxAppendFile;
    private JButton ButtonBrowse;
    private JCheckBox CheckBox_ResultsOnVertices;

    public JPanel_EnsightExport_FolderandOptions() {
        //Layout für JPanel festlegn
        layout = new GridBagLayout();
        setLayout(layout);

        //Ziel Ordner Label
        LabelExportDestinationHeadline = new JLabel(
                "<html>Export Destination: </html>");
        GridBagConstraints GBCon = GetGridBagConstraints.get(0, 0, 0, 1, new Insets(
                0, 0, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.CENTER);
        LabelExportDestinationHeadline.setFont(LabelExportDestinationHeadline.getFont().deriveFont(Font.BOLD));

        this.add(LabelExportDestinationHeadline, GBCon);

        // Ordner 
        LabelDestinationFile = new JLabel(
                "<html>Export Destination Destination File</html>");
        LabelDestinationFile.setBackground(Color.WHITE);
        GBCon = GetGridBagConstraints.get(1, 0, 1, 1, new Insets(
                0, 10, 0, 0),
                GridBagConstraints.HORIZONTAL,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridwidth = 2;
        LabelDestinationFile.setOpaque(true);
        LabelDestinationFile.setBorder(BorderFactory.createEtchedBorder());

        this.add(LabelDestinationFile, GBCon);

        //Browse Button
        ButtonBrowse = new JButton("Browse...");
        GBCon = GetGridBagConstraints.get(3, 0, 0, 1, new Insets(
                0, 10, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridwidth = 1;
        this.add(ButtonBrowse, GBCon);



        LabelDestinationFile.setPreferredSize(ButtonBrowse.getPreferredSize());
        //CheckBox Append to existing File
        CheckBoxAppendFile = new JCheckBox("Append to Existing File");
        GBCon = GetGridBagConstraints.get(1, 1, 0, 1, new Insets(
                5, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_START);
        this.add(CheckBoxAppendFile, GBCon);
        //CheckBox Export Results on Vertices
        CheckBox_ResultsOnVertices = new JCheckBox("Export Results on Vertices");
        GBCon = GetGridBagConstraints.get(2, 1, 0, 1, new Insets(
                5, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_START);
        this.add(CheckBox_ResultsOnVertices, GBCon);

        javax.swing.GroupLayout layouts = new GroupLayout(this);
        layouts.linkSize(javax.swing.SwingConstants.VERTICAL,
                new java.awt.Component[]{ButtonBrowse, LabelDestinationFile, LabelExportDestinationHeadline});


    }

    public JButton getButtonBrowse() {
        return ButtonBrowse;
    }

    public JLabel getLabelDestinationFile() {
        return LabelDestinationFile;
    }

    public JCheckBox getCheckBoxAppendFile() {
        return CheckBoxAppendFile;
    }

    public JCheckBox getCheckBox_ResultsOnVertices() {
        return CheckBox_ResultsOnVertices;
    }
}
