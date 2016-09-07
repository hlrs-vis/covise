/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package de.hlrs.starplugin.gui.covise_net_generation;

import de.hlrs.starplugin.util.GetGridBagConstraints;
import java.awt.Color;
import java.awt.Font;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;

/**
 *
 *  @author Weiss HLRS Stuttgart
 */
public class JPanel_CoviseNetGenerationFolder extends JPanel {

    final GridBagLayout layout;
    private JLabel LabelExportDestinationHeadline;
    private JLabel LabelDestinationFile;
    private JButton ButtonBrowse;
    private String ExportPath;

    public JPanel_CoviseNetGenerationFolder() {
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
                GridBagConstraints.CENTER);
        LabelDestinationFile.setOpaque(true);
        LabelDestinationFile.setBorder(BorderFactory.createEtchedBorder());
        this.add(LabelDestinationFile, GBCon);

        //Browse Button
        ButtonBrowse = new JButton("Browse...");
        GBCon = GetGridBagConstraints.get(2, 0, 0, 1, new Insets(
                0, 10, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_START);
        this.add(ButtonBrowse, GBCon);


        LabelDestinationFile.setPreferredSize(ButtonBrowse.getPreferredSize());
    }

    public JLabel getLabelDestinationFile() {
        return this.LabelDestinationFile;
    }

    public JButton getButtonBrowse() {
        return this.ButtonBrowse;
    }



}
