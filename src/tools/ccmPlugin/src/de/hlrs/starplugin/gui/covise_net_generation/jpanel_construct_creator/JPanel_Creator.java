package de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator;

import de.hlrs.starplugin.configuration.Configuration_Tool;
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
 *  @author Weiss HLRS Stuttgart
 */
public class JPanel_Creator extends JPanel {

    private JLabel JLabelSettings;
    final GridBagLayout layout;
    private JPanel_SettingsCardLayout JPanel_CardLayout_TypeSettings;
    private JButton JButton_CreateVisalizationConsturct_Geometry;
    private JButton JButton_CreateVisalizationConsturct_CuttingSurface;
    private JButton JButton_CreateVisalizationConsturct_CuttingSurfaceSeries;
    private JButton JButton_CreateVisalizationConsturct_Streamline;
    private JButton JButton_CreateVisalizationConsturct_IsoSurface;

    public JPanel_Creator() {

        //Rahmen
        setBorder(BorderFactory.createEtchedBorder(
                EtchedBorder.RAISED));
        //Layout
        layout = new GridBagLayout();
        setLayout(layout);

        //ButtonBar Construct Type
        //Label Type Chooser
        JPanel Type_Chooser = new JPanel();
        JLabel JLabel_TypeChooser = new JLabel("Type:");
        JLabel_TypeChooser.setFont(JLabel_TypeChooser.getFont().deriveFont(Font.BOLD));


        JButton_CreateVisalizationConsturct_Geometry = new JButton("Geometry");
        JButton_CreateVisalizationConsturct_Geometry.setActionCommand(
                Configuration_Tool.VisualizationType_Geometry);
        JButton_CreateVisalizationConsturct_CuttingSurface = new JButton("CuttingSurface");
        JButton_CreateVisalizationConsturct_CuttingSurface.setActionCommand(
                Configuration_Tool.VisualizationType_CuttingSurface);
        JButton_CreateVisalizationConsturct_CuttingSurfaceSeries = new JButton("CuttingSurfaceSeries");
        JButton_CreateVisalizationConsturct_CuttingSurfaceSeries.setActionCommand(
                Configuration_Tool.VisualizationType_CuttingSurface_Series);
        JButton_CreateVisalizationConsturct_Streamline = new JButton("Streamline");
        JButton_CreateVisalizationConsturct_Streamline.setActionCommand(
                Configuration_Tool.VisualizationType_Streamline);
        JButton_CreateVisalizationConsturct_IsoSurface = new JButton("IsoSurface");
        JButton_CreateVisalizationConsturct_IsoSurface.setActionCommand(
                Configuration_Tool.VisualizationType_IsoSurface);

        Type_Chooser.add(JLabel_TypeChooser);
        Type_Chooser.add(JButton_CreateVisalizationConsturct_Geometry);
        Type_Chooser.add(JButton_CreateVisalizationConsturct_CuttingSurface);
        Type_Chooser.add(JButton_CreateVisalizationConsturct_CuttingSurfaceSeries);
        Type_Chooser.add(JButton_CreateVisalizationConsturct_Streamline);
        Type_Chooser.add(JButton_CreateVisalizationConsturct_IsoSurface);

        GridBagConstraints GBCon = GetGridBagConstraints.get(0, 0, 0, 0, new Insets(
                10, 0, 0, 3),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridwidth = GridBagConstraints.REMAINDER;
        this.add(Type_Chooser, GBCon);


        //Headline (jLabel) erzeugen und hinzufügen
        JLabelSettings = new JLabel("<html>Settings</html>");
        JLabelSettings.setFont(JLabelSettings.getFont().deriveFont(Font.BOLD));
        GBCon = GetGridBagConstraints.get(0, 2, 0, 0, new Insets(
                5, 0, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridwidth = GridBagConstraints.REMAINDER;
        this.add(JLabelSettings, GBCon);

        //JPanel CardLayout Settings
        JPanel_CardLayout_TypeSettings = new JPanel_SettingsCardLayout();

        JPanel_CardLayout_TypeSettings.setBorder(BorderFactory.createEtchedBorder(
                EtchedBorder.RAISED));
        GBCon = GetGridBagConstraints.get(0, 3, 1, 1, new Insets(
                0, 0, 2, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridwidth = GridBagConstraints.REMAINDER;

        this.add(JPanel_CardLayout_TypeSettings, GBCon);


    }

    public JPanel_SettingsCardLayout getJPanel_CardLayout_TypeSettings() {
        return JPanel_CardLayout_TypeSettings;
    }

    public JButton getJButton_CreateVisalizationConsturct_CuttingSurface() {
        return JButton_CreateVisalizationConsturct_CuttingSurface;
    }

    public JButton getJButton_CreateVisalizationConsturct_CuttingSurfaceSeries() {
        return JButton_CreateVisalizationConsturct_CuttingSurfaceSeries;
    }

    public JButton getJButton_CreateVisalizationConsturct_Geometry() {
        return JButton_CreateVisalizationConsturct_Geometry;
    }

    public JButton getJButton_CreateVisalizationConsturct_IsoSurface() {
        return JButton_CreateVisalizationConsturct_IsoSurface;
    }

    public JButton getJButton_CreateVisalizationConsturct_Streamline() {
        return JButton_CreateVisalizationConsturct_Streamline;
    }
}
