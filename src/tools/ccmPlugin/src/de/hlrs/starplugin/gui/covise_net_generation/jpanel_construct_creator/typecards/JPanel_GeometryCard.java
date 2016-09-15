package de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards;

import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.util.GetGridBagConstraints;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import javax.swing.ButtonGroup;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JScrollPane;
import javax.swing.JTextField;
import javax.swing.JTree;
import javax.swing.tree.DefaultTreeCellRenderer;

/**
 *
 *  @author Weiss HLRS Stuttgart
 */
public class JPanel_GeometryCard extends JPanel {

    private GridBagLayout layout;
    private JTree JTree_ChosenGeometry;
    private JTree JTree_GeometrytoVisualize;
    private JComboBox<String> JComboBox_DataTypeChooser;
    private JComboBox<Object> JComboBox_DataChooser;
    private final JLabel JLabel_SurfaceColor;
    private final JRadioButton JRadioButton_Grey;
    private final ButtonGroup BGroup;
    private final JRadioButton JRadioButton_Red;
    private final JRadioButton JRadioButton_Green;
    private final JTextField JTextField_Transparency;
    private final JPanel JPanel_SurfaceColor;

    public JPanel_GeometryCard() {

        layout = new GridBagLayout();
        setLayout(layout);
        setBackground(Color.white);
        Dimension Dimension_inputField = new Dimension(70, 19);


//Line Data Choice

        JPanel ComboBoxPanel = new JPanel();
        ComboBoxPanel.setBackground(Color.white);
        //Label Data Chooser
        JLabel JLabel_DataChooser = new JLabel("Data:");
        JLabel_DataChooser.setFont(JLabel_DataChooser.getFont().deriveFont(Font.BOLD));
        ComboBoxPanel.add(JLabel_DataChooser);

        //Combobox DataType Chooser
        JComboBox_DataTypeChooser = new JComboBox<String>(
                new String[]{Configuration_Tool.DataType_none, Configuration_Tool.DataType_scalar, Configuration_Tool.DataType_vector});
        JComboBox_DataTypeChooser.setBackground(Color.white);
        JComboBox_DataTypeChooser.setEnabled(true);
        ComboBoxPanel.add(JComboBox_DataTypeChooser);

        //Combobox Data Chooser
        JComboBox_DataChooser = new JComboBox<Object>();
        JComboBox_DataChooser.setBackground(Color.white);
        JComboBox_DataChooser.setEnabled(true);
        ComboBoxPanel.add(JComboBox_DataChooser);
        GridBagConstraints GBCon = GetGridBagConstraints.get(0, 0, 0, 0, new Insets(
                0, 0, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridwidth = GridBagConstraints.REMAINDER;
        this.add(ComboBoxPanel, GBCon);

        //Line Color of Surface

        JLabel_SurfaceColor = new JLabel("Surface Color");
        GBCon = GetGridBagConstraints.get(0, 1, 0, 0, new Insets(
                0, 0, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.LINE_START);
        GBCon.gridwidth = 1;
        this.add(JLabel_SurfaceColor, GBCon);

        JRadioButton_Grey = new JRadioButton("grey");
        JRadioButton_Grey.setBackground(Color.white);
        JRadioButton_Grey.setActionCommand(Configuration_Tool.Color_grey);


        JRadioButton_Red = new JRadioButton("red");
        JRadioButton_Red.setBackground(Color.white);
        JRadioButton_Red.setActionCommand(Configuration_Tool.Color_red);

        JRadioButton_Green = new JRadioButton("green");
        JRadioButton_Green.setBackground(Color.white);
        JRadioButton_Green.setActionCommand(Configuration_Tool.Color_green);

        BGroup = new ButtonGroup();
        BGroup.add(JRadioButton_Grey);
        BGroup.add(JRadioButton_Red);
        BGroup.add(JRadioButton_Green);

        JTextField_Transparency = new JTextField("0.5");
        JTextField_Transparency.setPreferredSize(Dimension_inputField);

        JPanel_SurfaceColor = new JPanel();
        JPanel_SurfaceColor.setBackground(Color.white);
        JPanel_SurfaceColor.add(this.JLabel_SurfaceColor);
        JPanel_SurfaceColor.add(JRadioButton_Grey);
        JPanel_SurfaceColor.add(JRadioButton_Red);
        JPanel_SurfaceColor.add(JRadioButton_Green);
        JPanel_SurfaceColor.add(new JLabel("Transparency:"));
        JPanel_SurfaceColor.add(JTextField_Transparency);
        JRadioButton_Grey.setSelected(true);

        GBCon = GetGridBagConstraints.get(1, 1, 0, 0, new Insets(
                0, 0, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.LINE_START);
        GBCon.gridwidth = GridBagConstraints.REMAINDER;
        this.add(JPanel_SurfaceColor, GBCon);


        //Chosen Gemtery Tree
        //Headline
        JLabel Headline = new JLabel("Choose Regions/Surfaces");

        GBCon = GetGridBagConstraints.get(0, 2, 0, 0, new Insets(5, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = 1;
        GBCon.gridwidth = 2;
        this.add(Headline, GBCon);

        JScrollPane GeometryTreeScrollPane = new JScrollPane();

        //Baumeinstellungen
        JTree_ChosenGeometry = new JTree();
        JTree_ChosenGeometry.setShowsRootHandles(true); //

        //Tree Renderer einstellen
        DefaultTreeCellRenderer renderer = (DefaultTreeCellRenderer) JTree_ChosenGeometry.getCellRenderer();
        renderer.setLeafIcon(null);//keine Icons
        renderer.setClosedIcon(null);
        renderer.setOpenIcon(null);
        renderer.setBackground(Color.white);//Hintergurnd
        JTree_ChosenGeometry.setRootVisible(false);//rootknoten ausblenden
        JTree_ChosenGeometry.setVisibleRowCount(JTree_ChosenGeometry.getRowCount()); //sichtbarkeit Dynamisch
        GeometryTreeScrollPane.add(JTree_ChosenGeometry);
        GeometryTreeScrollPane.setViewportView(JTree_ChosenGeometry);

        GBCon = GetGridBagConstraints.get(0, 3, 1, 1, new Insets(0, 5, 0, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = GridBagConstraints.REMAINDER;
        GBCon.gridwidth = 3;

        this.add(GeometryTreeScrollPane, GBCon);

        //JTree Geometry Visualisation
        //Headline
        JScrollPane GeometryVisualisationTreeScrollPane = new JScrollPane();
        JLabel Headline2 = new JLabel("Regions/Surfaces Visualization");

        GBCon = GetGridBagConstraints.get(3, 2, 0, 0, new Insets(5, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = 1;
        GBCon.gridwidth = 1;

        this.add(Headline2, GBCon);

        //Baumeinstellungen
        JTree_GeometrytoVisualize = new JTree();
        JTree_GeometrytoVisualize.setShowsRootHandles(true); //
        JTree_GeometrytoVisualize.setSelectionModel(null);//Selectionmodell einstzen keine Selection ermöglichen

        //Tree Renderer einstellen
        DefaultTreeCellRenderer Renderer2 = (DefaultTreeCellRenderer) JTree_GeometrytoVisualize.getCellRenderer();
        Renderer2.setLeafIcon(null);//keine Icons
        Renderer2.setClosedIcon(null);
        Renderer2.setOpenIcon(null);
        Renderer2.setBackground(Color.white);//Hintergurnd
        JTree_GeometrytoVisualize.setRootVisible(false);//rootknoten ausblenden
        JTree_GeometrytoVisualize.setVisibleRowCount(JTree_GeometrytoVisualize.getRowCount()); //sichtbarkeit Dynamisch
        GeometryVisualisationTreeScrollPane.add(JTree_GeometrytoVisualize);
        GeometryVisualisationTreeScrollPane.setViewportView(JTree_GeometrytoVisualize);

        GBCon = GetGridBagConstraints.get(3, 3, 1, 1, new Insets(0, 5, 0, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = GridBagConstraints.REMAINDER;
        GBCon.gridwidth = GridBagConstraints.REMAINDER;

        this.add(GeometryVisualisationTreeScrollPane, GBCon);

    }

    public JTree getJTree_ChosenGeometry() {
        return JTree_ChosenGeometry;
    }

    public JTree getJTree_GeometrytoVisualize() {
        return JTree_GeometrytoVisualize;
    }

    public JComboBox<Object> getJComboBox_DataChooser() {
        return JComboBox_DataChooser;
    }

    public JComboBox<String> getJComboBox_DataTypeChooser() {
        return JComboBox_DataTypeChooser;
    }

    public JTextField getJTextField_Transparency() {
        return JTextField_Transparency;
    }

    public JPanel getJPanel_SurfaceColor() {
        return JPanel_SurfaceColor;
    }

    public JRadioButton getJRadioButton_Green() {
        return JRadioButton_Green;
    }

    public JRadioButton getJRadioButton_Grey() {
        return JRadioButton_Grey;
    }

    public JRadioButton getJRadioButton_Red() {
        return JRadioButton_Red;
    }

    public ButtonGroup getBGroup() {
        return BGroup;
    }
}
