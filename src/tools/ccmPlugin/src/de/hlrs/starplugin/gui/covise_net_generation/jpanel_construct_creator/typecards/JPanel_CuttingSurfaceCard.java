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
public class JPanel_CuttingSurfaceCard extends JPanel {

    private GridBagLayout layout;
    private JLabel JLabel_Direction;
    private JLabel JLabel_Distance;
    private JTextField JTextField_Distance;
    private JRadioButton JRadioButton_X_Direction;
    private JRadioButton JRadioButton_Y_Direction;
    private JRadioButton JRadioButton_Z_Direction;
    private ButtonGroup BGroup;
    private JTree JTree_ChosenGeometry;
    private JTree JTree_GeometrytoVisualize;
    private JComboBox<String> JComboBox_DataTypeChooser;
    private JComboBox<Object> JComboBox_DataChooser;
    private final JLabel JLabel_Distance_InvalidInput;

    public JPanel_CuttingSurfaceCard() {
        super();
        layout = new GridBagLayout();
        setLayout(layout);
        setBackground(Color.white);
        Dimension Dimension_inputField = new Dimension(70, 19);


        JPanel ComboBoxPanel = new JPanel();
        ComboBoxPanel.setBackground(Color.white);
        //Label Data Chooser
        JLabel JLabel_DataChooser = new JLabel("Data:");
        JLabel_DataChooser.setFont(JLabel_DataChooser.getFont().deriveFont(Font.BOLD));
        ComboBoxPanel.add(JLabel_DataChooser);

        //Combobox DataType Chooser
        JComboBox_DataTypeChooser = new JComboBox<String>(
                new String[]{Configuration_Tool.DataType_scalar, Configuration_Tool.DataType_vector});
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


        //Direction Linie
        JLabel_Direction = new JLabel("Direction");
        GBCon = GetGridBagConstraints.get(0, 1, 0, 0, new Insets(
                0, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.LINE_START);
        GBCon.gridwidth = 1;
        this.add(JLabel_Direction, GBCon);

        JRadioButton_X_Direction = new JRadioButton("X");
        JRadioButton_X_Direction.setBackground(Color.white);
        JRadioButton_X_Direction.setActionCommand(Configuration_Tool.RadioButtonActionCommand_X_Direction);


        JRadioButton_Y_Direction = new JRadioButton("Y");
        JRadioButton_Y_Direction.setBackground(Color.white);
        JRadioButton_Y_Direction.setActionCommand(Configuration_Tool.RadioButtonActionCommand_Y_Direction);

        JRadioButton_Z_Direction = new JRadioButton("Z");
        JRadioButton_Z_Direction.setBackground(Color.white);
        JRadioButton_Z_Direction.setActionCommand(Configuration_Tool.RadioButtonActionCommand_Z_Direction);

        BGroup = new ButtonGroup();
        BGroup.add(JRadioButton_X_Direction);
        BGroup.add(JRadioButton_Y_Direction);
        BGroup.add(JRadioButton_Z_Direction);

        JPanel XYZ_Panel = new JPanel();
        XYZ_Panel.setBackground(Color.white);
        XYZ_Panel.add(JRadioButton_X_Direction);
        XYZ_Panel.add(JRadioButton_Y_Direction);
        XYZ_Panel.add(JRadioButton_Z_Direction);
        JRadioButton_X_Direction.setSelected(true);

        GBCon = GetGridBagConstraints.get(1, 1, 0, 0, new Insets(
                0, 0, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.LINE_START);
        GBCon.gridwidth = GridBagConstraints.REMAINDER;
        this.add(XYZ_Panel, GBCon);


        //Line Distance to Origin
        JLabel_Distance = new JLabel("Distance to Origin");

        GBCon = GetGridBagConstraints.get(0, 2, 0, 0, new Insets(
                5, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.LINE_START);
        GBCon.gridwidth = 1;
        this.add(JLabel_Distance, GBCon);

        JTextField_Distance = new JTextField("0");
        GBCon = GetGridBagConstraints.get(1, 2, 0, 0, new Insets(5, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.LINE_START);
        JTextField_Distance.setPreferredSize(Dimension_inputField);
        this.add(JTextField_Distance, GBCon);

        JLabel_Distance_InvalidInput = new JLabel("Invalid Input!");
        JLabel_Distance_InvalidInput.setBackground(Color.white);
        JLabel_Distance_InvalidInput.setForeground(Color.red);
        GBCon = GetGridBagConstraints.get(2, 2, 0, 0, new Insets(5, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridwidth=GridBagConstraints.REMAINDER;
        JLabel_Distance_InvalidInput.setVisible(false);

        this.add(JLabel_Distance_InvalidInput, GBCon);



        //Chosen Gemtery Tree
        //Headline
        JLabel Headline = new JLabel("Choose Regions to cut");

        GBCon = GetGridBagConstraints.get(0, 3, 0, 0, new Insets(5, 5, 0, 0),
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

        GBCon = GetGridBagConstraints.get(0, 4, 1, 1, new Insets(0, 5, 0, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = GridBagConstraints.REMAINDER;
        GBCon.gridwidth = 3;

        this.add(GeometryTreeScrollPane, GBCon);

        //JTree Geometry Visualisation
        //Headline
        JScrollPane GeometryVisualisationTreeScrollPane = new JScrollPane();
        JLabel Headline2 = new JLabel("Regions to cut");

        GBCon = GetGridBagConstraints.get(3, 3, 0, 0, new Insets(5, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = 1;
        GBCon.gridwidth = 1;

        this.add(Headline2, GBCon);

        //Baumeinstellungen
        JTree_GeometrytoVisualize = new JTree();
        JTree_GeometrytoVisualize.setShowsRootHandles(true); //

        //Tree Renderer einstellen
        DefaultTreeCellRenderer Renderer2 = (DefaultTreeCellRenderer) JTree_GeometrytoVisualize.
                getCellRenderer();
        Renderer2.setLeafIcon(null);//keine Icons
        Renderer2.setClosedIcon(null);
        Renderer2.setOpenIcon(null);
        Renderer2.setBackground(Color.white);//Hintergurnd
        JTree_GeometrytoVisualize.setRootVisible(false);//rootknoten ausblenden
        JTree_GeometrytoVisualize.setVisibleRowCount(JTree_GeometrytoVisualize.getRowCount()); //sichtbarkeit Dynamisch
        GeometryVisualisationTreeScrollPane.add(JTree_GeometrytoVisualize);
        GeometryVisualisationTreeScrollPane.setViewportView(JTree_GeometrytoVisualize);

        GBCon = GetGridBagConstraints.get(3, 4, 1, 1, new Insets(0, 5, 0, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = GridBagConstraints.REMAINDER;
        GBCon.gridwidth = GridBagConstraints.REMAINDER;

        this.add(GeometryVisualisationTreeScrollPane, GBCon);
    }

    public ButtonGroup getBGroup() {
        return BGroup;
    }

    public JTextField getJTextField_Distance() {
        return JTextField_Distance;
    }

    public JTree getJTree_ChosenGeometry() {
        return JTree_ChosenGeometry;
    }

    public JTree getJTree_GeometrytoVisualize() {
        return JTree_GeometrytoVisualize;
    }

    public JRadioButton getJRadioButton_X_Direction() {
        return JRadioButton_X_Direction;
    }

    public JRadioButton getJRadioButton_Y_Direction() {
        return JRadioButton_Y_Direction;
    }

    public JRadioButton getJRadioButton_Z_Direction() {
        return JRadioButton_Z_Direction;
    }

    public JComboBox<Object> getJComboBox_DataChooser() {
        return JComboBox_DataChooser;
    }

    public JComboBox<String> getJComboBox_DataTypeChooser() {
        return JComboBox_DataTypeChooser;
    }

    public JLabel getJLabel_Distance_InvalidInput() {
        return JLabel_Distance_InvalidInput;
    }
}
