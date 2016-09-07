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
public class JPanel_StreamlineCard extends JPanel {

    private GridBagLayout layout;
    private JLabel JLabel_Divisions;
    private JTextField JTextField_DivisionX;
    private JTextField JTextField_DivisionY;
    private JTextField JTextField_DivisionZ;
    private JLabel JLabel_max_out_of_Domain;
    private JTextField JTextField_max_out_of_domain;
    private JTree JTree_ChosenGeometry;
    private JTree JTree_GeometrytoVisualize;
    private JLabel JLabel_trace_length;
    private final JTextField JTextField_trace_length;
    private final JLabel JLabel_tube_Radius;
    private final JTextField JTextField_tube_Radius;
    private final JTree JTree_InitialBoundary;
    private final JComboBox<Object> JComboBox_DataChooser;
    private final JLabel JLabel_Tdirection;
    private final ButtonGroup BGroup;
    private final JRadioButton JRadioButton_back;
    private final JRadioButton JRadioButton_forward;
    private final JRadioButton JRadioButton_both;

    public JPanel_StreamlineCard() {
        super();
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


        //tDirection


        JLabel_Tdirection = new JLabel("Direction");
        GBCon = GetGridBagConstraints.get(0, 1, 0, 0, new Insets(
                5, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.LINE_START);
        GBCon.gridwidth = 1;
        this.add(JLabel_Tdirection, GBCon);

        JPanel TDirection = new JPanel();
        TDirection.setLayout(new GridBagLayout());
        TDirection.setBackground(Color.white);
        JRadioButton_forward = new JRadioButton("forward");
        JRadioButton_forward.setBackground(Color.white);
        JRadioButton_forward.setActionCommand(Configuration_Tool.forward);


        JRadioButton_back = new JRadioButton("back");
        JRadioButton_back.setBackground(Color.white);
        JRadioButton_back.setActionCommand(Configuration_Tool.back);

        JRadioButton_both = new JRadioButton("both");
        JRadioButton_both.setBackground(Color.white);
        JRadioButton_both.setActionCommand(Configuration_Tool.both);

        BGroup = new ButtonGroup();
        BGroup.add(JRadioButton_forward);
        BGroup.add(JRadioButton_back);
        BGroup.add(JRadioButton_both);

        TDirection.add(JRadioButton_forward);
        TDirection.add(JRadioButton_back);
        TDirection.add(JRadioButton_both);
        JRadioButton_forward.setSelected(true);
        GBCon = GetGridBagConstraints.get(1, 1, 0, 0, new Insets(5, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.LINE_START);
        GBCon.gridheight = 1;
        GBCon.gridwidth = GridBagConstraints.REMAINDER;
        this.add(TDirection, GBCon);

        //Line Divisions
        JLabel_Divisions = new JLabel("Divisions");
        GBCon = GetGridBagConstraints.get(0, 2, 0, 0, new Insets(
                5, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.LINE_START);
        GBCon.gridwidth = 1;
        this.add(JLabel_Divisions, GBCon);

        JPanel JPanel_Division = new JPanel();

        JTextField_DivisionX = new JTextField("1");
        JTextField_DivisionX.setPreferredSize(Dimension_inputField);
        JPanel_Division.add(JTextField_DivisionX, GBCon);

        JTextField_DivisionY = new JTextField("1");
        JTextField_DivisionY.setPreferredSize(Dimension_inputField);
        JPanel_Division.add(JTextField_DivisionY, GBCon);

        JTextField_DivisionZ = new JTextField("1");
        JTextField_DivisionZ.setPreferredSize(Dimension_inputField);
        JPanel_Division.add(JTextField_DivisionZ, GBCon);

        GBCon = GetGridBagConstraints.get(1, 2, 0, 0, new Insets(0, 0, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.LINE_START);
        GBCon.gridwidth = GridBagConstraints.REMAINDER;
        JPanel_Division.setBackground(Color.white);
        this.add(JPanel_Division, GBCon);

        //Tube Radius
        JLabel_tube_Radius = new JLabel("Tube Radius");
        GBCon = GetGridBagConstraints.get(0, 3, 0, 0, new Insets(
                5, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.LINE_START);
        GBCon.gridwidth = 1;
        this.add(JLabel_tube_Radius, GBCon);

        JTextField_tube_Radius = new JTextField("0.1");
        GBCon = GetGridBagConstraints.get(1, 3, 0, 0, new Insets(5, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.LINE_START);
        JTextField_tube_Radius.setPreferredSize(Dimension_inputField);
        this.add(JTextField_tube_Radius, GBCon);

        //Tracer_length
        JLabel_trace_length = new JLabel("Trace Length");
        GBCon = GetGridBagConstraints.get(0, 4, 0, 0, new Insets(
                5, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.LINE_START);
        GBCon.gridwidth = 1;
        this.add(JLabel_trace_length, GBCon);

        JTextField_trace_length = new JTextField("1");
        GBCon = GetGridBagConstraints.get(1, 4, 0, 0, new Insets(5, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.LINE_START);
        JTextField_trace_length.setPreferredSize(Dimension_inputField);
        this.add(JTextField_trace_length, GBCon);

        //Max out of domain
        JLabel_max_out_of_Domain = new JLabel("max out of domain");
        GBCon = GetGridBagConstraints.get(0, 5, 0, 0, new Insets(
                5, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.LINE_START);
        GBCon.gridwidth = 1;
        this.add(JLabel_max_out_of_Domain, GBCon);

        JTextField_max_out_of_domain = new JTextField("0.2");
        GBCon = GetGridBagConstraints.get(1, 5, 0, 0, new Insets(5, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.LINE_START);
        JTextField_max_out_of_domain.setPreferredSize(Dimension_inputField);
        this.add(JTextField_max_out_of_domain, GBCon);

        //Chosen Gemtery Tree
        //Headline
        JLabel Headline = new JLabel("Choose Regions");

        GBCon = GetGridBagConstraints.get(0, 6, 0, 0, new Insets(5, 5, 0, 0),
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

        GBCon = GetGridBagConstraints.get(0, 7, 1, 1, new Insets(0, 5, 0, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = GridBagConstraints.REMAINDER;
        GBCon.gridwidth = 3;

        this.add(GeometryTreeScrollPane, GBCon);

        //JTree Geometry Visualisation
        //Headline
        JScrollPane GeometryVisualisationTreeScrollPane = new JScrollPane();
        JLabel Headline2 = new JLabel("Regions chosen");

        GBCon = GetGridBagConstraints.get(3, 6, 0, 0, new Insets(5, 5, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = 1;
        GBCon.gridwidth = 1;

        this.add(Headline2, GBCon);

        //Baumeinstellungen
        JTree_GeometrytoVisualize = new JTree();
        JTree_GeometrytoVisualize.setShowsRootHandles(true); //

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

        GBCon = GetGridBagConstraints.get(3, 7, 1, 1, new Insets(0, 5, 0, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = GridBagConstraints.REMAINDER;
        GBCon.gridwidth = 2;

        this.add(GeometryVisualisationTreeScrollPane, GBCon);

        //JTree Start Surface (Initial Boundary)
        //Headline
        JScrollPane InitialSurfaceTreeScrollPane = new JScrollPane();
        JLabel Headline3 = new JLabel("Choose Initial Surface");

        GBCon = GetGridBagConstraints.get(5, 6, 0, 0, new Insets(5, 20, 0, 0),
                GridBagConstraints.NONE,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = 1;
        GBCon.gridwidth = 1;

        this.add(Headline3, GBCon);

        //Baumeinstellungen
        JTree_InitialBoundary = new JTree();
        JTree_InitialBoundary.setShowsRootHandles(true); //
     

        //Tree Renderer einstellen
        DefaultTreeCellRenderer Renderer3 = (DefaultTreeCellRenderer) JTree_InitialBoundary.getCellRenderer();
        Renderer3.setLeafIcon(null);//keine Icons
        Renderer3.setClosedIcon(null);
        Renderer3.setOpenIcon(null);
        Renderer3.setBackground(Color.white);//Hintergurnd
        JTree_InitialBoundary.setRootVisible(false);//rootknoten ausblenden
        JTree_InitialBoundary.setVisibleRowCount(JTree_InitialBoundary.getRowCount()); //sichtbarkeit Dynamisch
        InitialSurfaceTreeScrollPane.add(JTree_InitialBoundary);
        InitialSurfaceTreeScrollPane.setViewportView(JTree_InitialBoundary);

        GBCon = GetGridBagConstraints.get(5, 7, 1, 1, new Insets(0, 20, 0, 0),
                GridBagConstraints.BOTH,
                GridBagConstraints.FIRST_LINE_START);
        GBCon.gridheight = GridBagConstraints.REMAINDER;
        GBCon.gridwidth = GridBagConstraints.REMAINDER;

        this.add(InitialSurfaceTreeScrollPane, GBCon);

    }

    public JTree getJTree_ChosenGeometry() {
        return JTree_ChosenGeometry;
    }

    public JTree getJTree_GeometrytoVisualize() {
        return JTree_GeometrytoVisualize;
    }

    public JTree getJTree_InitialBoundary() {
        return JTree_InitialBoundary;
    }

    public JTextField getJTextField_DivisionX() {
        return JTextField_DivisionX;
    }

    public JTextField getJTextField_DivisionY() {
        return JTextField_DivisionY;
    }

    public JTextField getJTextField_DivisionZ() {
        return JTextField_DivisionZ;
    }

    public JTextField getJTextField_max_out_of_domain() {
        return JTextField_max_out_of_domain;
    }

    public JTextField getJTextField_trace_length() {
        return JTextField_trace_length;
    }

    public JTextField getJTextField_Tube_Radius() {
        return JTextField_tube_Radius;
    }

    public JComboBox<Object> getJComboBox_DataChooser() {
        return JComboBox_DataChooser;
    }

    public ButtonGroup getBGroup() {
        return BGroup;
    }

    public JRadioButton getJRadioButton_back() {
        return JRadioButton_back;
    }

    public JRadioButton getJRadioButton_both() {
        return JRadioButton_both;
    }

    public JRadioButton getJRadioButton_forward() {
        return JRadioButton_forward;
    }
}
