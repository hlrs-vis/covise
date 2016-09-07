package de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards.controller;

import Main.PluginContainer;
import de.hlrs.starplugin.configuration.Configuration_GUI_Strings;
import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_CuttingSurfaceSeries;
import de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards.JPanel_CuttingSurfaceSeriesCard;
import de.hlrs.starplugin.gui.dialogs.Error_Dialog;
import de.hlrs.starplugin.util.FieldFunctionplusType;
import de.hlrs.starplugin.util.Vec;
import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import javax.swing.JComboBox;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.event.TreeSelectionEvent;
import javax.swing.event.TreeSelectionListener;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreePath;
import javax.swing.tree.TreeSelectionModel;
import star.common.FieldFunction;
import star.common.Region;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Controller_CuttingSurfaceSeriesCard {

    private PluginContainer PC;
    private JPanel_CuttingSurfaceSeriesCard JP_CSS;
    private boolean enabled = true;

    public Controller_CuttingSurfaceSeriesCard(PluginContainer PC) {
        this.PC = PC;

        this.JP_CSS = PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                getJPanel_CardLayout_TypeSettings().getCuttingSurfaceSeriesCard();
        JP_CSS.getJComboBox_DataTypeChooser().addActionListener(JComboBox_DataType_Listener());
        JP_CSS.getJComboBox_DataChooser().addActionListener(JComboBox_DataChooser_Listener());

        JP_CSS.getJTree_ChosenGeometry().getSelectionModel().addTreeSelectionListener(
                JTree_GeometryCard_ChosenGeometry_Listsner());
        ActionListener BuGL = RadioButtons_Direction_Listener();
        JP_CSS.getJRadioButton_X_Direction().addActionListener(BuGL);
        JP_CSS.getJRadioButton_Y_Direction().addActionListener(BuGL);
        JP_CSS.getJRadioButton_Z_Direction().addActionListener(BuGL);

        JP_CSS.getJTextField_Distance().getDocument().addDocumentListener(JTextField_Distance_Listener());
        JP_CSS.getJTextField_CuttingDistance().getDocument().addDocumentListener(
                JTextField_CuttingDistance_Listener());
        JP_CSS.getJTextField_NumberOfCuts().getDocument().addDocumentListener(
                JTextField_NumberofCuts_Listener());

    }

    private TreeSelectionListener JTree_GeometryCard_ChosenGeometry_Listsner() {
        TreeSelectionListener TSL = new TreeSelectionListener() {

            public void valueChanged(TreeSelectionEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_CuttingSurfaceSeries) {

                        TreeSelectionModel tsm = (TreeSelectionModel) e.getSource();

                        TreePath[] SelectedPaths = tsm.getSelectionPaths();
                        ArrayList<DefaultMutableTreeNode> tmpSelectedNodesList = new ArrayList<DefaultMutableTreeNode>();

                        for (TreePath t : SelectedPaths) {
                            DefaultMutableTreeNode tmpNode = (DefaultMutableTreeNode) t.getLastPathComponent();
                            if (tmpNode.getUserObject() instanceof Region) {
                                tmpSelectedNodesList.add(new DefaultMutableTreeNode(tmpNode.getUserObject()));
                            }
                        }
                        PC.getCNGVMC().CuttingSurfaceSeriesCard_GeometryToVisualizeSelectionChanged(
                                tmpSelectedNodesList);
                        changePartsofConstruct();
                    }
                }
            }
        };
        return TSL;
    }

    private ActionListener JComboBox_DataType_Listener() {
        ActionListener AL = new ActionListener() {

            public void actionPerformed(ActionEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_CuttingSurfaceSeries) {
                        JComboBox cb = (JComboBox) e.getSource();

                        if (Configuration_Tool.DataType_scalar.equals(cb.getSelectedItem())) {

                            JP_CSS.getJComboBox_DataChooser().setEnabled(true);
                            JP_CSS.getJComboBox_DataChooser().setModel(PC.getCNGVMC().getComboBoxModel_Scalar());
                            enabled = false;
                            JP_CSS.getJComboBox_DataChooser().setSelectedIndex(-1);
                            enabled = true;
                        }
                        if (Configuration_Tool.DataType_vector.equals(cb.getSelectedItem())) {

                            JP_CSS.getJComboBox_DataChooser().setEnabled(true);
                            JP_CSS.getJComboBox_DataChooser().setModel(PC.getCNGVMC().getComboBoxModel_Vector());
                            enabled = false;
                            JP_CSS.getJComboBox_DataChooser().setSelectedIndex(-1);
                            enabled = true;
                        }
                    }
                }
            }
        };
        return AL;

    }

    private ActionListener JComboBox_DataChooser_Listener() {
        ActionListener AL = new ActionListener() {

            public void actionPerformed(ActionEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_CuttingSurfaceSeries) {
                        changeFieldFunctionofConstruct();
                    }
                }
            }
        };
        return AL;

    }

    private ActionListener RadioButtons_Direction_Listener() {
        ActionListener AL = new ActionListener() {

            public void actionPerformed(ActionEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_CuttingSurfaceSeries) {
                        ChangeDirectionOfConstruct();
                    }
                }
            }
        };
        return AL;
    }

    private DocumentListener JTextField_Distance_Listener() {
        DocumentListener DL = new DocumentListener() {

            public void insertUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_CuttingSurfaceSeries) {
                        ChangeDistanceofConstruct();
                    }
                }
            }

            public void removeUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_CuttingSurfaceSeries) {

                        ChangeDistanceofConstruct();
                    }
                }
            }

            public void changedUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_CuttingSurfaceSeries) {
                        ChangeDistanceofConstruct();
                    }
                }
            }
        };
        return DL;
    }

    private DocumentListener JTextField_CuttingDistance_Listener() {
        DocumentListener DL = new DocumentListener() {

            public void insertUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_CuttingSurfaceSeries) {
                        ChangeCuttingDistanceofConstruct();
                    }
                }
            }

            public void removeUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_CuttingSurfaceSeries) {
                        ChangeCuttingDistanceofConstruct();
                    }
                }
            }

            public void changedUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_CuttingSurfaceSeries) {
                        ChangeCuttingDistanceofConstruct();
                    }
                }
            }
        };
        return DL;
    }

    private DocumentListener JTextField_NumberofCuts_Listener() {
        DocumentListener DL = new DocumentListener() {

            public void insertUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_CuttingSurfaceSeries) {
                        ChangeNumberofCutsofConstruct();
                    }
                }
            }

            public void removeUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_CuttingSurfaceSeries) {

                        ChangeNumberofCutsofConstruct();
                    }
                }
            }

            public void changedUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_CuttingSurfaceSeries) {

                        ChangeNumberofCutsofConstruct();
                    }
                }
            }
        };
        return DL;
    }

    private void changeFieldFunctionofConstruct() {
        Construct_CuttingSurfaceSeries Con = null;
        try {
            Con = (Construct_CuttingSurfaceSeries) PC.getCNGDMan().getConMan().
                    getSelectedConstruct();
        } catch (Exception Ex) {
            StringWriter sw = new StringWriter();
            Ex.printStackTrace(new PrintWriter(sw));
            new Error_Dialog(Configuration_GUI_Strings.Occourence + Configuration_GUI_Strings.eol
                    + Configuration_GUI_Strings.ErrMass + Ex.getMessage() + Configuration_GUI_Strings.eol
                    + Configuration_GUI_Strings.StackTrace + sw.toString());
        }

        if (Con != null) {
            FieldFunctionplusType FFpT;
            String Type = (String) JP_CSS.getJComboBox_DataTypeChooser().getSelectedItem();
            if (!Type.equals(Configuration_Tool.DataType_none)) {
                FieldFunction FF = (FieldFunction) JP_CSS.getJComboBox_DataChooser().getSelectedItem();
                FFpT = new FieldFunctionplusType(FF, Type);
                Con.setFFplType(FFpT);
            } else {
                Con.setFFplType(null);
            }
        } else {
            PC.getSim().println("noConstruct1");
        }

    }

    private void changePartsofConstruct() {
        Construct_CuttingSurfaceSeries Con = (Construct_CuttingSurfaceSeries) PC.getCNGDMan().getConMan().
                getSelectedConstruct();
        Con.getParts().clear();
        try {
            for (DefaultMutableTreeNode n : PC.getCNGVMC().
                    getList_CuttingSurfaceSeriesCard_SelectedGeometryNodes()) {
                Con.addPart(n.getUserObject(), PC.getEEMan().getPartsList().get(n.getUserObject()));
            }
        } catch (Exception Ex) {
            StringWriter sw = new StringWriter();
            Ex.printStackTrace(new PrintWriter(sw));
            new Error_Dialog(Configuration_GUI_Strings.Occourence + Configuration_GUI_Strings.eol
                    + Configuration_GUI_Strings.ErrMass + Ex.getMessage() + Configuration_GUI_Strings.eol
                    + Configuration_GUI_Strings.StackTrace + sw.toString());


        }
    }

    private void ChangeDirectionOfConstruct() {
        Construct_CuttingSurfaceSeries Con = (Construct_CuttingSurfaceSeries) PC.getCNGDMan().getConMan().
                getSelectedConstruct();
        //Direction
        String Selection = JP_CSS.getBGroup().getSelection().
                getActionCommand();


        if (Selection.equals(Configuration_Tool.RadioButtonActionCommand_X_Direction)) {
            Con.setVertex(new Vec(1, 0, 0));
            Con.setDirection(Configuration_Tool.RadioButtonActionCommand_X_Direction);


        }
        if (Selection.equals(Configuration_Tool.RadioButtonActionCommand_Y_Direction)) {
            Con.setVertex(new Vec(0, 1, 0));
            Con.setDirection(Configuration_Tool.RadioButtonActionCommand_Y_Direction);


        }
        if (Selection.equals(Configuration_Tool.RadioButtonActionCommand_Z_Direction)) {
            Con.setVertex(new Vec(0, 0, 1));
            Con.setDirection(Configuration_Tool.RadioButtonActionCommand_Z_Direction);

        }
    }

    private void ChangeDistanceofConstruct() {
        Construct_CuttingSurfaceSeries Con = (Construct_CuttingSurfaceSeries) PC.getCNGDMan().getConMan().
                getSelectedConstruct();
        String Selection = JP_CSS.getBGroup().getSelection().getActionCommand();

        //Distance
        try {
            String Distance = JP_CSS.getJTextField_Distance().
                    getText();
            float f = Float.parseFloat(Distance);
            Con.setDistance(f);
            if (Selection.equals(Configuration_Tool.RadioButtonActionCommand_X_Direction)) {
                Con.setPoint(new Vec(f, 0, 0));
            }
            if (Selection.equals(Configuration_Tool.RadioButtonActionCommand_Y_Direction)) {
                Con.setPoint(new Vec(0, f, 0));
            }
            if (Selection.equals(Configuration_Tool.RadioButtonActionCommand_Z_Direction)) {
                Con.setPoint(new Vec(0, 0, f));
            }
            JP_CSS.getJTextField_Distance().setForeground(Color.BLACK);
        } catch (Exception c) {
            JP_CSS.getJTextField_Distance().setForeground(Color.red);

        }
    }

    private void ChangeNumberofCutsofConstruct() {
        Construct_CuttingSurfaceSeries Con = (Construct_CuttingSurfaceSeries) PC.getCNGDMan().getConMan().
                getSelectedConstruct();


        //Number of Cuts
        try {
            String NumberofCuts = JP_CSS.getJTextField_NumberOfCuts().getText();
            int i = Integer.parseInt(NumberofCuts);
            if (i >= 1) {
                Con.setAmount(i);
            } else {
                throw new Exception("Amount to small");
            }
            JP_CSS.getJTextField_NumberOfCuts().setForeground(Color.BLACK);
        } catch (Exception c) {
            JP_CSS.getJTextField_NumberOfCuts().setForeground(Color.red);

        }
    }

    private void ChangeCuttingDistanceofConstruct() {
        Construct_CuttingSurfaceSeries Con = (Construct_CuttingSurfaceSeries) PC.getCNGDMan().getConMan().
                getSelectedConstruct();

        //Cutting Distance
        try {
            String Distance = JP_CSS.getJTextField_CuttingDistance().getText();
            float f = Float.parseFloat(Distance);
            Con.setDistanceBetween(f);

            JP_CSS.getJTextField_CuttingDistance().setForeground(Color.BLACK);
        } catch (Exception c) {
            JP_CSS.getJTextField_CuttingDistance().setForeground(Color.red);
        }
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }
}

