package de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards.controller;

import Main.PluginContainer;
import de.hlrs.starplugin.configuration.Configuration_GUI_Strings;
import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_CuttingSurface;
import de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards.JPanel_CuttingSurfaceCard;
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
 *  @author Weiss HLRS Stuttgart
 */
public class Controller_CuttingSurfaceCard {

    private PluginContainer PC;
    private JPanel_CuttingSurfaceCard JP_CS;
    private boolean enabled = true;

    public Controller_CuttingSurfaceCard(PluginContainer PC) {
        this.PC = PC;

        this.JP_CS = PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                getJPanel_CardLayout_TypeSettings().getCuttingSurfaceCard();
        JP_CS.getJComboBox_DataTypeChooser().addActionListener(JComboBox_DataType_Listener());
        JP_CS.getJComboBox_DataChooser().addActionListener(JComboBox_DataChooser_Listener());

        JP_CS.getJTree_ChosenGeometry().getSelectionModel().addTreeSelectionListener(
                JTree_GeometryCard_ChosenGeometry_Listsner());
        ActionListener BuGL = RadioButtons_Direction_Listener();
        JP_CS.getJRadioButton_X_Direction().addActionListener(BuGL);
        JP_CS.getJRadioButton_Y_Direction().addActionListener(BuGL);
        JP_CS.getJRadioButton_Z_Direction().addActionListener(BuGL);

        JP_CS.getJTextField_Distance().getDocument().addDocumentListener(JTextField_Distance_Listener());

    }

    private TreeSelectionListener JTree_GeometryCard_ChosenGeometry_Listsner() {
        TreeSelectionListener TSL = new TreeSelectionListener() {

            public void valueChanged(TreeSelectionEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_CuttingSurface) {

                        TreeSelectionModel tsm = (TreeSelectionModel) e.getSource();

                        TreePath[] SelectedPaths = tsm.getSelectionPaths();
                        ArrayList<DefaultMutableTreeNode> tmpSelectedNodesList = new ArrayList<DefaultMutableTreeNode>();

                        for (TreePath t : SelectedPaths) {
                            DefaultMutableTreeNode tmpNode = (DefaultMutableTreeNode) t.getLastPathComponent();
                            if (tmpNode.getUserObject() instanceof Region) {
                                tmpSelectedNodesList.add(new DefaultMutableTreeNode(tmpNode.getUserObject()));
                            }
                        }
                        PC.getCNGVMC().CuttingSurfaceCard_GeometryToVisualizeSelectionChanged(
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
                if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_CuttingSurface) {
                    if (enabled) {

                        JComboBox cb = (JComboBox) e.getSource();


                        if (Configuration_Tool.DataType_scalar.equals(cb.getSelectedItem())) {

                            JP_CS.getJComboBox_DataChooser().setEnabled(true);
                            JP_CS.getJComboBox_DataChooser().setModel(PC.getCNGVMC().getComboBoxModel_Scalar());
                            enabled = false;
                            JP_CS.getJComboBox_DataChooser().setSelectedIndex(-1);
                            enabled = true;
                        }
                        if (Configuration_Tool.DataType_vector.equals(cb.getSelectedItem())) {

                            JP_CS.getJComboBox_DataChooser().setEnabled(true);
                            JP_CS.getJComboBox_DataChooser().setModel(PC.getCNGVMC().getComboBoxModel_Vector());
                            enabled = false;
                            JP_CS.getJComboBox_DataChooser().setSelectedIndex(-1);
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
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_CuttingSurface) {
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
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_CuttingSurface) {
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
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_CuttingSurface) {
                        ChangeDistanceofConstruct();
                    }
                }
            }

            public void removeUpdate(DocumentEvent e) {
                if (enabled) {
                    ChangeDistanceofConstruct();
                }
            }

            public void changedUpdate(DocumentEvent e) {
                if (enabled) {
                    ChangeDistanceofConstruct();
                }
            }
        };
        return DL;
    }

    private void changeFieldFunctionofConstruct() {
        Construct_CuttingSurface Con = null;
        try {
            Con = (Construct_CuttingSurface) PC.getCNGDMan().getConMan().
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
            String Type = (String) JP_CS.getJComboBox_DataTypeChooser().getSelectedItem();
            if (!Type.equals(Configuration_Tool.DataType_none)) {
                FieldFunction FF = (FieldFunction) JP_CS.getJComboBox_DataChooser().getSelectedItem();
                FFpT = new FieldFunctionplusType(FF, Type);
                Con.setFFplType(FFpT);
            } else {
                Con.setFFplType(null);
            }
        } else {
            PC.getSim().println("noConstruct3");
        }

    }

    private void changePartsofConstruct() {
        Construct_CuttingSurface Con = (Construct_CuttingSurface) PC.getCNGDMan().getConMan().
                getSelectedConstruct();
        Con.getParts().clear();
        try {
            for (DefaultMutableTreeNode n : PC.getCNGVMC().getList_CuttingSurfaceCard_SelectedGeometryNodes()) {
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
        Construct_CuttingSurface Con = (Construct_CuttingSurface) PC.getCNGDMan().getConMan().
                getSelectedConstruct();
        //Direction
        String Selection = JP_CS.getBGroup().getSelection().
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
        Construct_CuttingSurface Con = (Construct_CuttingSurface) PC.getCNGDMan().getConMan().
                getSelectedConstruct();
        String Selection = JP_CS.getBGroup().getSelection().
                getActionCommand();

        //Distance
        try {
            String Distance = JP_CS.getJTextField_Distance().
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
            JP_CS.getJTextField_Distance().setForeground(Color.BLACK);

        } catch (Exception c) {
            JP_CS.getJTextField_Distance().setForeground(Color.RED);
        }
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }
}
