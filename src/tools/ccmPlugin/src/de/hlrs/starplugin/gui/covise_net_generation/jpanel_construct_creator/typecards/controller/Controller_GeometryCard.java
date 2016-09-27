package de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards.controller;

import Main.PluginContainer;
import de.hlrs.starplugin.configuration.Configuration_GUI_Strings;
import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_GeometryVisualization;
import de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards.JPanel_GeometryCard;
import de.hlrs.starplugin.gui.dialogs.Error_Dialog;
import de.hlrs.starplugin.util.FieldFunctionplusType;
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
import star.common.Boundary;
import star.common.FieldFunction;
import star.common.Region;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Controller_GeometryCard {

    private PluginContainer PC;
    private JPanel_GeometryCard JP_GC;
    private boolean enabled = true;
    private boolean fromMapper = false;

    public Controller_GeometryCard(PluginContainer PC) {
        this.PC = PC;

        this.JP_GC = PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                getJPanel_CardLayout_TypeSettings().getGeometryCard();
        JP_GC.getJComboBox_DataTypeChooser().addActionListener(JComboBox_DataType_Listener());
        JP_GC.getJComboBox_DataChooser().addActionListener(JComboBox_DataChooser_Listener());

        JP_GC.getJTree_ChosenGeometry().getSelectionModel().addTreeSelectionListener(
                JTree_GeometryCard_ChosenGeometry_Listsner());
        ActionListener BListener = RadioButtons_Color_Listener();
        JP_GC.getJRadioButton_Green().addActionListener(BListener);
        JP_GC.getJRadioButton_Grey().addActionListener(BListener);
        JP_GC.getJRadioButton_Red().addActionListener(BListener);
        JP_GC.getJTextField_Transparency().getDocument().addDocumentListener(
                JTextField_Transparency_Listener());



    }

    private TreeSelectionListener JTree_GeometryCard_ChosenGeometry_Listsner() {
        TreeSelectionListener TSL = new TreeSelectionListener() {

            public void valueChanged(TreeSelectionEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_GeometryVisualization) {

                        TreeSelectionModel tsm = (TreeSelectionModel) e.getSource();

                        TreePath[] SelectedPaths = tsm.getSelectionPaths();
                        ArrayList<DefaultMutableTreeNode> tmpSelectedNodesList = new ArrayList<DefaultMutableTreeNode>();

                        for (TreePath t : SelectedPaths) {
                            DefaultMutableTreeNode tmpNode = (DefaultMutableTreeNode) t.getLastPathComponent();
                            if (tmpNode.getUserObject() instanceof Region || tmpNode.getUserObject() instanceof Boundary) {
                                tmpSelectedNodesList.add(new DefaultMutableTreeNode(tmpNode.getUserObject()));
                            }
                        }
                        PC.getCNGVMC().GeoemtryCard_GeometryToVisualizeSelectionChanged(tmpSelectedNodesList);
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
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_GeometryVisualization) {

                        JComboBox cb = (JComboBox) e.getSource();

                        if (Configuration_Tool.DataType_none.equals(cb.getSelectedItem())) {
                            JP_GC.getJComboBox_DataChooser().setEnabled(false);
                            JP_GC.getJPanel_SurfaceColor().setVisible(true);
                            if (!fromMapper) {
                                changeFieldFunctionofConstruct();
                            }
                        }
                        if (Configuration_Tool.DataType_scalar.equals(cb.getSelectedItem())) {

                            JP_GC.getJComboBox_DataChooser().setEnabled(true);
                            JP_GC.getJComboBox_DataChooser().setModel(PC.getCNGVMC().getComboBoxModel_Scalar());
                            enabled = false;
                            JP_GC.getJComboBox_DataChooser().setSelectedIndex(-1);
                            enabled = true;
                            JP_GC.getJPanel_SurfaceColor().setVisible(false);

                        }
                        if (Configuration_Tool.DataType_vector.equals(cb.getSelectedItem())) {

                            JP_GC.getJComboBox_DataChooser().setEnabled(true);
                            JP_GC.getJComboBox_DataChooser().setModel(PC.getCNGVMC().getComboBoxModel_Vector());
                            enabled = false;
                            JP_GC.getJComboBox_DataChooser().setSelectedIndex(-1);
                            enabled = true;
                            JP_GC.getJPanel_SurfaceColor().setVisible(false);


                        }                     
                    }
                }
            }
        };
        return AL;

    }

    private DocumentListener JTextField_Transparency_Listener() {
        DocumentListener DL = new DocumentListener() {

            public void insertUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_GeometryVisualization) {
                        ChangeTransparencyValueofConstruct();
                    }
                }
            }

            public void removeUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_GeometryVisualization) {
                        ChangeTransparencyValueofConstruct();
                    }
                }
            }

            public void changedUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_GeometryVisualization) {
                        ChangeTransparencyValueofConstruct();
                    }
                }
            }
        };
        return DL;
    }

    private ActionListener JComboBox_DataChooser_Listener() {
        ActionListener AL = new ActionListener() {

            public void actionPerformed(ActionEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_GeometryVisualization) {
                        if (!fromMapper) {
                            changeFieldFunctionofConstruct();
                        }
                    }
                }
            }
        };
        return AL;

    }

    private ActionListener RadioButtons_Color_Listener() {
        ActionListener AL = new ActionListener() {

            public void actionPerformed(ActionEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_GeometryVisualization) {
                        ChangeColorOfConstruct();
                    }
                }
            }
        };
        return AL;
    }

    private void changeFieldFunctionofConstruct() {
        Construct_GeometryVisualization Con = (Construct_GeometryVisualization) PC.getCNGDMan().getConMan().
                getSelectedConstruct();

        if (Con != null) {
            FieldFunctionplusType FFpT;
            String Type = (String) JP_GC.getJComboBox_DataTypeChooser().getSelectedItem();
            if (!Type.equals(Configuration_Tool.DataType_none)) {

                FieldFunction FF = (FieldFunction) JP_GC.getJComboBox_DataChooser().getSelectedItem();
                FFpT = new FieldFunctionplusType(FF, Type);
                Con.setFFplType(FFpT);

            } else {

                Con.setFFplType(null);
            }
        } else {
            PC.getSim().println("noConstruct4");

        }

    }

    private void changePartsofConstruct() {
        Construct_GeometryVisualization Con = (Construct_GeometryVisualization) PC.getCNGDMan().getConMan().
                getSelectedConstruct();
        Con.getParts().clear();
        try {
            for (DefaultMutableTreeNode n : PC.getCNGVMC().getList_GeometryCard_SelectedGeometryNodes()) {
                Con.addPart(n.getUserObject(), PC.getEEMan().getPartsList().get(n.getUserObject()));
            }
        } catch (Exception Ex) {
            StringWriter sw = new StringWriter();
            Ex.printStackTrace(new PrintWriter(sw));
            new Error_Dialog(Configuration_GUI_Strings.Occourence + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.ErrMass + Ex.getMessage() + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.StackTrace + sw.toString());

        }
    }

    private void ChangeTransparencyValueofConstruct() {
        Construct_GeometryVisualization Con = (Construct_GeometryVisualization) PC.getCNGDMan().getConMan().
                getSelectedConstruct();


        //IsoValue
        try {
            String Transparency = JP_GC.getJTextField_Transparency().getText();
            float f = Float.parseFloat(Transparency);
            if (f > 1) {
                throw new Exception();
            }
            if (f < 0) {
                throw new Exception();
            }
            Con.setTransparency(f);


            JP_GC.getJTextField_Transparency().setForeground(Color.BLACK);
        } catch (Exception c) {
            JP_GC.getJTextField_Transparency().setForeground(Color.red);

        }
    }

    private void ChangeColorOfConstruct() {

        Construct_GeometryVisualization Con = (Construct_GeometryVisualization) PC.getCNGDMan().getConMan().
                getSelectedConstruct();
        //Direction
        String Selection = JP_GC.getBGroup().getSelection().
                getActionCommand();
        if (Selection.equals(Configuration_Tool.Color_grey)) {
            Con.setColor(Configuration_Tool.Color_grey);
        }
        if (Selection.equals(Configuration_Tool.Color_green)) {
            Con.setColor(Configuration_Tool.Color_green);
        }
        if (Selection.equals(Configuration_Tool.Color_red)) {
            Con.setColor(Configuration_Tool.Color_red);
        }

    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public void setFromMapper(boolean fromMapper) {
        this.fromMapper = fromMapper;
    }
}
