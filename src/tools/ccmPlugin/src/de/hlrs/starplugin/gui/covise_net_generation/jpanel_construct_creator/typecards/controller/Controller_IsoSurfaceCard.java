package de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards.controller;

import Main.PluginContainer;
import de.hlrs.starplugin.configuration.Configuration_GUI_Strings;
import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_IsoSurface;


import de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards.JPanel_IsoSurfaceCard;
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
import star.common.FieldFunction;
import star.common.Region;

/**
 *
 *@author Weiss HLRS Stuttgart
 */
public class Controller_IsoSurfaceCard {

    private PluginContainer PC;
    private JPanel_IsoSurfaceCard JP_IS;
    private boolean enabled = true;

    public Controller_IsoSurfaceCard(PluginContainer PC) {
        this.PC = PC;

        this.JP_IS = PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                getJPanel_CardLayout_TypeSettings().getIsoSurfaceCard();
        JP_IS.getJComboBox_DataTypeChooser().addActionListener(JComboBox_DataType_Listener());
        JP_IS.getJComboBox_DataChooser().addActionListener(JComboBox_DataChooser_Listener());

        JP_IS.getJTree_ChosenGeometry().getSelectionModel().addTreeSelectionListener(
                JTree_IsoSurfaceCard_ChosenGeometry_Listener());
        JP_IS.getJTextField_IsoValue().getDocument().addDocumentListener(JTextField_MaxoutofDomain_Listener());


    }

    private TreeSelectionListener JTree_IsoSurfaceCard_ChosenGeometry_Listener() {
        TreeSelectionListener TSL = new TreeSelectionListener() {

            public void valueChanged(TreeSelectionEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_IsoSurface) {


                        TreeSelectionModel tsm = (TreeSelectionModel) e.getSource();

                        TreePath[] SelectedPaths = tsm.getSelectionPaths();
                        ArrayList<DefaultMutableTreeNode> tmpSelectedNodesList = new ArrayList<DefaultMutableTreeNode>();

                        for (TreePath t : SelectedPaths) {
                            DefaultMutableTreeNode tmpNode = (DefaultMutableTreeNode) t.getLastPathComponent();
                            if (tmpNode.getUserObject() instanceof Region) {
                                tmpSelectedNodesList.add(new DefaultMutableTreeNode(tmpNode.getUserObject()));
                            }
                        }
                        PC.getCNGVMC().IsoSurfaceCard_GeometryToVisualizeSelectionChanged(tmpSelectedNodesList);
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
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_IsoSurface) {

                        JComboBox cb = (JComboBox) e.getSource();

                        if (Configuration_Tool.DataType_scalar.equals(cb.getSelectedItem())) {

                            JP_IS.getJComboBox_DataChooser().setEnabled(true);
                            JP_IS.getJComboBox_DataChooser().setModel(PC.getCNGVMC().getComboBoxModel_Scalar());
                            enabled = false;
                            JP_IS.getJComboBox_DataChooser().setSelectedIndex(-1);
                            enabled = true;

                        }
                        if (Configuration_Tool.DataType_vector.equals(cb.getSelectedItem())) {

                            JP_IS.getJComboBox_DataChooser().setEnabled(true);
                            JP_IS.getJComboBox_DataChooser().setModel(PC.getCNGVMC().getComboBoxModel_Vector());
                            enabled = false;
                            JP_IS.getJComboBox_DataChooser().setSelectedIndex(-1);
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
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_IsoSurface) {
                        changeFieldFunctionofConstruct();
                    }
                }
            }
        };
        return AL;

    }

    private DocumentListener JTextField_MaxoutofDomain_Listener() {
        DocumentListener DL = new DocumentListener() {

            public void insertUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_IsoSurface) {
                        ChangeIsoValueofConstruct();
                    }
                }
            }

            public void removeUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_IsoSurface) {
                        ChangeIsoValueofConstruct();
                    }
                }
            }

            public void changedUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_IsoSurface) {
                        ChangeIsoValueofConstruct();
                    }
                }
            }
        };
        return DL;
    }

    private void changeFieldFunctionofConstruct() {
        Construct_IsoSurface Con = null;
        try {
            Con = (Construct_IsoSurface) PC.getCNGDMan().getConMan().
                    getSelectedConstruct();
        } catch (Exception Ex) {
            StringWriter sw = new StringWriter();
            Ex.printStackTrace(new PrintWriter(sw));
            new Error_Dialog(Configuration_GUI_Strings.Occourence + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.ErrMass + Ex.getMessage() + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.StackTrace + sw.toString());
        }

        if (Con != null) {
            FieldFunctionplusType FFpT;
            String Type = (String) JP_IS.getJComboBox_DataTypeChooser().getSelectedItem();
            if (!Type.equals(Configuration_Tool.DataType_none)) {

                FieldFunction FF = (FieldFunction) JP_IS.getJComboBox_DataChooser().getSelectedItem();
                FFpT = new FieldFunctionplusType(FF, Type);
                Con.setFFplType(FFpT);

            } else {

                Con.setFFplType(null);
            }
        } else {
            PC.getSim().println("noConstruct5");
        }

    }

    private void changePartsofConstruct() {
        Construct_IsoSurface Con = (Construct_IsoSurface) PC.getCNGDMan().getConMan().
                getSelectedConstruct();
        Con.getParts().clear();
        try {
            for (DefaultMutableTreeNode n : PC.getCNGVMC().
                    getList_IsoSurfaceCard_SelectedGeometryNodes()) {
                Con.addPart(n.getUserObject(), PC.getEEMan().getPartsList().get(n.getUserObject()));
            }
        } catch (Exception Ex) {
            StringWriter sw = new StringWriter();
            Ex.printStackTrace(new PrintWriter(sw));
            new Error_Dialog(Configuration_GUI_Strings.Occourence + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.ErrMass + Ex.getMessage() + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.StackTrace + sw.toString());

        }
    }

    private void ChangeIsoValueofConstruct() {
        Construct_IsoSurface Con = (Construct_IsoSurface) PC.getCNGDMan().getConMan().
                getSelectedConstruct();


        //IsoValue
        try {
            String IsoValue = JP_IS.getJTextField_IsoValue().getText();
            float f = Float.parseFloat(IsoValue);
            Con.setIsoValue(f);


            JP_IS.getJTextField_IsoValue().setForeground(Color.BLACK);
        } catch (Exception c) {
            JP_IS.getJTextField_IsoValue().setForeground(Color.red);

        }
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }
}
