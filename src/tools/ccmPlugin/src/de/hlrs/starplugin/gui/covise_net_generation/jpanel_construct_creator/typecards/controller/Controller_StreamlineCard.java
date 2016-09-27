package de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards.controller;

import Main.PluginContainer;
import de.hlrs.starplugin.configuration.Configuration_GUI_Strings;
import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_Streamline;
import de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards.JPanel_StreamlineCard;
import de.hlrs.starplugin.gui.dialogs.Error_Dialog;
import de.hlrs.starplugin.util.FieldFunctionplusType;
import de.hlrs.starplugin.util.Vec;
import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
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
public class Controller_StreamlineCard {

    private PluginContainer PC;
    private JPanel_StreamlineCard JP_SL;
    private boolean enabled = true;

    public Controller_StreamlineCard(PluginContainer PC) {
        this.PC = PC;

        this.JP_SL = PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                getJPanel_CardLayout_TypeSettings().getStreamlineCard();

        JP_SL.getJTree_ChosenGeometry().getSelectionModel().addTreeSelectionListener(
                JTree_StreamlineCard_ChosenGeometry_Listsner());
        JP_SL.getJTextField_DivisionX().getDocument().addDocumentListener(JTextField_DivisionX_Listener());
        JP_SL.getJTextField_DivisionY().getDocument().addDocumentListener(JTextField_DivisionY_Listener());
        JP_SL.getJTextField_DivisionZ().getDocument().addDocumentListener(JTextField_DivisionZ_Listener());

        JP_SL.getJTextField_Tube_Radius().getDocument().addDocumentListener(JTextField_TubeRadius_Listener());
        JP_SL.getJTextField_trace_length().getDocument().addDocumentListener(JTextField_TraceLength_Listener());
        JP_SL.getJTextField_max_out_of_domain().getDocument().addDocumentListener(
                JTextField_MaxoutofDomain_Listener());

        JP_SL.getJTree_InitialBoundary().getSelectionModel().addTreeSelectionListener(
                JTree_StreamlineCard_initialSurface_Listener());
        JP_SL.getJComboBox_DataChooser().addActionListener(JComboBox_DataChooser_Listener());

        ActionListener BListener = RadioButtons_Direction_Listener();
        JP_SL.getJRadioButton_forward().addActionListener(BListener);
        JP_SL.getJRadioButton_back().addActionListener(BListener);
        JP_SL.getJRadioButton_both().addActionListener(BListener);




    }

    private TreeSelectionListener JTree_StreamlineCard_ChosenGeometry_Listsner() {
        TreeSelectionListener TSL = new TreeSelectionListener() {

            public void valueChanged(TreeSelectionEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {


                        TreeSelectionModel tsm = (TreeSelectionModel) e.getSource();

                        TreePath[] SelectedPaths = tsm.getSelectionPaths();
                        ArrayList<DefaultMutableTreeNode> tmpSelectedNodesList = new ArrayList<DefaultMutableTreeNode>();

                        for (TreePath t : SelectedPaths) {
                            DefaultMutableTreeNode tmpNode = (DefaultMutableTreeNode) t.getLastPathComponent();
                            if (tmpNode.getUserObject() instanceof Region) {
                                tmpSelectedNodesList.add(new DefaultMutableTreeNode(tmpNode.getUserObject()));
                            }
                        }
                        PC.getCNGVMC().StreamlineCard_GeometryToVisualizeSelectionChanged(
                                tmpSelectedNodesList);
                        changePartsofConstruct();
                    }
                }
            }
        };
        return TSL;
    }

    private TreeSelectionListener JTree_StreamlineCard_initialSurface_Listener() {
        TreeSelectionListener TSL = new TreeSelectionListener() {

            public void valueChanged(TreeSelectionEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {

                        TreeSelectionModel tsm = (TreeSelectionModel) e.getSource();

                        TreePath[] SelectedPaths = tsm.getSelectionPaths();
                        ArrayList<Boundary> tmpSelectedNodesList = new ArrayList<Boundary>();
                        Construct Con = PC.getCNGDMan().getConMan().getSelectedConstruct();

                        for (TreePath t : SelectedPaths) {
                            DefaultMutableTreeNode tmpNode = (DefaultMutableTreeNode) t.getLastPathComponent();
                            if (tmpNode.getUserObject() instanceof Boundary) {
                                tmpSelectedNodesList.add((Boundary) tmpNode.getUserObject());
                            }
                        }
                        changeInitialSurfaceofConstruct(tmpSelectedNodesList);
                    }
                }
            }
        };
        return TSL;
    }

    private DocumentListener JTextField_DivisionX_Listener() {
        DocumentListener DL = new DocumentListener() {

            public void insertUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {
                        ChangeDivisionXofConstruct();
                    }
                }
            }

            public void removeUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {
                        ChangeDivisionXofConstruct();
                    }
                }
            }

            public void changedUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {
                        ChangeDivisionXofConstruct();
                    }
                }
            }
        };
        return DL;
    }

    private DocumentListener JTextField_DivisionY_Listener() {
        DocumentListener DL = new DocumentListener() {

            public void insertUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {
                        ChangeDivisionYofConstruct();
                    }
                }
            }

            public void removeUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {
                        ChangeDivisionYofConstruct();
                    }
                }
            }

            public void changedUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {
                        ChangeDivisionYofConstruct();
                    }
                }
            }
        };
        return DL;
    }

    private DocumentListener JTextField_DivisionZ_Listener() {
        DocumentListener DL = new DocumentListener() {

            public void insertUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {
                        ChangeDivisionZofConstruct();
                    }
                }
            }

            public void removeUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {
                        ChangeDivisionZofConstruct();
                    }
                }
            }

            public void changedUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {
                        ChangeDivisionZofConstruct();
                    }
                }
            }
        };
        return DL;
    }

    private DocumentListener JTextField_TubeRadius_Listener() {
        DocumentListener DL = new DocumentListener() {

            public void insertUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {
                        ChangeTubeRadiusofConstruct();
                    }
                }
            }

            public void removeUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {
                        ChangeTubeRadiusofConstruct();
                    }
                }
            }

            public void changedUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {
                        ChangeTubeRadiusofConstruct();
                    }
                }
            }
        };
        return DL;
    }

    private DocumentListener JTextField_TraceLength_Listener() {
        DocumentListener DL = new DocumentListener() {

            public void insertUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {
                        ChangeTraceLengthofConstruct();
                    }
                }
            }

            public void removeUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {
                        ChangeTraceLengthofConstruct();
                    }
                }
            }

            public void changedUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {
                        ChangeTraceLengthofConstruct();
                    }
                }
            }
        };
        return DL;
    }

    private DocumentListener JTextField_MaxoutofDomain_Listener() {
        DocumentListener DL = new DocumentListener() {

            public void insertUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {
                        ChangeMaxoutofDomainofConstruct();
                    }
                }
            }

            public void removeUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {
                        ChangeMaxoutofDomainofConstruct();
                    }
                }
            }

            public void changedUpdate(DocumentEvent e) {
                if (enabled) {
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {
                        ChangeMaxoutofDomainofConstruct();
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
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {
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
                    if (PC.getCNGDMan().getConMan().getSelectedConstruct() instanceof Construct_Streamline) {
                        ChangeDirectionOfConstruct();
                    }
                }
            }
        };
        return AL;
    }

    private void changeFieldFunctionofConstruct() {
        Construct_Streamline Con = (Construct_Streamline) PC.getCNGDMan().getConMan().
                getSelectedConstruct();

        if (Con != null) {
            FieldFunctionplusType FFpT;
            String Type = (String) Configuration_Tool.DataType_vector;
            FieldFunction FF = (FieldFunction) JP_SL.getJComboBox_DataChooser().getSelectedItem();
            if (!PC.getEEDMan().getVectorsSelectedList().contains(FF)) {
                PC.getEEDMan().addVectorFieldFunctiontoSelection(FF);
                PC.getEEDMan().dataChanged(Configuration_Tool.onChange_Vector);

            }
            FFpT = new FieldFunctionplusType(FF, Type);
            Con.setFFplType(FFpT);


        } else {
            PC.getSim().println("noConstruct2");
        }

    }

    private void changeInitialSurfaceofConstruct(ArrayList<Boundary> AL) {
        Construct_Streamline Con = (Construct_Streamline) PC.getCNGDMan().getConMan().
                getSelectedConstruct();
        Con.getInitialSurface().clear();
        try {
            for (Boundary B : AL) {
                Con.addInitialSurface(B, PC.getEEMan().getPartsList().get(B));
            }
        } catch (Exception Ex) {
            StringWriter sw = new StringWriter();
            Ex.printStackTrace(new PrintWriter(sw));
            new Error_Dialog(Configuration_GUI_Strings.Occourence + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.ErrMass + Ex.getMessage() + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.StackTrace + sw.toString());

        }

    }

    private void changePartsofConstruct() {
        Construct_Streamline Con = (Construct_Streamline) PC.getCNGDMan().getConMan().
                getSelectedConstruct();
        Con.getParts().clear();
        try {
            for (DefaultMutableTreeNode n : PC.getCNGVMC().
                    getList_StreamlineCard_SelectedGeometryNodes()) {
                Con.addPart(n.getUserObject(), PC.getEEMan().getPartsList().get(n.getUserObject()));
            }
        } catch (Exception Ex) {
            StringWriter sw = new StringWriter();
            Ex.printStackTrace(new PrintWriter(sw));
            new Error_Dialog(Configuration_GUI_Strings.Occourence + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.ErrMass + Ex.getMessage() + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.StackTrace + sw.toString());

        }

    }

    private void ChangeDivisionXofConstruct() {
        Construct_Streamline Con = (Construct_Streamline) PC.getCNGDMan().getConMan().
                getSelectedConstruct();


        //DivisionX
        try {
            String DivisionX = JP_SL.getJTextField_DivisionX().getText();
            float f = Float.parseFloat(DivisionX);
            Vec Division = Con.getDivisions();
            Division.x = f;

            JP_SL.getJTextField_DivisionX().setForeground(Color.BLACK);
        } catch (Exception c) {
            JP_SL.getJTextField_DivisionX().setForeground(Color.red);

        }
    }

    private void ChangeDivisionYofConstruct() {
        Construct_Streamline Con = (Construct_Streamline) PC.getCNGDMan().getConMan().
                getSelectedConstruct();


        //Divisiony
        try {
            String DivisionY = JP_SL.getJTextField_DivisionY().getText();
            float f = Float.parseFloat(DivisionY);
            Vec Division = Con.getDivisions();
            Division.y = f;

            JP_SL.getJTextField_DivisionY().setForeground(Color.BLACK);
        } catch (Exception c) {
            JP_SL.getJTextField_DivisionY().setForeground(Color.red);

        }
    }

    private void ChangeDivisionZofConstruct() {
        Construct_Streamline Con = (Construct_Streamline) PC.getCNGDMan().getConMan().
                getSelectedConstruct();


        //DivisionZ
        try {
            String DivisionZ = JP_SL.getJTextField_DivisionZ().getText();
            float f = Float.parseFloat(DivisionZ);
            Vec Division = Con.getDivisions();
            Division.z = f;

            JP_SL.getJTextField_DivisionZ().setForeground(Color.BLACK);
        } catch (Exception c) {
            JP_SL.getJTextField_DivisionZ().setForeground(Color.red);

        }
    }

    private void ChangeTubeRadiusofConstruct() {
        Construct_Streamline Con = (Construct_Streamline) PC.getCNGDMan().getConMan().
                getSelectedConstruct();


        //DivisionZ
        try {
            String TubeRadius = JP_SL.getJTextField_Tube_Radius().getText();
            float f = Float.parseFloat(TubeRadius);
            Con.setTube_Radius(f);


            JP_SL.getJTextField_Tube_Radius().setForeground(Color.BLACK);
        } catch (Exception c) {
            JP_SL.getJTextField_Tube_Radius().setForeground(Color.red);

        }
    }

    private void ChangeTraceLengthofConstruct() {
        Construct_Streamline Con = (Construct_Streamline) PC.getCNGDMan().getConMan().
                getSelectedConstruct();


        //DivisionZ
        try {
            String TraceLength = JP_SL.getJTextField_trace_length().getText();
            float f = Float.parseFloat(TraceLength);
            Con.setTrace_length(f);


            JP_SL.getJTextField_trace_length().setForeground(Color.BLACK);
        } catch (Exception c) {
            JP_SL.getJTextField_trace_length().setForeground(Color.red);

        }
    }

    private void ChangeMaxoutofDomainofConstruct() {
        Construct_Streamline Con = (Construct_Streamline) PC.getCNGDMan().getConMan().
                getSelectedConstruct();


        //DivisionZ
        try {
            String MaxoutofDomain = JP_SL.getJTextField_max_out_of_domain().getText();
            float f = Float.parseFloat(MaxoutofDomain);
            Con.setMax_out_of_domain(f);


            JP_SL.getJTextField_max_out_of_domain().setForeground(Color.BLACK);
        } catch (Exception c) {
            JP_SL.getJTextField_max_out_of_domain().setForeground(Color.red);

        }
    }

    private void ChangeDirectionOfConstruct() {
        Construct_Streamline Con = (Construct_Streamline) PC.getCNGDMan().getConMan().
                getSelectedConstruct();
        //Direction
        String Selection = JP_SL.getBGroup().getSelection().
                getActionCommand();
        if (Selection.equals(Configuration_Tool.forward)) {
            Con.setDirection(Configuration_Tool.forward);
        }
        if (Selection.equals(Configuration_Tool.back)) {
            Con.setDirection(Configuration_Tool.back);
        }
        if (Selection.equals(Configuration_Tool.both)) {
            Con.setDirection(Configuration_Tool.both);
        }
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }
}
