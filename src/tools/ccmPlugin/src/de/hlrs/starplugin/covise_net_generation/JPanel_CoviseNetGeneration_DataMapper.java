package de.hlrs.starplugin.covise_net_generation;

import de.hlrs.starplugin.interfaces.Interface_CoviseNetGeneration_DataChangedListener;
import de.hlrs.starplugin.interfaces.Interface_CoviseNetGeneration_ViewModelChangedListener;
import de.hlrs.starplugin.interfaces.Interface_EnsightExport_DataChangedListener;
import Main.PluginContainer;
import de.hlrs.starplugin.configuration.Configuration_GUI_Strings;
import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_CuttingSurface;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_CuttingSurfaceSeries;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_GeometryVisualization;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_IsoSurface;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_Streamline;
import de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.JPanel_SettingsCardLayout;
import de.hlrs.starplugin.gui.dialogs.Error_Dialog;




import de.hlrs.starplugin.util.FieldFunctionplusType;
import de.hlrs.starplugin.util.JTreeExpansion;
import de.hlrs.starplugin.util.SortJTree;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.HashMap;
import java.util.Map.Entry;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.DefaultTreeModel;
import javax.swing.tree.TreePath;
import javax.swing.tree.TreeSelectionModel;
import star.common.BlowerInletBoundary;
import star.common.BlowerOutletBoundary;
import star.common.Boundary;
import star.common.BoundaryType;
import star.common.InletBoundary;
import star.common.MassFlowBoundary;
import star.common.MixingPlaneInflowBoundary;
import star.common.OutletBoundary;
import star.common.PressureBoundary;
import star.common.Region;
import star.common.StagnationBoundary;

/**
 *Maps the Data from the CoviseNetGeneration_DataManager to the GUI
 * and adds the Swing Data Models to the Swing View of the CoviseNetGeneration
 * @author Weiss HLRS Stuttgart
 */
public class JPanel_CoviseNetGeneration_DataMapper implements Interface_CoviseNetGeneration_ViewModelChangedListener,
        Interface_CoviseNetGeneration_DataChangedListener, Interface_EnsightExport_DataChangedListener {

    private PluginContainer PC;

    public JPanel_CoviseNetGeneration_DataMapper(PluginContainer pC) {
        this.PC = pC;
        PC.getCNGVMC().addListener(this);
        PC.getEEDMan().addListener(this);

        addModelsToViews();

        PC.getCNGDMan().addListener(this);


        Map();
    }

    private void addModelsToViews() {
        JPanel_SettingsCardLayout JP_SCL = PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().
                getjPanelCreator().getJPanel_CardLayout_TypeSettings();
        DefaultTreeModel DTM = PC.getEEVMC().getEnsightExport_ChosenGeometryTree_TreeModel();
        //Geometry Trees beziehen sich auf den Chosen tree im EnsightExport
        JP_SCL.getGeometryCard().getJTree_ChosenGeometry().setModel(DTM);
        JP_SCL.getCuttingSurfaceCard().getJTree_ChosenGeometry().setModel(DTM);
        JP_SCL.getCuttingSurfaceSeriesCard().getJTree_ChosenGeometry().setModel(DTM);
        JP_SCL.getStreamlineCard().getJTree_ChosenGeometry().setModel(DTM);
        JP_SCL.getIsoSurfaceCard().getJTree_ChosenGeometry().setModel(DTM);
        //Geometry to Visualize Trees
        JP_SCL.getGeometryCard().getJTree_GeometrytoVisualize().setModel(
                PC.getCNGVMC().getJTree_GeometryCard_GeometryToVisualize_TreeModel());
        JP_SCL.getCuttingSurfaceCard().getJTree_GeometrytoVisualize().
                setModel(PC.getCNGVMC().getJTree_CuttingSurfaceCard_GeometryToVisualize_TreeModel());
        JP_SCL.getCuttingSurfaceSeriesCard().getJTree_GeometrytoVisualize().
                setModel(PC.getCNGVMC().getJTree_CuttingSurfaceSeriesCard_GeometryToVisualize_TreeModel());
        JP_SCL.getStreamlineCard().getJTree_GeometrytoVisualize().
                setModel(PC.getCNGVMC().getJTree_StreamlineCard_GeometryToVisualize_TreeModel());
        JP_SCL.getIsoSurfaceCard().getJTree_GeometrytoVisualize().
                setModel(PC.getCNGVMC().getJTree_IsoSurfaceCard_GeometryToVisualize_TreeModel());
        JP_SCL.getStreamlineCard().getJTree_InitialBoundary().setModel(
                PC.getCNGVMC().getJTree_StreamlineCard_InitialBoundary_TreeModel());

        //Created Constructs TreeModel
        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanel_CreatedConstructsList().
                getJScrollPane_JTree_CreatedVisualizationConstructs().getJTree_createdConstructsList().
                setModel(PC.getCNGVMC().getJTree_CreatedConstructs_TreeModel());

        //ComboBoxes
        JP_SCL.getGeometryCard().getJComboBox_DataTypeChooser().setModel(
                PC.getCNGVMC().getComboBoxModel_DataType());
        JP_SCL.getStreamlineCard().getJComboBox_DataChooser().setModel(
                PC.getCNGVMC().getComboBoxModel_StreamlineCard_Velocity());


    }

    public void Map() {
        CoviseNetGeneration_ExportPathChanged();
//        Map_GeometryCard_GeometryToVisualizeSelection();
        MapConstructs();
    }

    public void Map_GeometryCard_GeometryToVisualizeSelection() {
        PC.getCNGVMC().getNode_Regions_JTree_GeometryCard_GeometryToVisualize().removeAllChildren();
        PC.getCNGVMC().getNode_Boundaries_JTree_GeometryCard_GeometryToVisualize().removeAllChildren();

        for (DefaultMutableTreeNode n : PC.getCNGVMC().getList_GeometryCard_SelectedGeometryNodes()) {
            if (n.getUserObject() instanceof Region) {
                PC.getCNGVMC().getNode_Regions_JTree_GeometryCard_GeometryToVisualize().add(n);
            }
            if (n.getUserObject() instanceof Boundary) {
                PC.getCNGVMC().getNode_Boundaries_JTree_GeometryCard_GeometryToVisualize().add(n);
            }
        }
        PC.getCNGVMC().getJTree_GeometryCard_GeometryToVisualize_TreeModel().reload();
        JTreeExpansion.expandAllNodes(PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().
                getjPanelCreator().getJPanel_CardLayout_TypeSettings().getGeometryCard().
                getJTree_GeometrytoVisualize());
    }

    public void Map_CuttingSurfaceCard_GeometryToVisualizeSelection() {

        PC.getCNGVMC().getRootnode_JTree_CuttingSurfaceCard_GeometryToVisualize().removeAllChildren();

        for (DefaultMutableTreeNode n : PC.getCNGVMC().getList_CuttingSurfaceCard_SelectedGeometryNodes()) {
            if (n.getUserObject() instanceof Region) {
                PC.getCNGVMC().getRootnode_JTree_CuttingSurfaceCard_GeometryToVisualize().add(n);
            }
        }

        PC.getCNGVMC().getJTree_CuttingSurfaceCard_GeometryToVisualize_TreeModel().reload();
        JTreeExpansion.expandAllNodes(PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().
                getjPanelCreator().getJPanel_CardLayout_TypeSettings().getCuttingSurfaceCard().
                getJTree_GeometrytoVisualize());

    }

    public void Map_CuttingSurfaceSeriesCard_GeometryToVisualizeSelection() {
        PC.getCNGVMC().getRootnode_JTree_CuttingSurfaceSeriesCard_GeometryToVisualize().removeAllChildren();

        for (DefaultMutableTreeNode n : PC.getCNGVMC().getList_CuttingSurfaceSeriesCard_SelectedGeometryNodes()) {
            if (n.getUserObject() instanceof Region) {
                PC.getCNGVMC().getRootnode_JTree_CuttingSurfaceSeriesCard_GeometryToVisualize().add(n);
            }
        }
        PC.getCNGVMC().getJTree_CuttingSurfaceSeriesCard_GeometryToVisualize_TreeModel().nodeStructureChanged(
                PC.getCNGVMC().getRootnode_JTree_CuttingSurfaceSeriesCard_GeometryToVisualize());
        JTreeExpansion.expandAllNodes(PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().
                getjPanelCreator().getJPanel_CardLayout_TypeSettings().getCuttingSurfaceSeriesCard().
                getJTree_GeometrytoVisualize());
    }

    private void Map_StreamlineCard_GeometryToVisualizeSelection() {
        PC.getCNGVMC().getRootnode_JTree_StreamlineCard_GeometryToVisualize().removeAllChildren();
        for (DefaultMutableTreeNode n : PC.getCNGVMC().getList_StreamlineCard_SelectedGeometryNodes()) {
            if (n.getUserObject() instanceof Region) {
                PC.getCNGVMC().getRootnode_JTree_StreamlineCard_GeometryToVisualize().add(n);
            }
        }
        PC.getCNGVMC().getJTree_StreamlineCard_GeometryToVisualize_TreeModel().nodeStructureChanged(
                PC.getCNGVMC().getRootnode_JTree_StreamlineCard_GeometryToVisualize());
        JTreeExpansion.expandAllNodes(PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().
                getjPanelCreator().getJPanel_CardLayout_TypeSettings().getStreamlineCard().
                getJTree_GeometrytoVisualize());

    }

    private void Map_IsoSurfaceCard_GeometryToVisualizeSelection() {
        PC.getCNGVMC().getRootnode_JTree_IsoSurfaceCard_GeometryToVisualize().removeAllChildren();
        for (DefaultMutableTreeNode n : PC.getCNGVMC().getList_IsoSurfaceCard_SelectedGeometryNodes()) {
            if (n.getUserObject() instanceof Region) {
                PC.getCNGVMC().getRootnode_JTree_IsoSurfaceCard_GeometryToVisualize().add(n);
            }
        }
        PC.getCNGVMC().getJTree_IsoSurfaceCard_GeometryToVisualize_TreeModel().nodeStructureChanged(
                PC.getCNGVMC().getRootnode_JTree_IsoSurfaceCard_GeometryToVisualize());

        JTreeExpansion.expandAllNodes(PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().
                getjPanelCreator().getJPanel_CardLayout_TypeSettings().getIsoSurfaceCard().
                getJTree_GeometrytoVisualize());
    }

    private void Map_StreamlineCard_InitialSurfaceChoice() {
        PC.getCNGVMC().getNode_Inlets_JTree_StreamlineCard_InitialBoundary().removeAllChildren();
        PC.getCNGVMC().getNode_Outlets_JTree_StreamlineCard_InitialBoundary().removeAllChildren();
        PC.getCNGVMC().getNode_other_JTree_StreamlineCard_InitialBoundary().removeAllChildren();
        PC.getCNGVMC().getHashMap_Nodes_InitialSurfaceTree_StreamlineCard().clear();

        for (Boundary B : PC.getEEDMan().getBoundariesSelectedList()) {


            BoundaryType BT = B.getBoundaryType();
            if (BT instanceof BlowerInletBoundary
                    || BT instanceof InletBoundary
                    || BT instanceof MassFlowBoundary
                    || BT instanceof MixingPlaneInflowBoundary
                    || BT instanceof StagnationBoundary) {
                DefaultMutableTreeNode DMTN = new DefaultMutableTreeNode(B);
                PC.getCNGVMC().getNode_Inlets_JTree_StreamlineCard_InitialBoundary().add(DMTN);
                PC.getCNGVMC().getHashMap_Nodes_InitialSurfaceTree_StreamlineCard().put(B, DMTN);
            } else {
                if (BT instanceof BlowerOutletBoundary
                        || BT instanceof MixingPlaneInflowBoundary
                        || BT instanceof OutletBoundary
                        || BT instanceof PressureBoundary) {

                    DefaultMutableTreeNode DMTN = new DefaultMutableTreeNode(B);
                    PC.getCNGVMC().getNode_Outlets_JTree_StreamlineCard_InitialBoundary().add(DMTN);
                    PC.getCNGVMC().getHashMap_Nodes_InitialSurfaceTree_StreamlineCard().put(B, DMTN);
                } else {

                    DefaultMutableTreeNode DMTN = new DefaultMutableTreeNode(B);
                    PC.getCNGVMC().getNode_other_JTree_StreamlineCard_InitialBoundary().add(DMTN);
                    PC.getCNGVMC().getHashMap_Nodes_InitialSurfaceTree_StreamlineCard().put(B, DMTN);
                }
            }
        }

        PC.getCNGVMC().getJTree_StreamlineCard_InitialBoundary_TreeModel().nodeStructureChanged(
                PC.getCNGVMC().getRootnode_JTree_StreamlineCard_InitialBoundary());
        JTreeExpansion.expandAllNodes(PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().
                getjPanelCreator().getJPanel_CardLayout_TypeSettings().getStreamlineCard().
                getJTree_InitialBoundary());

    }

    private void MapConstructs() {
        PC.getCNGVMC().getRootnode_JTree_CreatedConstructs().removeAllChildren();
        PC.getCNGVMC().getHashMap_Nodes_CreatedConstructsTree().clear();

        for (Entry<String, Construct> e : PC.getCNGDMan().getConMan().getConstructList().entrySet()) {
            DefaultMutableTreeNode DMTN = new DefaultMutableTreeNode(e.getKey());
            PC.getCNGVMC().getRootnode_JTree_CreatedConstructs().add(DMTN);
            PC.getCNGVMC().getHashMap_Nodes_CreatedConstructsTree().put(e.getKey(), DMTN);
        }

        SortJTree.sortTree(PC.getCNGVMC().getRootnode_JTree_CreatedConstructs());

        PC.getCNGVMC().getJTree_CreatedConstructs_TreeModel().nodeStructureChanged(PC.getCNGVMC().
                getRootnode_JTree_CreatedConstructs());

        JTreeExpansion.expandAllNodes(PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().
                getjPanel_CreatedConstructsList().
                getJScrollPane_JTree_CreatedVisualizationConstructs().getJTree_createdConstructsList());
    }

    private void Map_ConstructToCard(Construct Con) {
        //Card Layout anpasssen
        if (Con == null) {
            PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                    getJPanel_CardLayout_TypeSettings().setStatus(
                    Configuration_Tool.STATUS_ChooseType);

        } else {

            if (Configuration_Tool.VisualizationType_Geometry.equals(Con.getTyp())) {

                PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                        getJPanel_CardLayout_TypeSettings().setStatus(
                        Configuration_Tool.STATUS_Geometry);


            }

            if (Configuration_Tool.VisualizationType_CuttingSurface.equals(Con.getTyp())) {
                PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                        getJPanel_CardLayout_TypeSettings().setStatus(
                        Configuration_Tool.STATUS_CuttingSurface);

            }

            if (Configuration_Tool.VisualizationType_CuttingSurface_Series.equals(Con.getTyp())) {
                PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                        getJPanel_CardLayout_TypeSettings().setStatus(
                        Configuration_Tool.STATUS_CuttingSurface_Series);

            }

            if (Configuration_Tool.VisualizationType_Streamline.equals(Con.getTyp())) {
                PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                        getJPanel_CardLayout_TypeSettings().setStatus(
                        Configuration_Tool.STATUS_Streamline);

            }
            if (Configuration_Tool.VisualizationType_IsoSurface.equals(Con.getTyp())) {
                PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                        getJPanel_CardLayout_TypeSettings().setStatus(
                        Configuration_Tool.STATUS_IsoSurface);

            }

            //Selection Anpassen Geometry
            TreePath[] Selected = new TreePath[Con.getParts().size()];
            int i = 0;
            for (Entry<Object, Integer> e : Con.getParts().entrySet()) {
                if (e.getKey() instanceof Region) {
                    DefaultMutableTreeNode a = PC.getEEVMC().getChosenGeometryTreeRegionsNodesHashMap().
                            get((Region) e.getKey());
                    TreePath t = new TreePath(PC.getEEVMC().getEnsightExport_ChosenGeometryTree_TreeModel().
                            getPathToRoot(a));
                    Selected[i] = t;
                    i++;
                }
                if (e.getKey() instanceof Boundary) {
                    DefaultMutableTreeNode a = PC.getEEVMC().getChosenGeometryTreeBoundariesNodesHashMap().
                            get((Boundary) e.getKey());
                    TreePath t = new TreePath(PC.getEEVMC().getEnsightExport_ChosenGeometryTree_TreeModel().
                            getPathToRoot(a));
                    Selected[i] = t;
                    i++;
                }
            }


            //Map to Spezific Card
            if (Con instanceof Construct_GeometryVisualization) {
                //Data
                try {
                    FieldFunctionplusType FFpT = Con.getFFplType();
                    if (FFpT != null) {
                        if (!PC.isEventManager_Enabled()) {
                            PC.getContr_CNG().getContr_GeometryCard().setEnabled(true);

                            PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                    getJPanel_CardLayout_TypeSettings().getGeometryCard().getJComboBox_DataTypeChooser().
                                    setSelectedItem(FFpT.getType());
                            PC.getContr_CNG().getContr_GeometryCard().setFromMapper(false);

                        } else {

                            PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                    getJPanel_CardLayout_TypeSettings().getGeometryCard().getJComboBox_DataTypeChooser().
                                    setSelectedItem(
                                    FFpT.getType());

                        }

                        PC.getContr_CNG().getContr_GeometryCard().setFromMapper(true);
                        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                getJPanel_CardLayout_TypeSettings().getGeometryCard().getJComboBox_DataChooser().
                                setSelectedItem(FFpT.getFF());
                        PC.getContr_CNG().getContr_GeometryCard().setFromMapper(false);
                    } else {
                        if (!PC.isEventManager_Enabled()) {
                            PC.getContr_CNG().getContr_GeometryCard().setEnabled(true);
                            PC.getContr_CNG().getContr_GeometryCard().setFromMapper(true);
                            PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                    getJPanel_CardLayout_TypeSettings().getGeometryCard().getJComboBox_DataTypeChooser().
                                    setSelectedItem(Configuration_Tool.DataType_none);
                            PC.getContr_CNG().getContr_GeometryCard().setFromMapper(false);
                            PC.getContr_CNG().getContr_GeometryCard().setEnabled(false);
                        } else {
                            PC.getContr_CNG().getContr_GeometryCard().setFromMapper(true);
                            PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                    getJPanel_CardLayout_TypeSettings().getGeometryCard().getJComboBox_DataTypeChooser().
                                    setSelectedItem(Configuration_Tool.DataType_none);
                            PC.getContr_CNG().getContr_GeometryCard().setFromMapper(false);

                        }




                        //Color
                        if (((Construct_GeometryVisualization) Con).getColor().equals(
                                Configuration_Tool.Color_grey)) {
                            PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                    getJPanel_CardLayout_TypeSettings().getGeometryCard().getJRadioButton_Grey().
                                    setSelected(true);
                        }
                        if (((Construct_GeometryVisualization) Con).getColor().equals(
                                Configuration_Tool.Color_red)) {
                            PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                    getJPanel_CardLayout_TypeSettings().getGeometryCard().getJRadioButton_Red().
                                    setSelected(true);
                        }
                        if (((Construct_GeometryVisualization) Con).getColor().equals(
                                Configuration_Tool.Color_green)) {
                            PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                    getJPanel_CardLayout_TypeSettings().getGeometryCard().getJRadioButton_Green().
                                    setSelected(true);
                        }
                        //Transparency
                        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                getJPanel_CardLayout_TypeSettings().getGeometryCard().getJTextField_Transparency().
                                setText(String.valueOf(((Construct_GeometryVisualization) Con).getTransparency()));

                    }
                } catch (Exception Ex) {
                    StringWriter sw = new StringWriter();
                    Ex.printStackTrace(new PrintWriter(sw));
                    new Error_Dialog(Configuration_GUI_Strings.Occourence + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.ErrMass + Ex.
                            getMessage()
                            + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.StackTrace + sw.toString());
                }



                //Trees

                if (!PC.isEventManager_Enabled()) {
                    PC.getContr_CNG().getContr_GeometryCard().setEnabled(true);
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getGeometryCard().getJTree_ChosenGeometry().
                            getSelectionModel().clearSelection();
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getGeometryCard().getJTree_ChosenGeometry().
                            getSelectionModel().addSelectionPaths(Selected);
                    PC.getContr_CNG().getContr_GeometryCard().setEnabled(false);
                } else {
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getGeometryCard().getJTree_ChosenGeometry().
                            getSelectionModel().clearSelection();

                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getGeometryCard().getJTree_ChosenGeometry().
                            getSelectionModel().addSelectionPaths(Selected);

                }

            }
            if (Con instanceof Construct_CuttingSurface && !(Con instanceof Construct_CuttingSurfaceSeries)) {
                //Data
                try {
                    FieldFunctionplusType FFpT = Con.getFFplType();
                    if (FFpT != null) {
                        if (!PC.isEventManager_Enabled()) {
                            PC.getContr_CNG().getContr_CuttingSurfaceCard().setEnabled(true);
                            PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                    getJPanel_CardLayout_TypeSettings().getCuttingSurfaceCard().
                                    getJComboBox_DataTypeChooser().
                                    setSelectedItem(
                                    FFpT.getType());
                            PC.getContr_CNG().getContr_CuttingSurfaceCard().setEnabled(false);
                        } else {
                            PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                    getJPanel_CardLayout_TypeSettings().getCuttingSurfaceCard().
                                    getJComboBox_DataTypeChooser().
                                    setSelectedItem(
                                    FFpT.getType());
                        }
                        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                getJPanel_CardLayout_TypeSettings().getCuttingSurfaceCard().
                                getJComboBox_DataChooser().
                                setSelectedItem(FFpT.getFF());
                    } else {

                        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                getJPanel_CardLayout_TypeSettings().getCuttingSurfaceCard().
                                getJComboBox_DataTypeChooser().
                                setSelectedItem(Configuration_Tool.DataType_scalar);
                        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                getJPanel_CardLayout_TypeSettings().getCuttingSurfaceCard().
                                getJComboBox_DataChooser().setModel(PC.getCNGVMC().getComboBoxModel_Scalar());
                        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                getJPanel_CardLayout_TypeSettings().getCuttingSurfaceCard().
                                getJComboBox_DataChooser().setSelectedIndex(-1);

                    }
                } catch (Exception Ex) {
                    StringWriter sw = new StringWriter();
                    Ex.printStackTrace(new PrintWriter(sw));
                    new Error_Dialog(Configuration_GUI_Strings.Occourence+ Configuration_GUI_Strings.eol + Configuration_GUI_Strings.ErrMass + Ex.getMessage() + Configuration_GUI_Strings.eol
                            + Configuration_GUI_Strings.StackTrace + sw.toString());
                }

                //Trees

                if (!PC.isEventManager_Enabled()) {
                    PC.getContr_CNG().getContr_CuttingSurfaceCard().setEnabled(true);
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getCuttingSurfaceCard().
                            getJTree_ChosenGeometry().
                            getSelectionModel().clearSelection();
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getCuttingSurfaceCard().
                            getJTree_ChosenGeometry().
                            getSelectionModel().addSelectionPaths(Selected);
                    PC.getContr_CNG().getContr_CuttingSurfaceCard().setEnabled(false);
                } else {
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getCuttingSurfaceCard().
                            getJTree_ChosenGeometry().
                            getSelectionModel().clearSelection();
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getCuttingSurfaceCard().
                            getJTree_ChosenGeometry().
                            getSelectionModel().addSelectionPaths(Selected);
                }




                //Direction
                if (((Construct_CuttingSurface) Con).getDirection().equals(
                        Configuration_Tool.RadioButtonActionCommand_X_Direction)) {
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getCuttingSurfaceCard().
                            getJRadioButton_X_Direction().setSelected(true);
                }
                if (((Construct_CuttingSurface) Con).getDirection().equals(
                        Configuration_Tool.RadioButtonActionCommand_Y_Direction)) {
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getCuttingSurfaceCard().
                            getJRadioButton_Y_Direction().setSelected(true);
                }
                if (((Construct_CuttingSurface) Con).getDirection().equals(
                        Configuration_Tool.RadioButtonActionCommand_Z_Direction)) {
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getCuttingSurfaceCard().
                            getJRadioButton_Z_Direction().setSelected(true);
                }
                if (((Construct_CuttingSurface) Con).getDirection().equals(
                        Configuration_Tool.RadioButtonActionCommand_notKart_Direction)) {
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getCuttingSurfaceCard().getBGroup().clearSelection();

                }

                //Distance
                PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                        getJPanel_CardLayout_TypeSettings().getCuttingSurfaceCard().getJTextField_Distance().
                        setText(String.valueOf(((Construct_CuttingSurface) Con).getDistance()));



            }
            if (Con instanceof Construct_CuttingSurfaceSeries) {
                //Data
                try {
                    FieldFunctionplusType FFpT = Con.getFFplType();
                    if (FFpT != null) {
                        if (!PC.isEventManager_Enabled()) {
                            PC.getContr_CNG().getContr_CuttingSurfaceSeriesCard().setEnabled(true);
                            PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                    getJPanel_CardLayout_TypeSettings().getCuttingSurfaceSeriesCard().
                                    getJComboBox_DataTypeChooser().setSelectedItem(
                                    FFpT.getType());
                            PC.getContr_CNG().getContr_CuttingSurfaceSeriesCard().setEnabled(false);
                        } else {
                            PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                    getJPanel_CardLayout_TypeSettings().getCuttingSurfaceSeriesCard().
                                    getJComboBox_DataTypeChooser().setSelectedItem(
                                    FFpT.getType());
                        }
                        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                getJPanel_CardLayout_TypeSettings().getCuttingSurfaceSeriesCard().
                                getJComboBox_DataChooser().
                                setSelectedItem(FFpT.getFF());
                    } else {

                        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                getJPanel_CardLayout_TypeSettings().getCuttingSurfaceSeriesCard().
                                getJComboBox_DataTypeChooser().
                                setSelectedItem(Configuration_Tool.DataType_scalar);
                        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                getJPanel_CardLayout_TypeSettings().getCuttingSurfaceSeriesCard().
                                getJComboBox_DataChooser().setModel(PC.getCNGVMC().getComboBoxModel_Scalar());
                        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                getJPanel_CardLayout_TypeSettings().getCuttingSurfaceSeriesCard().
                                getJComboBox_DataChooser().setSelectedIndex(-1);
                    }
                } catch (Exception Ex) {
                    StringWriter sw = new StringWriter();
                    Ex.printStackTrace(new PrintWriter(sw));
                    new Error_Dialog(Configuration_GUI_Strings.Occourence + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.ErrMass + Ex.getMessage() + Configuration_GUI_Strings.eol
                            + Configuration_GUI_Strings.StackTrace + sw.toString());
                }




                //Trees

                if (!PC.isEventManager_Enabled()) {
                    PC.getContr_CNG().getContr_CuttingSurfaceSeriesCard().setEnabled(true);
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getCuttingSurfaceSeriesCard().
                            getJTree_ChosenGeometry().
                            getSelectionModel().clearSelection();
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getCuttingSurfaceSeriesCard().
                            getJTree_ChosenGeometry().
                            getSelectionModel().addSelectionPaths(Selected);
                    PC.getContr_CNG().getContr_CuttingSurfaceSeriesCard().setEnabled(false);
                } else {
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getCuttingSurfaceSeriesCard().
                            getJTree_ChosenGeometry().
                            getSelectionModel().clearSelection();
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getCuttingSurfaceSeriesCard().
                            getJTree_ChosenGeometry().
                            getSelectionModel().addSelectionPaths(Selected);
                }



                //Direction
                if (((Construct_CuttingSurfaceSeries) Con).getDirection().equals(
                        Configuration_Tool.RadioButtonActionCommand_X_Direction)) {
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getCuttingSurfaceSeriesCard().
                            getJRadioButton_X_Direction().setSelected(true);
                }
                if (((Construct_CuttingSurfaceSeries) Con).getDirection().equals(
                        Configuration_Tool.RadioButtonActionCommand_Y_Direction)) {
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getCuttingSurfaceSeriesCard().
                            getJRadioButton_Y_Direction().setSelected(true);
                }
                if (((Construct_CuttingSurfaceSeries) Con).getDirection().equals(
                        Configuration_Tool.RadioButtonActionCommand_Z_Direction)) {
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getCuttingSurfaceSeriesCard().
                            getJRadioButton_Z_Direction().setSelected(true);
                }

                //Distance
                PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                        getJPanel_CardLayout_TypeSettings().getCuttingSurfaceSeriesCard().
                        getJTextField_Distance().
                        setText(String.valueOf(((Construct_CuttingSurfaceSeries) Con).getDistance()));
                //Amount
                PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                        getJPanel_CardLayout_TypeSettings().getCuttingSurfaceSeriesCard().
                        getJTextField_NumberOfCuts().setText(String.valueOf(((Construct_CuttingSurfaceSeries) Con).
                        getAmount()));

                //cutting Distance
                PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                        getJPanel_CardLayout_TypeSettings().getCuttingSurfaceSeriesCard().
                        getJTextField_CuttingDistance().setText(String.valueOf(((Construct_CuttingSurfaceSeries) Con).
                        getDistanceBetween()));

            }
            if (Con instanceof Construct_Streamline) {
                //JComboBox FieldFunction
                try {
                    if (Con.getFFplType() != null) {
                        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                getJPanel_CardLayout_TypeSettings().getStreamlineCard().
                                getJComboBox_DataChooser().
                                setSelectedItem(Con.getFFplType().getFF());
                    } else {
                        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                getJPanel_CardLayout_TypeSettings().getStreamlineCard().
                                getJComboBox_DataChooser().
                                setSelectedIndex(-1);

                    }
                } catch (Exception Ex) {
                    StringWriter sw = new StringWriter();
                    Ex.printStackTrace(new PrintWriter(sw));
                    new Error_Dialog(Configuration_GUI_Strings.Occourence + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.ErrMass + Ex.getMessage() + Configuration_GUI_Strings.eol
                            + Configuration_GUI_Strings.StackTrace + sw.toString());
                }


                //Trees
                //Geometry Trees
                if (!PC.isEventManager_Enabled()) {
                    PC.getContr_CNG().getContr_StreamlineCard().setEnabled(true);
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getStreamlineCard().
                            getJTree_ChosenGeometry().
                            getSelectionModel().clearSelection();
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getStreamlineCard().
                            getJTree_ChosenGeometry().
                            getSelectionModel().addSelectionPaths(Selected);

                    PC.getContr_CNG().getContr_StreamlineCard().setEnabled(false);
                } else {
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getStreamlineCard().
                            getJTree_ChosenGeometry().
                            getSelectionModel().clearSelection();
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getStreamlineCard().
                            getJTree_ChosenGeometry().
                            getSelectionModel().addSelectionPaths(Selected);
                }



                //Initial Surface
                TreeSelectionModel tsm = PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().
                        getjPanelCreator().getJPanel_CardLayout_TypeSettings().getStreamlineCard().
                        getJTree_InitialBoundary().getSelectionModel();



                if (!((Construct_Streamline) Con).getInitialSurface().isEmpty()) {
                    HashMap<Object, Integer> InitialSurfaceHashMap = ((Construct_Streamline) Con).getInitialSurface();
                    TreePath[] TreePaths = new TreePath[InitialSurfaceHashMap.size()];
                    int ii = 0;
                    for (Entry<Object, Integer> E : InitialSurfaceHashMap.entrySet()) {
                        TreePaths[ii] = new TreePath(PC.getCNGVMC().getHashMap_Nodes_InitialSurfaceTree_StreamlineCard().
                                get(E.getKey()).getPath());
                        ii++;
                    }
                    tsm.setSelectionPaths(TreePaths);

                } else {
                    tsm.setSelectionPath(null);
                }

                //Divisions
                PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                        getJPanel_CardLayout_TypeSettings().getStreamlineCard().getJTextField_DivisionX().
                        setText(String.valueOf(((Construct_Streamline) Con).getDivisions().x));
                PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                        getJPanel_CardLayout_TypeSettings().getStreamlineCard().getJTextField_DivisionY().
                        setText(String.valueOf(((Construct_Streamline) Con).getDivisions().y));
                PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                        getJPanel_CardLayout_TypeSettings().getStreamlineCard().getJTextField_DivisionZ().
                        setText(String.valueOf(((Construct_Streamline) Con).getDivisions().z));

                //Tube Radius
                PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                        getJPanel_CardLayout_TypeSettings().getStreamlineCard().getJTextField_Tube_Radius().
                        setText(String.valueOf(((Construct_Streamline) Con).getTube_Radius()));

                //Trace Length
                PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                        getJPanel_CardLayout_TypeSettings().getStreamlineCard().getJTextField_trace_length().
                        setText(String.valueOf(((Construct_Streamline) Con).getTrace_length()));

                //max out of domain
                PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                        getJPanel_CardLayout_TypeSettings().getStreamlineCard().
                        getJTextField_max_out_of_domain().
                        setText(String.valueOf(((Construct_Streamline) Con).getMax_out_of_domain()));
                //Direction
                if (((Construct_Streamline) Con).getDirection().equals(
                        Configuration_Tool.forward)) {
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getStreamlineCard().getJRadioButton_forward().
                            setSelected(true);
                }
                if (((Construct_Streamline) Con).getDirection().equals(
                        Configuration_Tool.back)) {
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getStreamlineCard().getJRadioButton_back().setSelected(
                            true);
                }
                if (((Construct_Streamline) Con).getDirection().equals(
                        Configuration_Tool.both)) {
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getStreamlineCard().getJRadioButton_both().setSelected(
                            true);
                }
            }
            if (Con instanceof Construct_IsoSurface) {
                FieldFunctionplusType FFpT = Con.getFFplType();
                if (FFpT != null) {
                    if (!PC.isEventManager_Enabled()) {
                        PC.getContr_CNG().getContr_IsoSurfaceCard().setEnabled(true);
                        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                getJPanel_CardLayout_TypeSettings().getIsoSurfaceCard().
                                getJComboBox_DataTypeChooser().setSelectedItem(
                                FFpT.getType());
                        PC.getContr_CNG().getContr_IsoSurfaceCard().setEnabled(false);
                    } else {
                        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                                getJPanel_CardLayout_TypeSettings().getCuttingSurfaceSeriesCard().
                                getJComboBox_DataTypeChooser().setSelectedItem(
                                FFpT.getType());
                    }
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getIsoSurfaceCard().getJComboBox_DataChooser().
                            setSelectedItem(FFpT.getFF());
                } else {

                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getIsoSurfaceCard().
                            getJComboBox_DataTypeChooser().
                            setSelectedItem(Configuration_Tool.DataType_scalar);
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getIsoSurfaceCard().
                            getJComboBox_DataChooser().setModel(PC.getCNGVMC().getComboBoxModel_Scalar());
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getIsoSurfaceCard().
                            getJComboBox_DataChooser().setSelectedIndex(-1);
                }


                //Trees


                if (!PC.isEventManager_Enabled()) {
                    PC.getContr_CNG().getContr_IsoSurfaceCard().setEnabled(true);
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getIsoSurfaceCard().getJTree_ChosenGeometry().
                            getSelectionModel().clearSelection();
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getIsoSurfaceCard().getJTree_ChosenGeometry().
                            getSelectionModel().addSelectionPaths(Selected);
                    PC.getContr_CNG().getContr_IsoSurfaceCard().setEnabled(false);
                } else {
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getIsoSurfaceCard().getJTree_ChosenGeometry().
                            getSelectionModel().clearSelection();
                    PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                            getJPanel_CardLayout_TypeSettings().getIsoSurfaceCard().getJTree_ChosenGeometry().
                            getSelectionModel().addSelectionPaths(Selected);
                }

                //IsoValue
                PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getjPanelCreator().
                        getJPanel_CardLayout_TypeSettings().getIsoSurfaceCard().getJTextField_IsoValue().
                        setText(String.valueOf(((Construct_IsoSurface) Con).getIsoValue()));
            }
        }
    }

    public void CoviseNetGeneration_ExportPathChanged() {
        PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().getCoviseNetGenerationFolderPanel().
                getLabelDestinationFile().setText(PC.getCNGDMan().getExportPath().getAbsolutePath());
    }

    public void GeometryCard_GeometrySelectionChanged() {
        Map_GeometryCard_GeometryToVisualizeSelection();
    }

    public void CuttingSurfaceCard_GeometrySelectionChanged() {
        Map_CuttingSurfaceCard_GeometryToVisualizeSelection();
    }

    public void CuttingSurfaceSeriesCard_GeometrySelectionChanged() {
        Map_CuttingSurfaceSeriesCard_GeometryToVisualizeSelection();
    }

    public void StreamlineCard_GeometrySelectionChanged() {
        Map_StreamlineCard_GeometryToVisualizeSelection();

    }

    public void IsoSurfaceCard_GeometrySelectionChanged() {
        Map_IsoSurfaceCard_GeometryToVisualizeSelection();
    }

    public void ConstructListChanged() {
        MapConstructs();
    }

    public void SelectionChanged(Construct Con) {
        TreeSelectionModel tsm = PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration().
                getjPanel_CreatedConstructsList().getJScrollPane_JTree_CreatedVisualizationConstructs().
                getJTree_createdConstructsList().getSelectionModel();

        if (Con != null) {

            tsm.setSelectionPath(new TreePath(
                    PC.getCNGVMC().getHashMap_Nodes_CreatedConstructsTree().get(Con.toString()).getPath()));
        } else {
            tsm.setSelectionPath(null);
        }
        Map_ConstructToCard(Con);


    }

    public void CoviseNetGenerationExportPathChanged() {
        CoviseNetGeneration_ExportPathChanged();
    }

    public void RegionSelectionChanged() {
    }

    public void BoundarySelectionChanged() {
        Map_StreamlineCard_InitialSurfaceChoice();
    }

    public void ScalarsSelectionChanged() {
    }

    public void VectorsSelectionChanged() {
    }

    public void EnsightExportPathChanged() {
    }

    public void ExportonVerticesChangedChanged() {
    }

    public void AppendToExistingFileChanged() {
    }
}
