package de.hlrs.starplugin.covise_net_generation;

import Main.PluginContainer;
import de.hlrs.starplugin.interfaces.Interface_CoviseNetGeneration_ViewModelChangedListener;
import de.hlrs.starplugin.interfaces.Interface_EnsightExport_DataChangedListener;
import de.hlrs.starplugin.configuration.Configuration_Tool;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import javax.swing.DefaultComboBoxModel;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.DefaultTreeModel;

/**
 *holds the DataModels for the CoviseNetGenertion Panel Swing Views
 * @author Weiss HLRS Stuttgart
 */
public class JPanel_CoviseNetGeneration_ViewModelContainer implements Interface_EnsightExport_DataChangedListener {

    private PluginContainer PC;
    //TreeModels
    private DefaultTreeModel JTree_GeometryCard_GeometryToVisualize_TreeModel;
    private DefaultTreeModel JTree_CuttingSurfaceCard_GeometryToVisualize_TreeModel;
    private DefaultTreeModel JTree_CuttingSurfaceSeriesCard_GeometryToVisualize_TreeModel;
    private DefaultTreeModel JTree_StreamlineCard_GeometryToVisualize_TreeModel;
    private DefaultTreeModel JTree_IsoSurfaceCard_GeometryToVisualize_TreeModel;
    private DefaultTreeModel JTree_StreamlineCard_InitialBoundary_TreeModel;
    private DefaultTreeModel JTree_CreatedConstructs_TreeModel;
    //ComboBoxModels
    private DefaultComboBoxModel<Object> ComboBoxModel_Vector;
    private DefaultComboBoxModel<Object> ComboBoxModel_Scalar;
    private DefaultComboBoxModel<Object> ComboBoxModel_StreamlineCard_Velocity;
    private DefaultComboBoxModel<String> ComboBoxModel_DataType;
    //Nodes
    private DefaultMutableTreeNode rootnode_JTree_GeometryCard_GeometryToVisualize = new DefaultMutableTreeNode(
            "rootnode_GeometryCard_GeometryToVisualize");
    private DefaultMutableTreeNode rootnode_JTree_CuttingSurfaceCard_GeometryToVisualize = new DefaultMutableTreeNode(
            "rootnode_CuttingSurfaceCard_GeometryToVisualize");
    private DefaultMutableTreeNode rootnode_JTree_CuttingSurfaceSeriesCard_GeometryToVisualize = new DefaultMutableTreeNode(
            "rootnode_CuttingSurfaceSeriesCard_GeometryToVisualize");
    private DefaultMutableTreeNode rootnode_JTree_StreamlineCard_GeometryToVisualize = new DefaultMutableTreeNode(
            "rootnode_StreamlineCard_GeometryToVisualize");
    private DefaultMutableTreeNode rootnode_JTree_IsoSurfaceCard_GeometryToVisualize = new DefaultMutableTreeNode(
            "rootnode_IsoSurfaceCard_GeometryToVisualize");
    private DefaultMutableTreeNode rootnode_JTree_StreamlineCard_InitialBoundary = new DefaultMutableTreeNode(
            "rootnode_StreamlineCard_InletBoundary");
    private DefaultMutableTreeNode rootnode_JTree_CreatedConstructs = new DefaultMutableTreeNode(
            "rootnodeCreatedConstructs");
    private DefaultMutableTreeNode Node_Regions_JTree_GeometryCard_GeometryToVisualize = new DefaultMutableTreeNode(
            "Regions");
    private DefaultMutableTreeNode Node_Boundaries_JTree_GeometryCard_GeometryToVisualize = new DefaultMutableTreeNode(
            "Boundaries");
    private DefaultMutableTreeNode Node_Inlets_JTree_StreamlineCard_InitialBoundary = new DefaultMutableTreeNode(
            "Inlets");
    private DefaultMutableTreeNode Node_Outlets_JTree_StreamlineCard_InitialBoundary = new DefaultMutableTreeNode(
            "Outlets");
    private DefaultMutableTreeNode Node_other_JTree_StreamlineCard_InitialBoundary = new DefaultMutableTreeNode(
            "other");
    private HashMap<String, DefaultMutableTreeNode> HashMap_Nodes_CreatedConstructsTree = new HashMap<String, DefaultMutableTreeNode>();
    private HashMap<Object, DefaultMutableTreeNode> HashMap_Nodes_InitialSurfaceTree_StreamlineCard = new HashMap<Object, DefaultMutableTreeNode>();
    //ArrayList SelectedGeometryNodes to Visualize
    private ArrayList<DefaultMutableTreeNode> List_GeometryCard_SelectedGeometryNodes;
    private ArrayList<DefaultMutableTreeNode> List_CuttingSurfaceCard_SelectedGeometryNodes;
    private ArrayList<DefaultMutableTreeNode> List_CuttingSurfaceSeriesCard_SelectedGeometryNodes;
    private ArrayList<DefaultMutableTreeNode> List_StreamlineCard_SelectedGeometryNodes;
    private ArrayList<DefaultMutableTreeNode> List_IsoSurfaceCard_SelectedGeometryNodes;
    //Funktionalität
    private List<Interface_CoviseNetGeneration_ViewModelChangedListener> listeners = new ArrayList<Interface_CoviseNetGeneration_ViewModelChangedListener>();

    //Konstructor
    public JPanel_CoviseNetGeneration_ViewModelContainer(PluginContainer Pc) {
        this.PC = Pc;
        initiateTreeModel_GeometryCard_GeometrytoVisualize();
        initiateTreeModel_CuttingSurfaceCard_GeometrytoVisualize();
        initiateTreeModel_CuttingSurfaceSeriesCard_GeometrytoVisualize();
        initiateTreeModel_StreamlineCard_GeometrytoVisualize();
        initiateTreeModel_IsoSurfaceCard_GeometrytoVisualize();

        initiateTreeModel_StreamlineCard_InitialSurface();

        initiateTreeModel_CreatedConstructs();

        initiate_ComboBoxModels();

        List_GeometryCard_SelectedGeometryNodes = new ArrayList<DefaultMutableTreeNode>();
        List_CuttingSurfaceCard_SelectedGeometryNodes = new ArrayList<DefaultMutableTreeNode>();
        List_CuttingSurfaceSeriesCard_SelectedGeometryNodes = new ArrayList<DefaultMutableTreeNode>();
        List_StreamlineCard_SelectedGeometryNodes = new ArrayList<DefaultMutableTreeNode>();
        List_IsoSurfaceCard_SelectedGeometryNodes = new ArrayList<DefaultMutableTreeNode>();
        PC.getEEDMan().addListener(this);

    }

    //Getter
    //JTree Models Type Cards
    public DefaultTreeModel getJTree_CuttingSurfaceCard_GeometryToVisualize_TreeModel() {
        return JTree_CuttingSurfaceCard_GeometryToVisualize_TreeModel;
    }

    public DefaultTreeModel getJTree_CuttingSurfaceSeriesCard_GeometryToVisualize_TreeModel() {
        return JTree_CuttingSurfaceSeriesCard_GeometryToVisualize_TreeModel;
    }

    public DefaultTreeModel getJTree_GeometryCard_GeometryToVisualize_TreeModel() {
        return JTree_GeometryCard_GeometryToVisualize_TreeModel;
    }

    public DefaultTreeModel getJTree_IsoSurfaceCard_GeometryToVisualize_TreeModel() {
        return JTree_IsoSurfaceCard_GeometryToVisualize_TreeModel;
    }

    public DefaultTreeModel getJTree_StreamlineCard_GeometryToVisualize_TreeModel() {
        return JTree_StreamlineCard_GeometryToVisualize_TreeModel;
    }

    public DefaultTreeModel getJTree_StreamlineCard_InitialBoundary_TreeModel() {
        return JTree_StreamlineCard_InitialBoundary_TreeModel;
    }

    //JTreeCreadted Constructs
    //ComboBoxes
    //critical Nodes
    public DefaultTreeModel getJTree_CreatedConstructs_TreeModel() {
        return JTree_CreatedConstructs_TreeModel;
    }

    public DefaultComboBoxModel<String> getComboBoxModel_DataType() {
        return ComboBoxModel_DataType;
    }

    public DefaultComboBoxModel<Object> getComboBoxModel_Scalar() {
        return ComboBoxModel_Scalar;
    }

    public DefaultComboBoxModel<Object> getComboBoxModel_StreamlineCard_Velocity() {
        return ComboBoxModel_StreamlineCard_Velocity;
    }

    public DefaultComboBoxModel<Object> getComboBoxModel_Vector() {
        return ComboBoxModel_Vector;
    }

    public DefaultMutableTreeNode getNode_Boundaries_JTree_GeometryCard_GeometryToVisualize() {
        return Node_Boundaries_JTree_GeometryCard_GeometryToVisualize;
    }

    public DefaultMutableTreeNode getNode_Inlets_JTree_StreamlineCard_InitialBoundary() {
        return Node_Inlets_JTree_StreamlineCard_InitialBoundary;
    }

    public DefaultMutableTreeNode getNode_Outlets_JTree_StreamlineCard_InitialBoundary() {
        return Node_Outlets_JTree_StreamlineCard_InitialBoundary;
    }

    public DefaultMutableTreeNode getNode_Regions_JTree_GeometryCard_GeometryToVisualize() {
        return Node_Regions_JTree_GeometryCard_GeometryToVisualize;
    }

    public DefaultMutableTreeNode getNode_other_JTree_StreamlineCard_InitialBoundary() {
        return Node_other_JTree_StreamlineCard_InitialBoundary;
    }

    //Rootnodes
    public DefaultMutableTreeNode getRootnode_JTree_CreatedConstructs() {
        return rootnode_JTree_CreatedConstructs;
    }

    public DefaultMutableTreeNode getRootnode_JTree_CuttingSurfaceCard_GeometryToVisualize() {
        return rootnode_JTree_CuttingSurfaceCard_GeometryToVisualize;
    }

    public DefaultMutableTreeNode getRootnode_JTree_CuttingSurfaceSeriesCard_GeometryToVisualize() {
        return rootnode_JTree_CuttingSurfaceSeriesCard_GeometryToVisualize;
    }

    public DefaultMutableTreeNode getRootnode_JTree_GeometryCard_GeometryToVisualize() {
        return rootnode_JTree_GeometryCard_GeometryToVisualize;
    }

    public DefaultMutableTreeNode getRootnode_JTree_IsoSurfaceCard_GeometryToVisualize() {
        return rootnode_JTree_IsoSurfaceCard_GeometryToVisualize;
    }

    public DefaultMutableTreeNode getRootnode_JTree_StreamlineCard_GeometryToVisualize() {
        return rootnode_JTree_StreamlineCard_GeometryToVisualize;
    }

    public DefaultMutableTreeNode getRootnode_JTree_StreamlineCard_InitialBoundary() {
        return rootnode_JTree_StreamlineCard_InitialBoundary;
    }

    //HashMaps
    public HashMap<Object, DefaultMutableTreeNode> getHashMap_Nodes_InitialSurfaceTree_StreamlineCard() {
        return HashMap_Nodes_InitialSurfaceTree_StreamlineCard;
    }

    public HashMap<String, DefaultMutableTreeNode> getHashMap_Nodes_CreatedConstructsTree() {
        return HashMap_Nodes_CreatedConstructsTree;
    }

    //Initialisierung der Modelle des Views
    //Initialisierung der TreeModels
    private void initiateTreeModel_GeometryCard_GeometrytoVisualize() {
        this.rootnode_JTree_GeometryCard_GeometryToVisualize.add(
                this.Node_Regions_JTree_GeometryCard_GeometryToVisualize);
        this.rootnode_JTree_GeometryCard_GeometryToVisualize.add(
                this.Node_Boundaries_JTree_GeometryCard_GeometryToVisualize);
        JTree_GeometryCard_GeometryToVisualize_TreeModel = new DefaultTreeModel(
                this.rootnode_JTree_GeometryCard_GeometryToVisualize);
    }

    private void initiateTreeModel_CuttingSurfaceCard_GeometrytoVisualize() {

        JTree_CuttingSurfaceCard_GeometryToVisualize_TreeModel = new DefaultTreeModel(
                this.rootnode_JTree_CuttingSurfaceCard_GeometryToVisualize);
    }

    private void initiateTreeModel_CuttingSurfaceSeriesCard_GeometrytoVisualize() {
        JTree_CuttingSurfaceSeriesCard_GeometryToVisualize_TreeModel = new DefaultTreeModel(
                this.rootnode_JTree_CuttingSurfaceSeriesCard_GeometryToVisualize);
    }

    private void initiateTreeModel_StreamlineCard_GeometrytoVisualize() {
        JTree_StreamlineCard_GeometryToVisualize_TreeModel = new DefaultTreeModel(
                this.rootnode_JTree_StreamlineCard_GeometryToVisualize);
    }

    private void initiateTreeModel_IsoSurfaceCard_GeometrytoVisualize() {
        JTree_IsoSurfaceCard_GeometryToVisualize_TreeModel = new DefaultTreeModel(
                this.rootnode_JTree_IsoSurfaceCard_GeometryToVisualize);
    }

    private void initiateTreeModel_StreamlineCard_InitialSurface() {
        this.rootnode_JTree_StreamlineCard_InitialBoundary.add(
                this.Node_Inlets_JTree_StreamlineCard_InitialBoundary);
        this.rootnode_JTree_StreamlineCard_InitialBoundary.add(
                this.Node_Outlets_JTree_StreamlineCard_InitialBoundary);
        this.rootnode_JTree_StreamlineCard_InitialBoundary.add(
                this.Node_other_JTree_StreamlineCard_InitialBoundary);
        JTree_StreamlineCard_InitialBoundary_TreeModel = new DefaultTreeModel(
                this.rootnode_JTree_StreamlineCard_InitialBoundary);
    }

    private void initiateTreeModel_CreatedConstructs() {
        JTree_CreatedConstructs_TreeModel = new DefaultTreeModel(this.rootnode_JTree_CreatedConstructs);
    }

    //Initialisierung der ComboBoxModels
    private void initiate_ComboBoxModels() {
        ComboBoxModel_Scalar = new DefaultComboBoxModel<Object>((Object[]) PC.getEEDMan().
                getScalarsSelectedList().toArray());
        ComboBoxModel_Vector = new DefaultComboBoxModel<Object>((Object[]) PC.getEEDMan().
                getVectorsSelectedList().toArray());
        ComboBoxModel_DataType = new DefaultComboBoxModel<String>(
                new String[]{Configuration_Tool.DataType_none, Configuration_Tool.DataType_scalar, Configuration_Tool.DataType_vector});
        ComboBoxModel_StreamlineCard_Velocity = new DefaultComboBoxModel<Object>((Object[]) PC.getEEVMC().
                getHashMap_VelocityFieldFuntions().keySet().toArray());
    }
    //Interface zum Update der ComboBoxModels

    public void RegionSelectionChanged() {
    }

    public void BoundarySelectionChanged() {
    }

    public void ScalarsSelectionChanged() {
        ComboBoxModel_Scalar.removeAllElements();
        for (Object o : PC.getEEDMan().getScalarsSelectedList().
                toArray()) {
            ComboBoxModel_Scalar.addElement(o);
        }
    }

    public void VectorsSelectionChanged() {
        ComboBoxModel_Vector.removeAllElements();
        for (Object o : PC.getEEDMan().getVectorsSelectedList().
                toArray()) {
            ComboBoxModel_Vector.addElement(o);
        }
    }

    public void EnsightExportPathChanged() {
    }

    public void ExportonVerticesChangedChanged() {
    }

    public void AppendToExistingFileChanged() {
    }

    public void addListener(Interface_CoviseNetGeneration_ViewModelChangedListener toAdd) {
        listeners.add(toAdd);
    }

    void dataChanged(String e) {

        // Notify everybody that may be interested.
        if (e != null) {
            if (e.equals(Configuration_Tool.onChange_ExportPath)) {
                for (Interface_CoviseNetGeneration_ViewModelChangedListener dcl : listeners) {
                    dcl.CoviseNetGeneration_ExportPathChanged();
                }
            }
            if (e.equals("GeometryCard_GeometrySelection")) {
                for (Interface_CoviseNetGeneration_ViewModelChangedListener dcl : listeners) {
                    dcl.GeometryCard_GeometrySelectionChanged();
                }
            }
            if (e.equals("CuttingSurfaceCard_GeometrySelection")) {
                for (Interface_CoviseNetGeneration_ViewModelChangedListener dcl : listeners) {
                    dcl.CuttingSurfaceCard_GeometrySelectionChanged();
                }
            }
            if (e.equals("CuttingSurfaceSeriesCard_GeometrySelection")) {
                for (Interface_CoviseNetGeneration_ViewModelChangedListener dcl : listeners) {
                    dcl.CuttingSurfaceSeriesCard_GeometrySelectionChanged();
                }
            }
            if (e.equals("StreamlineCard_GeometrySelection")) {
                for (Interface_CoviseNetGeneration_ViewModelChangedListener dcl : listeners) {
                    dcl.StreamlineCard_GeometrySelectionChanged();
                }
            }
            if (e.equals("IsoSurfaceCard_GeometrySelection")) {
                for (Interface_CoviseNetGeneration_ViewModelChangedListener dcl : listeners) {
                    dcl.IsoSurfaceCard_GeometrySelectionChanged();
                }
            }
        }
    }

    public void GeoemtryCard_GeometryToVisualizeSelectionChanged(ArrayList<DefaultMutableTreeNode> List_SelectedNodes) {
        this.List_GeometryCard_SelectedGeometryNodes.clear();
        this.List_GeometryCard_SelectedGeometryNodes.addAll(List_SelectedNodes);
        this.dataChanged("GeometryCard_GeometrySelection");

    }

    public ArrayList<DefaultMutableTreeNode> getList_GeometryCard_SelectedGeometryNodes() {
        return List_GeometryCard_SelectedGeometryNodes;
    }

    public void CuttingSurfaceCard_GeometryToVisualizeSelectionChanged(ArrayList<DefaultMutableTreeNode> List_SelectedNodes) {
        this.List_CuttingSurfaceCard_SelectedGeometryNodes.clear();
        this.List_CuttingSurfaceCard_SelectedGeometryNodes.addAll(List_SelectedNodes);
        this.dataChanged("CuttingSurfaceCard_GeometrySelection");
    }

    public ArrayList<DefaultMutableTreeNode> getList_CuttingSurfaceCard_SelectedGeometryNodes() {
        return List_CuttingSurfaceCard_SelectedGeometryNodes;
    }

    public void CuttingSurfaceSeriesCard_GeometryToVisualizeSelectionChanged(ArrayList<DefaultMutableTreeNode> List_SelectedNodes) {
        this.List_CuttingSurfaceSeriesCard_SelectedGeometryNodes.clear();
        this.List_CuttingSurfaceSeriesCard_SelectedGeometryNodes.addAll(List_SelectedNodes);
        this.dataChanged("CuttingSurfaceSeriesCard_GeometrySelection");
    }

    public ArrayList<DefaultMutableTreeNode> getList_CuttingSurfaceSeriesCard_SelectedGeometryNodes() {
        return List_CuttingSurfaceSeriesCard_SelectedGeometryNodes;
    }

    public void StreamlineCard_GeometryToVisualizeSelectionChanged(ArrayList<DefaultMutableTreeNode> List_SelectedNodes) {
        this.List_StreamlineCard_SelectedGeometryNodes.clear();
        this.List_StreamlineCard_SelectedGeometryNodes.addAll(List_SelectedNodes);
        this.dataChanged("StreamlineCard_GeometrySelection");
    }

    public ArrayList<DefaultMutableTreeNode> getList_StreamlineCard_SelectedGeometryNodes() {
        return List_StreamlineCard_SelectedGeometryNodes;
    }

    public void IsoSurfaceCard_GeometryToVisualizeSelectionChanged(ArrayList<DefaultMutableTreeNode> List_SelectedNodes) {
        this.List_IsoSurfaceCard_SelectedGeometryNodes.clear();
        this.List_IsoSurfaceCard_SelectedGeometryNodes.addAll(List_SelectedNodes);
        this.dataChanged("IsoSurfaceCard_GeometrySelection");
    }

    public ArrayList<DefaultMutableTreeNode> getList_IsoSurfaceCard_SelectedGeometryNodes() {
        return List_IsoSurfaceCard_SelectedGeometryNodes;
    }
}
