/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package de.hlrs.starplugin.ensight_export;

import Main.PluginContainer;
import de.hlrs.starplugin.configuration.Configuration_Tool;
import java.util.HashMap;
import java.util.Map.Entry;
import javax.swing.DefaultListModel;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.DefaultTreeModel;
import star.common.AbstractReferenceFrame;
import star.common.Boundary;
import star.common.CoordinateSystem;
import star.common.FieldFunction;
import star.common.Region;

/**
 *holds the DataModels for the EnsightExport Panel Swing Views
 * @author Weiss HLRS Stuttgart
 */
public class JPanel_EnsightExport_ViewModelContainer {

    private PluginContainer PC;
    private DefaultListModel<Object> EnsightExportTabbedPaneRegionListModel;
    private DefaultTreeModel EnsightExport_BoundariesTree_TreeModel;
    private DefaultTreeModel EnsightExport_ScalarsTree_TreeModel;
    private DefaultTreeModel EnsightExport_VectorsTree_TreeModel;
    private DefaultTreeModel EnsightExport_ChosenGeometryTree_TreeModel;
    private DefaultTreeModel EnsightExport_ChosenScalarsTree_TreeModel;
    private DefaultTreeModel EnsightExport_ChosenVectorsTree_TreeModel;
    //rootnodes
    private DefaultMutableTreeNode rootnodeTreeBoundaries = new DefaultMutableTreeNode(
            "rootnodeBoundaries");
    private DefaultMutableTreeNode rootnodeTreeChosenGeometry = new DefaultMutableTreeNode(
            "rootnodeChosenGeometryTree");
    private DefaultMutableTreeNode nodeTreeBoundariesBoundarieswithoutRegionChosen = new DefaultMutableTreeNode(
            "Region not Chosen");
    private DefaultMutableTreeNode nodeTreeChosenGeometryBoundarieswithoutRegionChosen = new DefaultMutableTreeNode(
            "Region not Chosen");
    private DefaultMutableTreeNode rootnodeTreeScalars = new DefaultMutableTreeNode(
            "rootnodeScalar");
    private DefaultMutableTreeNode rootnodeTreeVectors = new DefaultMutableTreeNode(
            "rootnodeVector");
    private DefaultMutableTreeNode rootnodeChosenScalarsTree = new DefaultMutableTreeNode(
            "rootnodeChosenScalars");
    private DefaultMutableTreeNode rootnodeChosenVectorsTree = new DefaultMutableTreeNode(
            "rootnodeChosenVectors");
    //HashMaps der Knoten
    private HashMap<Region, DefaultMutableTreeNode> BoundariesTreeRegionsNodesHashMap = new HashMap<Region, DefaultMutableTreeNode>();
    private HashMap<Boundary, DefaultMutableTreeNode> BoundariesTreeBoundariesNodesHashMap = new HashMap<Boundary, DefaultMutableTreeNode>();
    private HashMap<Region, DefaultMutableTreeNode> ChosenGeometryTreeRegionsNodesHashMap = new HashMap<Region, DefaultMutableTreeNode>();
    private HashMap<Boundary, DefaultMutableTreeNode> ChosenGeometryTreeBoundariesNodesHashMap = new HashMap<Boundary, DefaultMutableTreeNode>();
    private HashMap<Object, DefaultMutableTreeNode> ScalarsTreeScalarsNodesHashMap = new HashMap<Object, DefaultMutableTreeNode>();
    private HashMap<DefaultMutableTreeNode, DefaultMutableTreeNode> ChosenScalarsTreeScalarsNodesHashMap = new HashMap<DefaultMutableTreeNode, DefaultMutableTreeNode>();
    private HashMap<Object, DefaultMutableTreeNode> VectorsTreeVectorsNodesHashMap = new HashMap<Object, DefaultMutableTreeNode>();
    private HashMap<DefaultMutableTreeNode, DefaultMutableTreeNode> ChosenVectorsTreeVectorsNodesHashMap = new HashMap<DefaultMutableTreeNode, DefaultMutableTreeNode>();
    private HashMap<FieldFunction, FieldFunction> HashMap_VelocityFieldFuntion = new HashMap<FieldFunction, FieldFunction>();

    //Konstructor
    public JPanel_EnsightExport_ViewModelContainer(PluginContainer Pc) {
        this.PC = Pc;
        initiateGeometryNodes();
        initiateFieldFunctionNodes();
        initiateViewDataModels();
    }

    //Getter
    //TreeModels
    public DefaultTreeModel getEnsightExport_ChosenGeometryTree_TreeModel() {
        return EnsightExport_ChosenGeometryTree_TreeModel;
    }

    public DefaultListModel<Object> getEnsightExportTabbedPaneRegionListModel() {
        return EnsightExportTabbedPaneRegionListModel;
    }

    public DefaultTreeModel getEnsightExport_BoundariesTree_TreeModel() {
        return EnsightExport_BoundariesTree_TreeModel;
    }

    public DefaultTreeModel getEnsightExport_ChosenScalarsTree_TreeModel() {
        return EnsightExport_ChosenScalarsTree_TreeModel;
    }

    public DefaultTreeModel getEnsightExport_ChosenVectorsTree_TreeModel() {
        return EnsightExport_ChosenVectorsTree_TreeModel;
    }

    public DefaultTreeModel getEnsightExport_ScalarsTree_TreeModel() {
        return EnsightExport_ScalarsTree_TreeModel;
    }

    public DefaultTreeModel getEnsightExport_VectorsTree_TreeMdoel() {
        return EnsightExport_VectorsTree_TreeModel;
    }

    //Getter
    //HashMaps
    //TabbedPanel Trees
    public HashMap<Boundary, DefaultMutableTreeNode> getBoundariesTreeBoundariesNodesHashMap() {
        return BoundariesTreeBoundariesNodesHashMap;
    }

    public HashMap<Region, DefaultMutableTreeNode> getBoundariesTreeRegionsNodesHashMap() {
        return BoundariesTreeRegionsNodesHashMap;
    }

    public HashMap<Object, DefaultMutableTreeNode> getScalarsTreeScalarsNodesHashMap() {
        return ScalarsTreeScalarsNodesHashMap;
    }

    public HashMap<Object, DefaultMutableTreeNode> getVectorsTreeVectorsNodesHashMap() {
        return VectorsTreeVectorsNodesHashMap;
    }

    public HashMap<FieldFunction, FieldFunction> getHashMap_VelocityFieldFuntions() {
        return HashMap_VelocityFieldFuntion;
    }
    //FieldFunctions for ComboBox CoviseNetGen
    

    //Chosen Trees
    public HashMap<Boundary, DefaultMutableTreeNode> getChosenGeometryTreeBoundariesNodesHashMap() {
        return ChosenGeometryTreeBoundariesNodesHashMap;
    }

    public HashMap<Region, DefaultMutableTreeNode> getChosenGeometryTreeRegionsNodesHashMap() {
        return ChosenGeometryTreeRegionsNodesHashMap;
    }

    public HashMap<DefaultMutableTreeNode, DefaultMutableTreeNode> getChosenScalarsTreeScalarsNodesHashMap() {
        return ChosenScalarsTreeScalarsNodesHashMap;
    }

    public HashMap<DefaultMutableTreeNode, DefaultMutableTreeNode> getChosenVectorsTreeVectorsNodesHashMap() {
        return ChosenVectorsTreeVectorsNodesHashMap;
    }

    //Getter
    //critical TreeNodes
    public DefaultMutableTreeNode getNodeTreeBoundariesBoundarieswithoutRegionChosen() {
        return nodeTreeBoundariesBoundarieswithoutRegionChosen;
    }

    public DefaultMutableTreeNode getNodeTreeChosenGeometryBoundarieswithoutRegionChosen() {
        return nodeTreeChosenGeometryBoundarieswithoutRegionChosen;
    }

    public DefaultMutableTreeNode getRootnodeChosenScalarsTree() {
        return rootnodeChosenScalarsTree;
    }

    public DefaultMutableTreeNode getRootnodeChosenVectorsTree() {
        return rootnodeChosenVectorsTree;
    }

    public DefaultMutableTreeNode getRootnodeTreeBoundaries() {
        return rootnodeTreeBoundaries;
    }

    public DefaultMutableTreeNode getRootnodeTreeChosenGeometry() {
        return rootnodeTreeChosenGeometry;
    }

    public DefaultMutableTreeNode getRootnodeTreeScalars() {
        return rootnodeTreeScalars;
    }

    public DefaultMutableTreeNode getRootnodeTreeVectors() {
        return rootnodeTreeVectors;
    }

    

    //Setter
    public void setEnsightExportChosenGeometryTree(DefaultTreeModel EnsightExportChosenGeometryTree) {
        this.EnsightExport_ChosenGeometryTree_TreeModel = EnsightExportChosenGeometryTree;
    }

    public void setEnsightExportTabbedPaneRegionListModel(DefaultListModel<Object> EnsightExportTabbedPaneRegionListModel) {
        this.EnsightExportTabbedPaneRegionListModel = EnsightExportTabbedPaneRegionListModel;

    }

    public void setEnsightExportBoundariesTree(DefaultTreeModel EnsightExportBoundariesTree) {
        this.EnsightExport_BoundariesTree_TreeModel = EnsightExportBoundariesTree;
    }

    public void setEnsightExportScalarsTree(DefaultTreeModel EnsightExportScalarsTree) {
        this.EnsightExport_ScalarsTree_TreeModel = EnsightExportScalarsTree;
    }

    public void setEnsightExportVectorsTree(DefaultTreeModel EnsightExportVectorsTree) {
        this.EnsightExport_VectorsTree_TreeModel = EnsightExportVectorsTree;
    }

    public void setEnsightExportChosenScalarsTree(DefaultTreeModel EnsightExportChosenScalarsTree) {
        this.EnsightExport_ChosenScalarsTree_TreeModel = EnsightExportChosenScalarsTree;
    }

    public void setEnsightExportChosenVectorsTree(DefaultTreeModel EnsightExportChosenVectorsTree) {
        this.EnsightExport_ChosenVectorsTree_TreeModel = EnsightExportChosenVectorsTree;
    }

    //Initialisierung der Modelle des Views
    //Initialisierung der Knoten (DefaultMutableTreeNodes)
    private void initiateGeometryNodes() {
        for (Region r : PC.getSIM_DATA_MANGER().getAllRegionsList()) {
            ChosenGeometryTreeRegionsNodesHashMap.put(r, new DefaultMutableTreeNode(r));
        }
        for (Boundary b : PC.getSIM_DATA_MANGER().getAllBoundariesList()) {
            ChosenGeometryTreeBoundariesNodesHashMap.put(b, new DefaultMutableTreeNode(b, false));
        }

        for (Boundary b : PC.getSIM_DATA_MANGER().getAllBoundariesList()) {
            BoundariesTreeBoundariesNodesHashMap.put(b, new DefaultMutableTreeNode(b, false));
        }
        for (Region r : PC.getSIM_DATA_MANGER().getAllRegionsList()) {
            BoundariesTreeRegionsNodesHashMap.put(r, new DefaultMutableTreeNode(r));
        }


    }

    private void initiateFieldFunctionNodes() {
        //Scalars and Chosen Scalars Nodes

        //Scalars from Scalars
        for (FieldFunction FF : PC.getSIM_DATA_MANGER().getAllFieldFunctionsList()) {
            if (FF.getPresentationName().equals("Total Pressure")) {
                DefaultMutableTreeNode TotalPressureNode = new DefaultMutableTreeNode(FF, true);
                for (AbstractReferenceFrame ARF : PC.getSIM_DATA_MANGER().getReferenceFramesList()) {

                    DefaultMutableTreeNode TotalPressureInReferenceFrameNode = new DefaultMutableTreeNode(FF.getFunctionInReferenceFrame(ARF), false);
                    TotalPressureNode.add(TotalPressureInReferenceFrameNode);

                    ScalarsTreeScalarsNodesHashMap.put(FF.getFunctionInReferenceFrame(ARF),
                            TotalPressureInReferenceFrameNode);
                    ChosenScalarsTreeScalarsNodesHashMap.put(TotalPressureInReferenceFrameNode,
                            new DefaultMutableTreeNode(FF.getFunctionInReferenceFrame(ARF), false));
                }
                ScalarsTreeScalarsNodesHashMap.put(FF.getPresentationName(), TotalPressureNode);
                ChosenScalarsTreeScalarsNodesHashMap.put(TotalPressureNode, new DefaultMutableTreeNode(FF,
                        true));


            } else {
                DefaultMutableTreeNode tmpNode = new DefaultMutableTreeNode(FF, false);
                ScalarsTreeScalarsNodesHashMap.put(FF, tmpNode);
                ChosenScalarsTreeScalarsNodesHashMap.put(tmpNode, new DefaultMutableTreeNode(FF, false));



            }
        }

        //Scalars from Vectors
        for (FieldFunction FF : PC.getSIM_DATA_MANGER().getAllVectorFieldFunctionsList()) {
            DefaultMutableTreeNode VectorNode = new DefaultMutableTreeNode(FF, true);
            ScalarsTreeScalarsNodesHashMap.put(FF, VectorNode);
            ChosenScalarsTreeScalarsNodesHashMap.put(VectorNode, new DefaultMutableTreeNode(FF, true));
            if (FF.getPresentationName().equals(Configuration_Tool.FunctionName_Velocity)) {
                for (AbstractReferenceFrame ARF : PC.getSIM_DATA_MANGER().getReferenceFramesList()) {
                    DefaultMutableTreeNode VelocityReferenceFrameNode = new DefaultMutableTreeNode(ARF, true);
                    ScalarsTreeScalarsNodesHashMap.put(ARF, VelocityReferenceFrameNode);
                    ChosenScalarsTreeScalarsNodesHashMap.put(VelocityReferenceFrameNode, new DefaultMutableTreeNode(
                            ARF, true));
                    //Magnitude
                    DefaultMutableTreeNode tmpMagRefFNode = new DefaultMutableTreeNode(FF.getFunctionInReferenceFrame(ARF).getMagnitudeFunction(), false);
                    VelocityReferenceFrameNode.add(tmpMagRefFNode);
                    ScalarsTreeScalarsNodesHashMap.put(FF.getFunctionInReferenceFrame(
                            ARF).getMagnitudeFunction(), tmpMagRefFNode);
                    ChosenScalarsTreeScalarsNodesHashMap.put(tmpMagRefFNode, new DefaultMutableTreeNode(FF.getFunctionInReferenceFrame(ARF).getMagnitudeFunction(), false));

                    //Component Functions
                    for (CoordinateSystem CS : PC.getSIM_DATA_MANGER().getCoordinateSystemsList()) {
                        DefaultMutableTreeNode CoordinateSystemNode = new DefaultMutableTreeNode(CS, true);
                        ScalarsTreeScalarsNodesHashMap.put(CS, CoordinateSystemNode);
                        ChosenScalarsTreeScalarsNodesHashMap.put(CoordinateSystemNode, new DefaultMutableTreeNode(
                                CS, true));
                        for (int i = 0; i < 3; i++) {
                            DefaultMutableTreeNode tmp = new DefaultMutableTreeNode(FF.getFunctionInReferenceFrame(ARF).getFunctionInCoordinateSystem(CS).
                                    getComponentFunction(i), false);
                            ScalarsTreeScalarsNodesHashMap.put(FF.getFunctionInReferenceFrame(ARF).
                                    getFunctionInCoordinateSystem(CS).getComponentFunction(i), tmp);
                            ChosenScalarsTreeScalarsNodesHashMap.put(tmp, new DefaultMutableTreeNode(FF.getFunctionInReferenceFrame(ARF).getFunctionInCoordinateSystem(CS).
                                    getComponentFunction(i), false));

                            CoordinateSystemNode.add(tmp);
                        }
                        VelocityReferenceFrameNode.add(CoordinateSystemNode);

                    }
                    VectorNode.add(VelocityReferenceFrameNode);

                }

            } else {
                //Magnitude
                DefaultMutableTreeNode tmpMagnitudeNode = new DefaultMutableTreeNode(FF.getMagnitudeFunction(),
                        false);
                VectorNode.add(tmpMagnitudeNode);
                ScalarsTreeScalarsNodesHashMap.put(FF.getMagnitudeFunction(), tmpMagnitudeNode);
                ChosenScalarsTreeScalarsNodesHashMap.put(tmpMagnitudeNode, new DefaultMutableTreeNode(FF.getMagnitudeFunction(),
                        false));
                //Component Functions
                for (CoordinateSystem CS : PC.getSIM_DATA_MANGER().getCoordinateSystemsList()) {
                    DefaultMutableTreeNode CoordinateSystemNode = new DefaultMutableTreeNode(CS, true);
                    ScalarsTreeScalarsNodesHashMap.put(CS, CoordinateSystemNode);
                    ChosenScalarsTreeScalarsNodesHashMap.put(CoordinateSystemNode, new DefaultMutableTreeNode(
                            CS, true));
                    for (int i = 0; i < 3; i++) {
                        DefaultMutableTreeNode tmpNode = new DefaultMutableTreeNode(FF.getFunctionInCoordinateSystem(CS).getComponentFunction(i), false);
                        CoordinateSystemNode.add(tmpNode);
                        ScalarsTreeScalarsNodesHashMap.put(FF.getFunctionInCoordinateSystem(
                                CS).getComponentFunction(i), tmpNode);
                        ChosenScalarsTreeScalarsNodesHashMap.put(tmpNode, new DefaultMutableTreeNode(FF.getFunctionInCoordinateSystem(CS).getComponentFunction(i), false));
                    }
                    VectorNode.add(CoordinateSystemNode);
                }
            }
        }


        //Vectors
        for (FieldFunction FF : PC.getSIM_DATA_MANGER().getAllVectorFieldFunctionsList()) {
            if (!FF.getFunctionName().equals(Configuration_Tool.FunctionName_Velocity)) {
                DefaultMutableTreeNode tmpNode = new DefaultMutableTreeNode(FF, false);
                VectorsTreeVectorsNodesHashMap.put(FF, tmpNode);
                ChosenVectorsTreeVectorsNodesHashMap.put(tmpNode, new DefaultMutableTreeNode(FF, false));


            } else {
                DefaultMutableTreeNode VelocityNode = new DefaultMutableTreeNode(FF, true);
                for (AbstractReferenceFrame ARF : PC.getSIM_DATA_MANGER().getReferenceFramesList()) {
                    DefaultMutableTreeNode VelocityReferenceFrameNode = new DefaultMutableTreeNode(FF.getFunctionInReferenceFrame(ARF), false);
                    VelocityNode.add(VelocityReferenceFrameNode);
                    VectorsTreeVectorsNodesHashMap.put(FF.getFunctionInReferenceFrame(ARF),
                            VelocityReferenceFrameNode);
                    ChosenVectorsTreeVectorsNodesHashMap.put(VelocityReferenceFrameNode,
                            new DefaultMutableTreeNode(FF.getFunctionInReferenceFrame(ARF), false));
                    HashMap_VelocityFieldFuntion.put(FF.getFunctionInReferenceFrame(ARF), FF.getFunctionInReferenceFrame(ARF));
                }
                VectorsTreeVectorsNodesHashMap.put(FF.getPresentationName(), VelocityNode);
                ChosenVectorsTreeVectorsNodesHashMap.put(VelocityNode, new DefaultMutableTreeNode(FF, true));
            }
        }
    }

    //Initialisierung der TreeModels
    public void initiateViewDataModels() {
        this.initiate_ListModel_Regions_Tab();
        this.initiate_TreeModel_ChosenGeometry_Panel();
        this.initiate_TreeModel_Boundaries_Tab();
        this.initiate_TreeModel_Scalars_Tab();
        this.initiate_TreeModel_Vectors_Tab();
        this.initiate_TreeModel_ChosenScalars_Panel();
        this.initiate_TreeModel_ChosenVectors_Panel();
    }

    private void initiate_ListModel_Regions_Tab() {
        EnsightExportTabbedPaneRegionListModel = new DefaultListModel<Object>();
        for (Region r : PC.getSIM_DATA_MANGER().getAllRegionsList()) {
            EnsightExportTabbedPaneRegionListModel.addElement((Region) r);
        }
    }

    private void initiate_TreeModel_Boundaries_Tab() {

        rootnodeTreeBoundaries.add(nodeTreeBoundariesBoundarieswithoutRegionChosen);
        EnsightExport_BoundariesTree_TreeModel = new DefaultTreeModel(rootnodeTreeBoundaries);

        for (Boundary b : PC.getSIM_DATA_MANGER().getAllBoundariesList()) {
            nodeTreeBoundariesBoundarieswithoutRegionChosen.add(BoundariesTreeBoundariesNodesHashMap.get(b));
        }
    }

    private void initiate_TreeModel_Scalars_Tab() {

        EnsightExport_ScalarsTree_TreeModel = new DefaultTreeModel(rootnodeTreeScalars);
        for (Entry<Object, DefaultMutableTreeNode> E : ScalarsTreeScalarsNodesHashMap.entrySet()) {
            if (E.getValue().getParent() == null) {
                rootnodeTreeScalars.add(E.getValue());
            }
        }
    }

    private void initiate_TreeModel_Vectors_Tab() {

        EnsightExport_VectorsTree_TreeModel = new DefaultTreeModel(rootnodeTreeVectors);

        for (Entry<Object, DefaultMutableTreeNode> E : VectorsTreeVectorsNodesHashMap.entrySet()) {
            if (E.getValue().getParent() == null) {
                rootnodeTreeVectors.add(E.getValue());
            }
        }
    }

    private void initiate_TreeModel_ChosenGeometry_Panel() {
        EnsightExport_ChosenGeometryTree_TreeModel = new DefaultTreeModel(rootnodeTreeChosenGeometry);
    }

    private void initiate_TreeModel_ChosenScalars_Panel() {
        EnsightExport_ChosenScalarsTree_TreeModel = new DefaultTreeModel(rootnodeChosenScalarsTree);
    }

    private void initiate_TreeModel_ChosenVectors_Panel() {
        EnsightExport_ChosenVectorsTree_TreeModel = new DefaultTreeModel(rootnodeChosenVectorsTree);
    }
}
