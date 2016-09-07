package de.hlrs.starplugin.ensight_export;

import de.hlrs.starplugin.interfaces.Interface_EnsightExport_DataChangedListener;
import Main.PluginContainer;

import de.hlrs.starplugin.util.JTreeExpansion;
import de.hlrs.starplugin.util.SortJTree;
import java.util.ArrayList;
import star.common.Region;
import star.common.Boundary;
import java.util.Map.Entry;
import javax.swing.JList;
import javax.swing.JTree;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreePath;
import star.common.FieldFunction;

/**
 *Maps the Data from the EnsightExport_DataManager to the GUI
 * and adds the Swing Data Models to the Swing View of the EnsightExport
 * @author Weiss HLRS Stuttgart
 */
public class JPanel_EnsightExport_DataMapper implements Interface_EnsightExport_DataChangedListener {

    private PluginContainer PC;
    private EnsightExport_DataManager EEDMan;

    public JPanel_EnsightExport_DataMapper(PluginContainer PC) {

        this.PC = PC;
        this.EEDMan = this.PC.getEEDMan();


        EEDMan.addListener(this);
        PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportFolderandOptionsPanel().getLabelDestinationFile().setText(EEDMan.getExportPath().
                getAbsolutePath());
        PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportFolderandOptionsPanel().getCheckBoxAppendFile().setSelected(
                EEDMan.isAppendtoFile());
        addModelsToViews();
    }

    private void addModelsToViews() {
        //Regions Tab
        PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportTabbedPane().getJPanelRegions().getRegionList().setModel(PC.getEEVMC().
                getEnsightExportTabbedPaneRegionListModel());

        //Boundaries Tab
        PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportTabbedPane().getJPanelBoundaries().getBoundaryTree().setModel(PC.getEEVMC().getEnsightExport_BoundariesTree_TreeModel());
        JTreeExpansion.expandAllNodes(PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportTabbedPane().getJPanelBoundaries().getBoundaryTree());

        //Scalars Tab
        PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportTabbedPane().getJPanelScalars().getScalarFieldFunctionTree().setModel(
                PC.getEEVMC().getEnsightExport_ScalarsTree_TreeModel());
        SortJTree.sortTree((DefaultMutableTreeNode) PC.getEEVMC().getEnsightExport_ScalarsTree_TreeModel().getRoot());
        PC.getEEVMC().getEnsightExport_ScalarsTree_TreeModel().reload();//.nodeStructureChanged(rootnodeTreeScalars);
        JTreeExpansion.expandAllNodes(PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportTabbedPane().getJPanelScalars().
                getScalarFieldFunctionTree());

        //Vectors Tab
        PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportTabbedPane().getJPanelVectors().getVectorFieldFunctionTree().setModel(
                PC.getEEVMC().getEnsightExport_VectorsTree_TreeMdoel());
        SortJTree.sortTree((DefaultMutableTreeNode) PC.getEEVMC().getEnsightExport_VectorsTree_TreeMdoel().getRoot());
        PC.getEEVMC().getEnsightExport_VectorsTree_TreeMdoel().reload();//VectorsTreeModel.nodeStructureChanged(rootnodeTreeVectors);
        JTreeExpansion.expandAllNodes(PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportTabbedPane().getJPanelVectors().
                getVectorFieldFunctionTree());

        //Chosen Geometry Panel
        PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportChosenGeometryPanel().getEnsightExpotChosenGeometryScrollPane().
                getBaumChosenGeometry().setModel(PC.getEEVMC().getEnsightExport_ChosenGeometryTree_TreeModel());

        //Chosen Scalars Panel
        PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportChosenScalarsPanel().getEnsightExportChosenScalarsPanelScrollPane().
                getBaumChosenScalars().setModel(PC.getEEVMC().getEnsightExport_ChosenScalarsTree_TreeModel());

        //Chosen Vectors Panel
        PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportChosenVectorsPanel().getEnsightExportChosenVectorsPanelScrollPane().
                getBaumChosenVectors().setModel(PC.getEEVMC().getEnsightExport_ChosenVectorsTree_TreeModel());
    }

    public void RegionSelectionChanged() {
        updateBoundariesTree();

    }

    public void BoundarySelectionChanged() {
        updateChosenGeometryTree();
    }

    public void ScalarsSelectionChanged() {
        updateChosenScalarsTree();
    }

    public void VectorsSelectionChanged() {
        updateChosenVectorsTree();
    }

    public void EnsightExportPathChanged() {
        this.PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportFolderandOptionsPanel().getLabelDestinationFile().setText(this.EEDMan.getExportPath().getAbsolutePath());
    }

    public void AppendToExistingFileChanged() {
        this.PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportFolderandOptionsPanel().getCheckBoxAppendFile().setSelected(this.EEDMan.isAppendtoFile());
    }

    public void ExportonVerticesChangedChanged() {
        this.PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportFolderandOptionsPanel().getCheckBox_ResultsOnVertices().setSelected(this.EEDMan.isAppendtoFile());
    }

    private void updateBoundariesTree() {

        EEDMan.setBoundarySelectionChangeable(false);
        for (Region r : EEDMan.getRegionsSelectedList()) {
            PC.getEEVMC().getEnsightExport_BoundariesTree_TreeModel().insertNodeInto(
                    PC.getEEVMC().getBoundariesTreeRegionsNodesHashMap().get(r),
                    (DefaultMutableTreeNode) PC.getEEVMC().getEnsightExport_BoundariesTree_TreeModel().getRoot(), 0);

            for (Boundary b : r.getBoundaryManager().getBoundaries()) {
                PC.getEEVMC().getEnsightExport_BoundariesTree_TreeModel().insertNodeInto(PC.getEEVMC().getBoundariesTreeBoundariesNodesHashMap().get(b),
                        PC.getEEVMC().getBoundariesTreeRegionsNodesHashMap().get(r), 0);
            }

        }
        for (Region r : PC.getSIM_DATA_MANGER().getAllRegionsList()) {
            if (!EEDMan.getRegionsSelectedList().contains(r)) {
                for (Boundary b : r.getBoundaryManager().getBoundaries()) {
                    PC.getEEVMC().getEnsightExport_BoundariesTree_TreeModel().insertNodeInto(PC.getEEVMC().getBoundariesTreeBoundariesNodesHashMap().get(b),
                            PC.getEEVMC().getNodeTreeBoundariesBoundarieswithoutRegionChosen(), 0);
                }

                if (PC.getEEVMC().getBoundariesTreeRegionsNodesHashMap().get(r).getParent() != null) {
                    PC.getEEVMC().getEnsightExport_BoundariesTree_TreeModel().removeNodeFromParent(PC.getEEVMC().getBoundariesTreeRegionsNodesHashMap().get(r));
                }
            }
        }
        PC.getEEVMC().getEnsightExport_BoundariesTree_TreeModel().nodeStructureChanged(PC.getEEVMC().getRootnodeTreeBoundaries());
        //expandAllNodes(PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportTabbedPane().getJPanelBoundaries().getBoundaryTree());

        TreePath[] Selected = new TreePath[EEDMan.getBoundariesSelectedList().size()];
        for (int i = 0; i < EEDMan.getBoundariesSelectedList().size(); i++) {
            Selected[i] = new TreePath(PC.getEEVMC().getBoundariesTreeBoundariesNodesHashMap().get(EEDMan.getBoundariesSelectedList().get(i)).getPath());
        }
        EEDMan.setBoundarySelectionChangeable(true);
        PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportTabbedPane().getJPanelBoundaries().getBoundaryTree().getSelectionModel().
                addSelectionPaths(Selected);
        updateChosenGeometryTree();

    }

    private void updateChosenGeometryTree() {
        PC.getEEVMC().getRootnodeTreeChosenGeometry().removeAllChildren();
        PC.getEEVMC().getNodeTreeChosenGeometryBoundarieswithoutRegionChosen().removeAllChildren();
        PC.getEEVMC().getRootnodeTreeChosenGeometry().add(PC.getEEVMC().getNodeTreeChosenGeometryBoundarieswithoutRegionChosen());
        for (Region R : EEDMan.getRegionsSelectedList()) {
            PC.getEEVMC().getRootnodeTreeChosenGeometry().add(PC.getEEVMC().getChosenGeometryTreeRegionsNodesHashMap().get(R));
            PC.getEEVMC().getChosenGeometryTreeRegionsNodesHashMap().get(R).removeAllChildren();

        }
        for (Boundary B : EEDMan.getBoundariesSelectedList()) {
            if (EEDMan.getRegionsSelectedList().contains(B.getRegion())) {
                PC.getEEVMC().getChosenGeometryTreeRegionsNodesHashMap().get(B.getRegion()).add(PC.getEEVMC().getChosenGeometryTreeBoundariesNodesHashMap().get(B));
            } else {
                PC.getEEVMC().getNodeTreeChosenGeometryBoundarieswithoutRegionChosen().add(PC.getEEVMC().getChosenGeometryTreeBoundariesNodesHashMap().get(B));
            }
        }
        if (PC.getEEVMC().getNodeTreeChosenGeometryBoundarieswithoutRegionChosen().getChildCount() < 1) {
            PC.getEEVMC().getRootnodeTreeChosenGeometry().remove(PC.getEEVMC().getNodeTreeChosenGeometryBoundarieswithoutRegionChosen());
        }
        SortJTree.sortTree(PC.getEEVMC().getRootnodeTreeChosenGeometry());

        this.PC.disableEventManager();
        PC.getEEVMC().getEnsightExport_ChosenGeometryTree_TreeModel().nodeStructureChanged(
                PC.getEEVMC().getRootnodeTreeChosenGeometry());
        this.PC.enableEventManager();

        JTreeExpansion.expandAllNodes(PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportChosenGeometryPanel().
                getEnsightExpotChosenGeometryScrollPane().
                getBaumChosenGeometry());
        checkRegionSelection();
        checkBoundarySelection();

    }

    private void updateChosenScalarsTree() {
        PC.getEEVMC().getRootnodeChosenScalarsTree().removeAllChildren();
        for (Entry<DefaultMutableTreeNode, DefaultMutableTreeNode> r : PC.getEEVMC().getChosenScalarsTreeScalarsNodesHashMap().entrySet()) {
            r.getValue().removeAllChildren();
        }
        for (FieldFunction FF : EEDMan.getScalarsSelectedList()) {
            addtoChosenScalarsTree(PC.getEEVMC().getScalarsTreeScalarsNodesHashMap().get(FF));
        }

        SortJTree.sortTree(PC.getEEVMC().getRootnodeChosenScalarsTree());

        PC.getEEVMC().getEnsightExport_ChosenScalarsTree_TreeModel().nodeStructureChanged(PC.getEEVMC().getRootnodeChosenScalarsTree());
        JTreeExpansion.expandAllNodes(PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportChosenScalarsPanel().
                getEnsightExportChosenScalarsPanelScrollPane().
                getBaumChosenScalars());
        checkScalarSelection();
    }

    private void updateChosenVectorsTree() {
        PC.getEEVMC().getRootnodeChosenVectorsTree().removeAllChildren();
        for (Entry<DefaultMutableTreeNode, DefaultMutableTreeNode> r : PC.getEEVMC().getChosenVectorsTreeVectorsNodesHashMap().entrySet()) {
            r.getValue().removeAllChildren();
        }
        for (FieldFunction FF : EEDMan.getVectorsSelectedList()) {
            addtoChosenVectorsTree(PC.getEEVMC().getVectorsTreeVectorsNodesHashMap().get(FF));
        }
        SortJTree.sortTree(PC.getEEVMC().getRootnodeChosenVectorsTree());
        PC.getEEVMC().getEnsightExport_ChosenVectorsTree_TreeModel().nodeStructureChanged(PC.getEEVMC().getRootnodeChosenVectorsTree());
        JTreeExpansion.expandAllNodes(PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportChosenVectorsPanel().
                getEnsightExportChosenVectorsPanelScrollPane().
                getBaumChosenVectors());
        checkVectorSelection();
    }

    private void addtoChosenScalarsTree(DefaultMutableTreeNode DMTN) {
        if (((DefaultMutableTreeNode) DMTN.getParent()).isRoot()) {
            PC.getEEVMC().getRootnodeChosenScalarsTree().add(PC.getEEVMC().getChosenScalarsTreeScalarsNodesHashMap().get(DMTN));
        } else {
            PC.getEEVMC().getChosenScalarsTreeScalarsNodesHashMap().get((DefaultMutableTreeNode) DMTN.getParent()).add(PC.getEEVMC().getChosenScalarsTreeScalarsNodesHashMap().get(DMTN));
            if (!PC.getEEVMC().getRootnodeChosenScalarsTree().isNodeDescendant(PC.getEEVMC().getChosenScalarsTreeScalarsNodesHashMap().get(
                    DMTN))) {
                this.addtoChosenScalarsTree((DefaultMutableTreeNode) DMTN.getParent());
            }
        }

    }

    private void addtoChosenVectorsTree(DefaultMutableTreeNode DMTN) {
        if (((DefaultMutableTreeNode) DMTN.getParent()).isRoot()) {
            DefaultMutableTreeNode a = PC.getEEVMC().getChosenVectorsTreeVectorsNodesHashMap().get(DMTN);
            PC.getEEVMC().getRootnodeChosenVectorsTree().add(PC.getEEVMC().getChosenVectorsTreeVectorsNodesHashMap().get(DMTN));
        } else {
            DefaultMutableTreeNode a = PC.getEEVMC().getChosenVectorsTreeVectorsNodesHashMap().get((DefaultMutableTreeNode) DMTN.getParent());
            PC.getEEVMC().getChosenVectorsTreeVectorsNodesHashMap().get((DefaultMutableTreeNode) DMTN.getParent()).add(PC.getEEVMC().getChosenVectorsTreeVectorsNodesHashMap().get(DMTN));
            if (!PC.getEEVMC().getRootnodeChosenVectorsTree().isNodeDescendant(PC.getEEVMC().getChosenVectorsTreeVectorsNodesHashMap().get(
                    DMTN))) {
                this.addtoChosenVectorsTree((DefaultMutableTreeNode) DMTN.getParent());
            }
        }
    }

    private void checkRegionSelection() {

        JList RegionList = this.PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().
                getEnsightExportTabbedPane().
                getJPanelRegions().getRegionList();
        // Get all selected Objects
        int[] selectedIndices = RegionList.getSelectedIndices();
        ArrayList<Region> SelectedRegions = new ArrayList();
        for (int i = 0; i < selectedIndices.length; i++) {
            SelectedRegions.add((Region) RegionList.getModel().getElementAt(selectedIndices[i]));
        }
        //In der Gui Selectierte aber nicht im Modell Selectierte Objects aus der Gui Selection entfernen
        ArrayList<Region> ToRemove = new ArrayList<Region>();
        for (Region r : SelectedRegions) {
            if (!this.PC.getEEDMan().getRegionsSelectedList().contains(r)) {
                ToRemove.add(r);
            }
        }
        SelectedRegions.removeAll(ToRemove);


        //Alle Regions die im Modell Selectiert sind aber nicht in der Gui zur Gui Selection Hinzufügen
        for (Region r : this.PC.getEEDMan().getRegionsSelectedList()) {
            if (PC.getEEVMC().getEnsightExportTabbedPaneRegionListModel().contains(r)) {
                if (!SelectedRegions.contains(r)) {
                    SelectedRegions.add(r);
                }
            }
        }
        //Sammeln aller zu Selectierenden Indizes
        int[] i = new int[SelectedRegions.size()];
        for (int ii = 0; ii < SelectedRegions.size(); ii++) {
            i[ii] = PC.getEEVMC().getEnsightExportTabbedPaneRegionListModel().indexOf(SelectedRegions.get(ii));
        }
        //Disable Event Manager and correct the Selection

        PC.disableEventManager();
        RegionList.setSelectedIndices(i);
        PC.enableEventManager();

    }

    private void checkBoundarySelection() {
        JTree BoundaryTree = this.PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().
                getEnsightExportTabbedPane().getJPanelBoundaries().getBoundaryTree();
        //Get all selected Objects
        TreePath[] SelectedNodes = BoundaryTree.getSelectionModel().getSelectionPaths();
        ArrayList<Boundary> SelectedBoundaryList = new ArrayList<Boundary>();
        for (TreePath t : SelectedNodes) {
            DefaultMutableTreeNode tmpNode = (DefaultMutableTreeNode) t.getLastPathComponent();
            if (tmpNode.getUserObject() instanceof Boundary) {
                SelectedBoundaryList.add((Boundary) tmpNode.getUserObject());
            }
        }
        //In der Gui Selectierte aber nicht im Modell Selectierte Objects aus der Gui Selection entfernen
        ArrayList<Boundary> ToRemove = new ArrayList<Boundary>();
        for (Boundary b : SelectedBoundaryList) {
            if (!this.PC.getEEDMan().getBoundariesSelectedList().contains(b)) {
                ToRemove.add(b);
            }
        }
        SelectedBoundaryList.removeAll(ToRemove);

        //Alle Boundaries die im Modell Selectiert sind aber nicht in der Gui zur Gui Selection Hinzufügen

        for (Boundary b : this.PC.getEEDMan().getBoundariesSelectedList()) {
            if (PC.getEEVMC().getBoundariesTreeBoundariesNodesHashMap().containsKey(b)) {
                if (!SelectedBoundaryList.contains(b)) {
                    SelectedBoundaryList.add(b);
                }
            }
        }
        //Sammeln aller zu Selectierenden Paths
        TreePath[] TP = new TreePath[SelectedBoundaryList.size()];
        for (int i = 0; i < SelectedBoundaryList.size(); i++) {
            TP[i] = new TreePath(
                    PC.getEEVMC().getBoundariesTreeBoundariesNodesHashMap().get(SelectedBoundaryList.get(i)).getPath());
        }
        //Disable Event Manager and correct the Selection
        PC.disableEventManager();
        BoundaryTree.getSelectionModel().setSelectionPaths(TP);
        PC.enableEventManager();
    }

    private void checkScalarSelection() {
        JTree ScalarTree = this.PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().
                getEnsightExportTabbedPane().getJPanelScalars().getScalarFieldFunctionTree();
        //Get all selected Objects
        TreePath[] SelectedNodes = ScalarTree.getSelectionModel().getSelectionPaths();
        ArrayList<FieldFunction> SelectedFieldFunctionList = new ArrayList<FieldFunction>();
        for (TreePath t : SelectedNodes) {
            DefaultMutableTreeNode tmpNode = (DefaultMutableTreeNode) t.getLastPathComponent();
            if (tmpNode.getUserObject() instanceof FieldFunction) {
                SelectedFieldFunctionList.add((FieldFunction) tmpNode.getUserObject());
            }
        }
        //In der Gui Selectierte aber nicht im Modell Selectierte Objects aus der Gui Selection entfernen
        ArrayList<FieldFunction> ToRemove = new ArrayList<FieldFunction>();
        for (FieldFunction FF : SelectedFieldFunctionList) {
            if (!this.PC.getEEDMan().getScalarsSelectedList().contains(FF)) {
                ToRemove.add(FF);
            }
        }
        SelectedFieldFunctionList.removeAll(ToRemove);

        //Alle Boundaries die im Modell Selectiert sind aber nicht in der Gui zur Gui Selection Hinzufügen

        for (FieldFunction FF : this.PC.getEEDMan().getScalarsSelectedList()) {
            if (PC.getEEVMC().getScalarsTreeScalarsNodesHashMap().containsKey(FF)) {
                if (!SelectedFieldFunctionList.contains(FF)) {
                    SelectedFieldFunctionList.add(FF);
                }
            }
        }
        //Sammeln aller zu Selectierenden Pfade
        TreePath[] TP = new TreePath[SelectedFieldFunctionList.size()];
        for (int i = 0; i < SelectedFieldFunctionList.size(); i++) {
            TP[i] = new TreePath(
                    PC.getEEVMC().getScalarsTreeScalarsNodesHashMap().get(SelectedFieldFunctionList.get(i)).getPath());
        }
        //Disable Event Manager and correct the Selection
        PC.disableEventManager();
        ScalarTree.getSelectionModel().setSelectionPaths(TP);
        PC.enableEventManager();
    }

    private void checkVectorSelection() {
        JTree VectorTree = this.PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().
                getEnsightExportTabbedPane().getJPanelVectors().getVectorFieldFunctionTree();
        //Get all selected Objects
        TreePath[] SelectedNodes = VectorTree.getSelectionModel().getSelectionPaths();
        ArrayList<FieldFunction> SelectedFieldFunctionList = new ArrayList<FieldFunction>();
        for (TreePath t : SelectedNodes) {
            DefaultMutableTreeNode tmpNode = (DefaultMutableTreeNode) t.getLastPathComponent();
            if (tmpNode.getUserObject() instanceof FieldFunction) {
                SelectedFieldFunctionList.add((FieldFunction) tmpNode.getUserObject());
            }
        }
        //In der Gui Selectierte aber nicht im Modell Selectierte Objects aus der Gui Selection entfernen
        ArrayList<FieldFunction> ToRemove = new ArrayList<FieldFunction>();
        for (FieldFunction FF : SelectedFieldFunctionList) {
            if (!this.PC.getEEDMan().getVectorsSelectedList().contains(FF)) {
                ToRemove.add(FF);
            }
        }
        SelectedFieldFunctionList.removeAll(ToRemove);

        //Alle Vectors die im Modell Selectiert sind aber nicht in der Gui zur Gui Selection Hinzufügen
        for (FieldFunction FF : this.PC.getEEDMan().getVectorsSelectedList()) {
            if (PC.getEEVMC().getVectorsTreeVectorsNodesHashMap().containsKey(FF)) {
                if (!SelectedFieldFunctionList.contains(FF)) {
                    SelectedFieldFunctionList.add(FF);
                }
            }
        }
        //Sammeln aller zu Selectierenden Pfade
        TreePath[] TP = new TreePath[SelectedFieldFunctionList.size()];
        for (int i = 0; i < SelectedFieldFunctionList.size(); i++) {
            TP[i] = new TreePath(
                    PC.getEEVMC().getVectorsTreeVectorsNodesHashMap().get(SelectedFieldFunctionList.get(i)).getPath());
        }
        //Disable Event Manager and correct the Selection
        PC.disableEventManager();
        VectorTree.getSelectionModel().setSelectionPaths(TP);
        PC.enableEventManager();
    }
}

