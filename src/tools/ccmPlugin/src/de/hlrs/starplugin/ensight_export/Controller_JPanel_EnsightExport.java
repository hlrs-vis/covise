package de.hlrs.starplugin.ensight_export;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.io.File;
import star.common.Region;
import star.common.Boundary;
import java.util.ArrayList;
import javax.swing.JFileChooser;
import javax.swing.ListSelectionModel;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import javax.swing.event.TreeSelectionEvent;
import javax.swing.event.TreeSelectionListener;
import javax.swing.filechooser.FileFilter;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreePath;
import javax.swing.tree.TreeSelectionModel;
import Main.PluginContainer;
import star.common.FieldFunction;

/**
 * This class is the Controller of the Ensight Export JPanel
 * therefore it is built out if listeners added to the Swing Views
 *  @author Weiss HLRS Stuttgart
 */
public class Controller_JPanel_EnsightExport {

    private PluginContainer PC;
    private ListSelectionListener RegionsListSelectionListener;
    private TreeSelectionListener BoundariesTreeSelectionListener;
    private TreeSelectionListener ScalarsTreeSelectionListener;
    private TreeSelectionListener VectorsTreeSelectionListener;
    private ActionListener ButtonBrowseListener;
    private ItemListener CheckBox_AppendtoFileListener;
    private ItemListener CheckBox_ExportResultsOnVerticesListener;
    private boolean enabled = true;

    public Controller_JPanel_EnsightExport(PluginContainer Pc) {
        this.PC = Pc;

        PC.getEEDMan();
        RegionsListSelectionListener = RegionsListSelectionListener();
        BoundariesTreeSelectionListener = BoundariesTreeSelectionListener();
        ScalarsTreeSelectionListener = ScalarsTreeSelectionListener();
        VectorsTreeSelectionListener = VectorsTreeSelectionListener();
        ButtonBrowseListener = ButtonBrowseListener();
        CheckBox_AppendtoFileListener = CheckBoxAppendtoFileListener();
        CheckBox_ExportResultsOnVerticesListener = this.CheckBox_ExportonVerticesListener();



        this.PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportTabbedPane().getJPanelRegions().getRegionList().getSelectionModel().
                addListSelectionListener(
                RegionsListSelectionListener);
        this.PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportTabbedPane().getJPanelBoundaries().getBoundaryTree().getSelectionModel().
                addTreeSelectionListener(BoundariesTreeSelectionListener);
        this.PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportTabbedPane().getJPanelScalars().getScalarFieldFunctionTree().getSelectionModel().
                addTreeSelectionListener(ScalarsTreeSelectionListener);
        this.PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportTabbedPane().getJPanelVectors().getVectorFieldFunctionTree().getSelectionModel().
                addTreeSelectionListener(VectorsTreeSelectionListener);
        this.PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportFolderandOptionsPanel().getButtonBrowse().addActionListener(ButtonBrowseListener);
        this.PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportFolderandOptionsPanel().getCheckBoxAppendFile().addItemListener(
                CheckBox_AppendtoFileListener);
        this.PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportFolderandOptionsPanel().getCheckBox_ResultsOnVertices().addItemListener(
                CheckBox_ExportResultsOnVerticesListener);


    }

    private ListSelectionListener RegionsListSelectionListener() {
        ListSelectionListener LSL = new ListSelectionListener() {

            public void valueChanged(ListSelectionEvent e) {
                if (enabled) {
                    ListSelectionModel lsm = (ListSelectionModel) e.getSource();

                    if (e.getValueIsAdjusting() == false) {
                        if (lsm.isSelectionEmpty()) {
                            PC.getEEDMan().setSelectedRegions(new ArrayList<Region>());

                        } else {
                            // Find out which indexes are selected.

                            int minIndex = lsm.getMinSelectionIndex();
                            int maxIndex = lsm.getMaxSelectionIndex();
                            ArrayList<Region> tmpSelectedRegionList = new ArrayList<Region>();
                            for (int i = minIndex; i <= maxIndex; i++) {
                                if (lsm.isSelectedIndex(i)) {
                                    tmpSelectedRegionList.add((Region) PC.getEEVMC().getEnsightExportTabbedPaneRegionListModel().get(i));
                                }
                            }
                            PC.getEEDMan().setSelectedRegions(tmpSelectedRegionList);


                        }
                    }
                }
            }
        };
        return LSL;
    }

    private TreeSelectionListener BoundariesTreeSelectionListener() {
        TreeSelectionListener TSL = new TreeSelectionListener() {

            public void valueChanged(TreeSelectionEvent e) {
                if (enabled) {
                    if (PC.getEEDMan().isBoundarySelectionChangeable()) {
                        TreeSelectionModel tsm = (TreeSelectionModel) e.getSource();
                        TreePath[] SelectedPaths = tsm.getSelectionPaths();
                        ArrayList<Boundary> tmpSelectedBoundaryList = new ArrayList<Boundary>();
                        for (TreePath t : SelectedPaths) {
                            DefaultMutableTreeNode tmpNode = (DefaultMutableTreeNode) t.getLastPathComponent();
                            if (tmpNode.getUserObject() instanceof Boundary) {
                                tmpSelectedBoundaryList.add((Boundary) tmpNode.getUserObject());
                            }
                        }
                        PC.getEEDMan().setSelectedBoundaries(tmpSelectedBoundaryList);


                    }
                }
            }
        };
        return TSL;
    }

    private TreeSelectionListener ScalarsTreeSelectionListener() {

        TreeSelectionListener TSL = new TreeSelectionListener() {

            public void valueChanged(TreeSelectionEvent e) {
                if (enabled) {

                    TreeSelectionModel tsm = (TreeSelectionModel) e.getSource();
                    PC.getEEDMan().getScalarsSelectedList().clear();
                    TreePath[] SelectedPaths = tsm.getSelectionPaths();

                    ArrayList<FieldFunction> SelectedFunctions = new ArrayList<FieldFunction>();
                    for (TreePath t : SelectedPaths) {
                        DefaultMutableTreeNode tmpNode = (DefaultMutableTreeNode) t.getLastPathComponent();
                        if (!(tmpNode.getChildCount() > 0)) {
                            if (tmpNode.getUserObject() instanceof FieldFunction) {
                                SelectedFunctions.add((FieldFunction) tmpNode.getUserObject());
                            }
                        }
                    }
                    PC.getEEDMan().setScalarFieldFunctionSelection(SelectedFunctions);

                }
            }
        };
        return TSL;
    }

    private TreeSelectionListener VectorsTreeSelectionListener() {

        TreeSelectionListener TSL = new TreeSelectionListener() {

            public void valueChanged(TreeSelectionEvent e) {
                if (enabled) {
                    TreeSelectionModel tsm = (TreeSelectionModel) e.getSource();
                    PC.getEEDMan().getVectorsSelectedList().clear();
                    TreePath[] SelectedPaths = tsm.getSelectionPaths();

                    ArrayList<FieldFunction> SelectedFunctions = new ArrayList<FieldFunction>();
                    for (TreePath t : SelectedPaths) {
                        DefaultMutableTreeNode tmpNode = (DefaultMutableTreeNode) t.getLastPathComponent();
                        if (!(tmpNode.getChildCount() > 0)) {
                            if (tmpNode.getUserObject() instanceof FieldFunction) {
                                SelectedFunctions.add((FieldFunction) tmpNode.getUserObject());
                            }
                        }
                    }
                    PC.getEEDMan().setVectorFieldFunctionSelection(SelectedFunctions);

                }
            }
        };
        return TSL;
    }

    public ListSelectionListener getRegionsListSelectionListener() {
        return RegionsListSelectionListener;
    }

    private ActionListener ButtonBrowseListener() {
        ActionListener AL = new ActionListener() {

            public void actionPerformed(ActionEvent e) {
                if (enabled) {
                    JFileChooser FileChosser = new JFileChooser();

                    File die = PC.getEEDMan().getExportPath();
                    FileChosser.setCurrentDirectory(die);
                    FileChosser.setApproveButtonText("save");
                    FileChosser.setDialogTitle("Save Ensight Export to");
                    FileChosser.setDialogType(FileChosser.SAVE_DIALOG);
                    FileChosser.setSelectedFile(die);
                    FileFilter filter = new FileNameExtensionFilter("Ensight Gold file (*.case)",
                            new String[]{"case"});
                    FileChosser.setFileFilter(filter);
                    int result = FileChosser.showDialog(PC.getGUI().getJPanelMainContent().getEnsightExportJpanel(), "Select Destination");
                    if (result == JFileChooser.APPROVE_OPTION) {
                        //TODO (Fehleingabe?)

                        File test = FileChosser.getSelectedFile();
                        if (test.getAbsolutePath().endsWith(".case")) {
                            PC.getEEDMan().setExportPath(FileChosser.getSelectedFile());
                        } else {
                            File FilewithSuffix = new File(test.getAbsolutePath() + ".case");
                            PC.getEEDMan().setExportPath(FilewithSuffix);
                        }
                    }
                }
            }
        };
        return AL;

    }

    private ItemListener CheckBoxAppendtoFileListener() {
        ItemListener IL = new ItemListener() {

            public void itemStateChanged(ItemEvent e) {
                if (enabled) {
                    PC.getEEDMan().setAppendtoFile(PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportFolderandOptionsPanel().getCheckBoxAppendFile().
                            isSelected());
                }
            }
        };
        return IL;
    }

    private ItemListener CheckBox_ExportonVerticesListener() {
        ItemListener IL = new ItemListener() {

            public void itemStateChanged(ItemEvent e) {
                if (enabled) {
                    PC.getEEDMan().setExportonVertices(PC.getGUI().getJPanelMainContent().getEnsightExportJpanel().getEnsightExportFolderandOptionsPanel().
                            getCheckBox_ResultsOnVertices().
                            isSelected());
                }
            }
        };
        return IL;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }
}


