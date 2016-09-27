package de.hlrs.starplugin.covise_net_generation;

import Main.PluginContainer;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct;
import de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards.controller.Controller_CuttingSurfaceCard;
import de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards.controller.Controller_CuttingSurfaceSeriesCard;
import de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards.controller.Controller_GeometryCard;
import de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards.controller.Controller_IsoSurfaceCard;
import de.hlrs.starplugin.gui.covise_net_generation.jpanel_construct_creator.typecards.controller.Controller_StreamlineCard;
import de.hlrs.starplugin.gui.covise_net_generation.JPanel_CoviseNetGeneration;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import javax.swing.filechooser.FileFilter;
import javax.swing.JFileChooser;
import javax.swing.event.TreeSelectionEvent;
import javax.swing.event.TreeSelectionListener;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreePath;
import javax.swing.tree.TreeSelectionModel;

/**
 *This class is the Controller of the CoviseNetGeneration JPanel
 * therefore it is built out if listeners added to the Swing Views
 *@author Weiss HLRS Stuttgart
 */
public class Controller_JPanel_CoviseNetGeneration {

    private ActionListener ActionListsner_Button_Browse;
    private PluginContainer PC;
    private JPanel_CoviseNetGeneration CNGJP;

    //Controller
    private TreeSelectionListener JTree_CreatedVisConstructs_Listener;
    private Controller_GeometryCard Contr_GeometryCard;
    private Controller_CuttingSurfaceCard Contr_CuttingSurfaceCard;
    private Controller_CuttingSurfaceSeriesCard Contr_CuttingSurfaceSeriesCard;
    private Controller_StreamlineCard Contr_StreamlineCard;
    private Controller_IsoSurfaceCard Contr_IsoSurfaceCard;
    private boolean enabled = true;

    public Controller_JPanel_CoviseNetGeneration(PluginContainer PC) {
        this.PC = PC;
        this.CNGJP = PC.getGUI().getJPanelMainContent().getJPanelCoviseNetGeneration();



        ActionListsner_Button_Browse = ButtonBrowseListener();
        CNGJP.getCoviseNetGenerationFolderPanel().getButtonBrowse().addActionListener(ActionListsner_Button_Browse);

        JTree_CreatedVisConstructs_Listener = JTree_CreatedVisConstructs_Listener();
        CNGJP.getjPanel_CreatedConstructsList().getJScrollPane_JTree_CreatedVisualizationConstructs().
                getJTree_createdConstructsList().getSelectionModel().addTreeSelectionListener(
                JTree_CreatedVisConstructs_Listener);


        Contr_GeometryCard = new Controller_GeometryCard(PC);
        Contr_CuttingSurfaceCard = new Controller_CuttingSurfaceCard(PC);
        Contr_CuttingSurfaceSeriesCard = new Controller_CuttingSurfaceSeriesCard(PC);
        Contr_StreamlineCard = new Controller_StreamlineCard(PC);
        Contr_IsoSurfaceCard = new Controller_IsoSurfaceCard(PC);
    }

    private ActionListener ButtonBrowseListener() {
        ActionListener AL = new ActionListener() {

            public void actionPerformed(ActionEvent e) {
                if (enabled) {
                    JFileChooser FileChooser = new JFileChooser();
                    FileChooser.setCurrentDirectory(PC.getCNGDMan().getExportPath());
                    FileChooser.setSelectedFile(PC.getCNGDMan().getExportPath());

                    FileChooser.setApproveButtonText("save");
                    FileChooser.setDialogTitle("Save Python Script");
                    FileChooser.setDialogType(FileChooser.SAVE_DIALOG);
                    FileFilter filter = new FileNameExtensionFilter("Python Skript (*.py)",
                            new String[]{"py"});
                    FileChooser.setFileFilter(filter);
                    int result = FileChooser.showDialog(CNGJP, "Select Destination");
                    if (result == JFileChooser.APPROVE_OPTION) {
                        //TODO (wrong file ?)

                        File test = FileChooser.getSelectedFile();
                        if (test.getAbsolutePath().endsWith(".py")) {
                            PC.getCNGDMan().setExportPath(FileChooser.getSelectedFile());
                        } else {
                            File FilewithSuffix = new File(test.getAbsolutePath() + ".py");
                            PC.getCNGDMan().setExportPath(FilewithSuffix);

                        }
                    }
                }
            }
        };
        return AL;

    }



    private TreeSelectionListener JTree_CreatedVisConstructs_Listener() {
        TreeSelectionListener TSL = new TreeSelectionListener() {

            public void valueChanged(TreeSelectionEvent e) {
                if (enabled) {

                    TreeSelectionModel tsm = (TreeSelectionModel) e.getSource();

                    TreePath SelectedPath = tsm.getSelectionPath();
                    if (SelectedPath != null) {
                        DefaultMutableTreeNode tmpNode = (DefaultMutableTreeNode) SelectedPath.getLastPathComponent();
                        String ConstructName = (String) tmpNode.getUserObject();
                        Construct Con = PC.getCNGDMan().getConMan().getConstructList().get(ConstructName);

                        PC.getCNGDMan().getConMan().setSelectedConstruct(Con);


                    } else {
                        PC.getCNGDMan().getConMan().setSelectedConstruct(null);
                    }
                }
            }
        };
        return TSL;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
        this.Contr_CuttingSurfaceCard.setEnabled(enabled);
        this.Contr_CuttingSurfaceSeriesCard.setEnabled(enabled);
        this.Contr_GeometryCard.setEnabled(enabled);
        this.Contr_IsoSurfaceCard.setEnabled(enabled);
        this.Contr_StreamlineCard.setEnabled(enabled);
    }

    public Controller_CuttingSurfaceCard getContr_CuttingSurfaceCard() {
        return Contr_CuttingSurfaceCard;
    }

    public Controller_CuttingSurfaceSeriesCard getContr_CuttingSurfaceSeriesCard() {
        return Contr_CuttingSurfaceSeriesCard;
    }

    public Controller_GeometryCard getContr_GeometryCard() {
        return Contr_GeometryCard;
    }

    public Controller_IsoSurfaceCard getContr_IsoSurfaceCard() {
        return Contr_IsoSurfaceCard;
    }

    public Controller_StreamlineCard getContr_StreamlineCard() {
        return Contr_StreamlineCard;
    }
}

