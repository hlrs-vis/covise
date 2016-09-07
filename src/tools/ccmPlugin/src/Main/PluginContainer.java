package Main;

import de.hlrs.starplugin.configuration.Configuration_GUI_Strings;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct;
import de.hlrs.starplugin.covise_net_generation.CoviseNetGeneration_DataManager;
import de.hlrs.starplugin.covise_net_generation.JPanel_CoviseNetGeneration_ViewModelContainer;
import de.hlrs.starplugin.covise_net_generation.JPanel_CoviseNetGeneration_DataMapper;
import de.hlrs.starplugin.covise_net_generation.Controller_JPanel_CoviseNetGeneration;
import de.hlrs.starplugin.ensight_export.JPanel_EnsightExport_ViewModelContainer;
import de.hlrs.starplugin.ensight_export.EnsightExport_DataManager;
import de.hlrs.starplugin.ensight_export.JPanel_EnsightExport_DataMapper;
import de.hlrs.starplugin.ensight_export.Controller_JPanel_EnsightExport;
import de.hlrs.starplugin.load_save.DataContainer;
import de.hlrs.starplugin.gui.listener.Manager_Controller;
import de.hlrs.starplugin.gui.covise_net_generation.listener.Manager_Controller_CoviseNetGeneration;
import de.hlrs.starplugin.gui.JFrame_MainFrame;
import de.hlrs.starplugin.gui.dialogs.Error_Dialog;
import java.io.StringWriter;
import star.common.Simulation;

/**
 *This ClasssContains the Main Classes as Attributes and
 * is the DataBridge to those classes
 * @author Weiss HLRS Stuttgart
 */
public class PluginContainer {

    //Simulation
    final private Simulation SIM;
    final private SimDataManager SIM_DATA_MANGER;
    //GUI
    private JFrame_MainFrame GUI;
    //EnsightExport
    private JPanel_EnsightExport_ViewModelContainer EEVMC;
    private JPanel_EnsightExport_DataMapper EEDMap;
    private EnsightExport_DataManager EEDMan;
    private Controller_JPanel_EnsightExport Contr_EE;
    private EnsightExportManager EEMan;
    //Covise Net Generierung
    private JPanel_CoviseNetGeneration_ViewModelContainer CNGVMC;
    private JPanel_CoviseNetGeneration_DataMapper CNGDMap;
    private Controller_JPanel_CoviseNetGeneration Contr_CNG;
    private CoviseNetGeneration_DataManager CNGDMan;
    //Funktionalität
    private boolean EventManager_EnablePossible = true;
    private boolean EventManager_Enabled = true;

    //Constructor
    public PluginContainer(Simulation Sim) {
        SIM = Sim;
        SIM_DATA_MANGER = new SimDataManager(this);
        try {

            GUI = new JFrame_MainFrame();
            EEVMC = new JPanel_EnsightExport_ViewModelContainer(this);




            EEDMan = new EnsightExport_DataManager(
                    this);
            EEMan = new EnsightExportManager(this);
            Contr_EE = new Controller_JPanel_EnsightExport(this);
            EEDMap = new JPanel_EnsightExport_DataMapper(this);


            CNGVMC = new JPanel_CoviseNetGeneration_ViewModelContainer(this);
            CNGDMan = new CoviseNetGeneration_DataManager(this);
            Contr_CNG = new Controller_JPanel_CoviseNetGeneration(this);
            CNGDMap = new JPanel_CoviseNetGeneration_DataMapper(this);


            Manager_Controller.add(this);
            Manager_Controller_CoviseNetGeneration.add(this);
            GUI.setVisible(true);
        } catch (Exception Ex) {
            StringWriter sw = new StringWriter();
            new Error_Dialog(Configuration_GUI_Strings.Occourence + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.ErrMass + Ex.getMessage() + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.StackTrace + sw.toString());
        }
    }

    //Simulation
    public Simulation getSim() {
        return SIM;
    }

    public SimDataManager getSIM_DATA_MANGER() {
        return SIM_DATA_MANGER;
    }

    //GUI
    public JFrame_MainFrame getGUI() {
        return GUI;
    }

    //EnsightExport
    public JPanel_EnsightExport_ViewModelContainer getEEVMC() {
        return EEVMC;
    }

    public EnsightExport_DataManager getEEDMan() {
        return EEDMan;
    }

    public EnsightExportManager getEEMan() {
        return EEMan;
    }

    public JPanel_EnsightExport_DataMapper getEEDMap() {
        return EEDMap;
    }

    public Controller_JPanel_EnsightExport getContr_EE() {
        return Contr_EE;
    }

    // Covise Net Generation
    public JPanel_CoviseNetGeneration_ViewModelContainer getCNGVMC() {
        return CNGVMC;
    }

    public JPanel_CoviseNetGeneration_DataMapper getCNGDMap() {
        return CNGDMap;
    }

    public Controller_JPanel_CoviseNetGeneration getContr_CNG() {
        return Contr_CNG;
    }

    public CoviseNetGeneration_DataManager getCNGDMan() {
        return CNGDMan;
    }

    //Funktionalität
    //for the use of the Loader and Saver
    public void setState_EnsightExport(DataContainer DC) {
        this.disableEventManager();
        this.EventManager_EnablePossible = false;
        this.CNGDMan.getConMan().clearConstructs();
        this.EEDMan.setExportPath(DC.getExportPath_EnsightExport());
        this.EEDMan.setSelectedRegions(DC.getSelected_Regions());
        this.EEDMan.setSelectedBoundaries(DC.getSelected_Boundaries());
        this.EEDMan.setScalarFieldFunctionSelection(DC.getSelected_Scalars());
        this.EEDMan.setVectorFieldFunctionSelection(DC.getSelected_Vectors());
        this.CNGDMan.setExportPath(DC.getExportPath_CviseNetGeneration());
        this.EEDMan.setAppendtoFile(DC.isAppendToFile());
        this.EEDMan.setExportonVertices(DC.isExportOnVertices());
        this.EventManager_EnablePossible = true;
        this.enableEventManager();


    }
    //for the use of the Loader and Saver

    public void setState_CoviseNetGeneration(DataContainer DC) {
        this.disableEventManager();
        this.EventManager_EnablePossible = false;
        if (DC.getConstructList() != null) {
            this.getCNGDMan().getConMan().getConstructList().clear();
            for (Construct C : DC.getConstructList()) {
                this.getCNGDMan().getConMan().addConsturct(C);
            }
        }


        this.EventManager_EnablePossible = true;
        this.enableEventManager();

    }

    public void enableEventManager() {
        if (EventManager_EnablePossible == true) {
            this.Contr_CNG.setEnabled(true);
            this.Contr_EE.setEnabled(true);
            this.EventManager_Enabled = true;
        }
    }

    public void disableEventManager() {

        this.Contr_EE.setEnabled(false);
        this.Contr_CNG.setEnabled(false);
        this.EventManager_Enabled = false;
    }

    public boolean isEventManager_Enabled() {
        return EventManager_Enabled;
    }
}
