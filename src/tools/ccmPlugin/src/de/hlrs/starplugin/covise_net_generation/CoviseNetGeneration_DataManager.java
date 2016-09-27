package de.hlrs.starplugin.covise_net_generation;

import Main.PluginContainer;
import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_Manager;
import de.hlrs.starplugin.interfaces.Interface_CoviseNetGeneration_DataChangedListener;

import java.io.File;
import java.util.ArrayList;

/**
 *Is the DataModel that holds Data about the Configuration of the Visualization
 * in Covise
 * @author Weiss HLRS Stuttgart
 */
public class CoviseNetGeneration_DataManager {

    private PluginContainer PC;
    private File ExportPath;
    private Construct_Manager ConMan;
    private ArrayList<Interface_CoviseNetGeneration_DataChangedListener> Listener = new ArrayList<Interface_CoviseNetGeneration_DataChangedListener>();

    public CoviseNetGeneration_DataManager(PluginContainer Pc) {
        this.PC = Pc;

        ConMan = new Construct_Manager(this);
        this.EnsightExportPathChanged();

    }

    public File getExportPath() {
        return this.ExportPath;
    }

    public void setExportPath(File ExportPath) {
        this.ExportPath = ExportPath;
        onChange(Configuration_Tool.onChange_ExportPath);
    }

    public void EnsightExportPathChanged() {
        if (this.PC.getEEDMan().getExportPath().getAbsolutePath().endsWith(".case")) {
            String Filename = PC.getEEDMan().getExportPath().getAbsolutePath();
            ExportPath = new File(Filename.substring(0, Filename.length() - 5) + ".py");
            this.setExportPath(ExportPath);
        } else {
            if (this.PC.getEEDMan().getExportPath().getAbsolutePath().endsWith(".py")) {
                ExportPath = new File(PC.getEEDMan().getExportPath().getAbsolutePath());
                this.setExportPath(ExportPath);
            } else {
                ExportPath = new File(PC.getEEDMan().getExportPath().getAbsolutePath() + ".py");
                this.setExportPath(ExportPath);
            }
        }
    }

    public Construct_Manager getConMan() {
        return ConMan;
    }

    public void addListener(Interface_CoviseNetGeneration_DataChangedListener clcl) {
        Listener.add(clcl);
    }

    public void onChange(String e) {
        if (!(e == null)) {
            if (e.equals(Configuration_Tool.onChange_add)) {
                for (Interface_CoviseNetGeneration_DataChangedListener clcl : Listener) {
                    clcl.ConstructListChanged();
                }
            }
            if (e.equals(Configuration_Tool.onChange_delete)) {
                for (Interface_CoviseNetGeneration_DataChangedListener clcl : Listener) {
                    clcl.ConstructListChanged();
                }
            }
            if (e.equals(Configuration_Tool.onChange_Selection)) {
                for (Interface_CoviseNetGeneration_DataChangedListener clcl : Listener) {
                    clcl.SelectionChanged(this.ConMan.getSelectedConstruct());
                }
            }
            if (e.equals(Configuration_Tool.onChange_ExportPath)) {
                for (Interface_CoviseNetGeneration_DataChangedListener clcl : Listener) {
                    clcl.CoviseNetGenerationExportPathChanged();
                }
            }
        }

    }

    public PluginContainer getPC() {
        return PC;
    }
    
}
