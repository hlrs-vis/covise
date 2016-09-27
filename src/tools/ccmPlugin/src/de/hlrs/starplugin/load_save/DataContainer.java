package de.hlrs.starplugin.load_save;

import de.hlrs.starplugin.covise_net_generation.constructs.Construct;
import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;

import star.common.Boundary;
import star.common.FieldFunction;
import star.common.Region;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class DataContainer implements Serializable {

    private static final long serialVersionUID = 3191075037401765218L;
    //Ensight Export
    private File ExportPath_EnsightExport;
    private boolean AppendToFile;
    private boolean ExportOnVertices;
    private ArrayList<Region> Selected_Regions;
    private ArrayList<Boundary> Selected_Boundaries;
    private ArrayList<FieldFunction> Selected_Scalars;
    private ArrayList<FieldFunction> Selected_Vectors;
    //Covise NetGeneration
    private File ExportPath_CviseNetGeneration;
    private ArrayList<Construct> ConstructList;

    public DataContainer() {
    }

    public DataContainer(File ExportPath_EnsightExport, boolean AppendToFile, boolean ExportOnVertices, ArrayList<Region> Selected_Regions, ArrayList<Boundary> Selected_Boundaries, ArrayList<FieldFunction> Selected_Scalars, ArrayList<FieldFunction> Selected_Vectors, File ExportPath_CviseNetGeneration, ArrayList<Construct> ConstructList) {
        this.ExportPath_EnsightExport = ExportPath_EnsightExport;
        this.AppendToFile = AppendToFile;
        this.ExportOnVertices = ExportOnVertices;
        this.Selected_Regions = Selected_Regions;
        this.Selected_Boundaries = Selected_Boundaries;
        this.Selected_Scalars = Selected_Scalars;
        this.Selected_Vectors = Selected_Vectors;
        this.ExportPath_CviseNetGeneration = ExportPath_CviseNetGeneration;
        this.ConstructList = ConstructList;
    }
    

    public ArrayList<Construct> getConstructList() {
        return ConstructList;
    }

    public void setConstructList(ArrayList<Construct> ConstructList) {
        this.ConstructList = ConstructList;
    }

    public File getExportPath_CviseNetGeneration() {
        return ExportPath_CviseNetGeneration;
    }

    public void setExportPath_CviseNetGeneration(File ExportPath_CviseNetGeneration) {
        this.ExportPath_CviseNetGeneration = ExportPath_CviseNetGeneration;
    }

    public File getExportPath_EnsightExport() {
        return ExportPath_EnsightExport;
    }

    public void setExportPath_EnsightExport(File ExportPath_EnsightExport) {
        this.ExportPath_EnsightExport = ExportPath_EnsightExport;
    }

    public boolean isAppendToFile() {
        return AppendToFile;
    }

    public void setAppendToFile(boolean AppendToFile) {
        this.AppendToFile = AppendToFile;
    }

    public boolean isExportOnVertices() {
        return ExportOnVertices;
    }

    public void setExportOnVertices(boolean ExportOnVertices) {
        this.ExportOnVertices = ExportOnVertices;
    }
    

    public ArrayList<Boundary> getSelected_Boundaries() {
        return Selected_Boundaries;
    }

    public void setSelected_Boundaries(ArrayList<Boundary> Selected_Boundaries) {
        this.Selected_Boundaries = Selected_Boundaries;
    }

    public ArrayList<Region> getSelected_Regions() {
        return Selected_Regions;
    }

    public void setSelected_Regions(ArrayList<Region> Selected_Regions) {
        this.Selected_Regions = Selected_Regions;
    }

    public ArrayList<FieldFunction> getSelected_Scalars() {
        return Selected_Scalars;
    }

    public void setSelected_Scalars(ArrayList<FieldFunction> Selected_Scalars) {
        this.Selected_Scalars = Selected_Scalars;
    }

    public ArrayList<FieldFunction> getSelected_Vectors() {
        return Selected_Vectors;
    }

    public void setSelected_Vectors(ArrayList<FieldFunction> Selected_Vectors) {
        this.Selected_Vectors = Selected_Vectors;
    }


    







    
}
