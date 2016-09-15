package de.hlrs.starplugin.ensight_export;

import de.hlrs.starplugin.interfaces.Interface_EnsightExport_DataChangedListener;
import Main.PluginContainer;
import de.hlrs.starplugin.configuration.Configuration_GUI_Strings;
import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_Streamline;
import de.hlrs.starplugin.gui.dialogs.Error_Dialog;
import de.hlrs.starplugin.util.FieldFunctionplusType;
import java.io.File;
import java.io.PrintWriter;
import java.io.StringWriter;
import star.common.Region;
import star.common.Boundary;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import star.common.FieldFunction;

/**
 *Is the DataModel that holds Data about the Configuration of the Ensight Export in StarCCM+
 * @author Weiss HLRS Stuttgart
 */
public class EnsightExport_DataManager {

    private PluginContainer PC;
    //Selection for Export
    final private ArrayList<Region> REGIONS_SELECTED_LIST;
    final private ArrayList<Boundary> BOUNDARIES_SELECTED_LIST;
    final private ArrayList<FieldFunction> SCALAR_SELECTED_LIST;
    final private ArrayList<FieldFunction> VECTOR_SELECTED_LIST;
    private boolean appendtoFile = false;
    private boolean exportonVertices = false;
    private File ExportPath;
    //Funktionalität
    private List<Interface_EnsightExport_DataChangedListener> listeners = new ArrayList<Interface_EnsightExport_DataChangedListener>();
    private boolean BoundarySelectionChangeable;

    //Konstruktor
    public EnsightExport_DataManager(PluginContainer PC) {
        this.PC = PC;
        BoundarySelectionChangeable = true;

        //Listen Initialisieren
        REGIONS_SELECTED_LIST = new ArrayList<Region>();
        BOUNDARIES_SELECTED_LIST = new ArrayList<Boundary>();
        SCALAR_SELECTED_LIST = new ArrayList<FieldFunction>();
        VECTOR_SELECTED_LIST = new ArrayList<FieldFunction>();

        //ExportPath
        ExportPath = PC.getSim().getExportManager().getExportPath();

    }

    public boolean isAppendtoFile() {
        return appendtoFile;
    }

    public void setAppendtoFile(boolean appendtoFile) {
        this.appendtoFile = appendtoFile;
        this.dataChanged(Configuration_Tool.onChange_AppendToFile);
    }

    public boolean isExportonVertices() {
        return exportonVertices;
    }

    public void setExportonVertices(boolean exportonVertices) {
        this.exportonVertices = exportonVertices;
        this.dataChanged(Configuration_Tool.onChange_ExportOnVertices);
    }

    public void setSelectedRegions(ArrayList<Region> Regions) {
        this.REGIONS_SELECTED_LIST.clear();
        this.REGIONS_SELECTED_LIST.addAll(Regions);
        checkUsage();
        for (Region r : this.REGIONS_SELECTED_LIST) {

            for (Boundary B : r.getBoundaryManager().getBoundaries()) {
                if (!this.BOUNDARIES_SELECTED_LIST.contains(B)) {
                    BOUNDARIES_SELECTED_LIST.add(B);
                }
            }
        }
        this.dataChanged(Configuration_Tool.onChange_Boundary);
        this.dataChanged(Configuration_Tool.onChange_Region);

    }

    public ArrayList<Region> getRegionsSelectedList() {
        return REGIONS_SELECTED_LIST;
    }

    public void setSelectedBoundaries(ArrayList<Boundary> tmpSelectedBoundaryList) {
        this.BOUNDARIES_SELECTED_LIST.clear();
        for (Region r : this.REGIONS_SELECTED_LIST) {
            this.BOUNDARIES_SELECTED_LIST.addAll(r.getBoundaryManager().getBoundaries());
        }
        for (Boundary B : tmpSelectedBoundaryList) {
            if (!this.BOUNDARIES_SELECTED_LIST.contains(B)) {
                this.BOUNDARIES_SELECTED_LIST.add(B);
            }
        }
        checkUsage();
        this.dataChanged(Configuration_Tool.onChange_Boundary);
    }

    public ArrayList<Boundary> getBoundariesSelectedList() {
        return BOUNDARIES_SELECTED_LIST;
    }

    public ArrayList<FieldFunction> getScalarsSelectedList() {
        return SCALAR_SELECTED_LIST;
    }

    public void addScalarFieldFunctiontoSelection(FieldFunction FF) {
        SCALAR_SELECTED_LIST.add(FF);
        Collections.sort(SCALAR_SELECTED_LIST, new Comparator<FieldFunction>() {

            @Override
            public int compare(FieldFunction a, FieldFunction b) {
                return a.getPresentationName().compareTo(b.getPresentationName());
            }
        });
    }

    public void setScalarFieldFunctionSelection(ArrayList<FieldFunction> AL_FF) {
        SCALAR_SELECTED_LIST.clear();
        for (FieldFunction FF : AL_FF) {
            SCALAR_SELECTED_LIST.add(FF);
        }
        checkUsage();


        Collections.sort(SCALAR_SELECTED_LIST, new Comparator<FieldFunction>() {

            @Override
            public int compare(FieldFunction a, FieldFunction b) {
                return a.getPresentationName().compareTo(b.getPresentationName());
            }
        });
        dataChanged(Configuration_Tool.onChange_Scalar);

    }

    //Used in Streamline to add prop. missing Velocity Function to the Export Selection
    public void addVectorFieldFunctiontoSelection(FieldFunction FF) {
        VECTOR_SELECTED_LIST.add(FF);
        Collections.sort(VECTOR_SELECTED_LIST, new Comparator<FieldFunction>() {

            @Override
            public int compare(FieldFunction a, FieldFunction b) {

                return a.getPresentationName().compareTo(b.getPresentationName());

            }
        });
    }

    public void setVectorFieldFunctionSelection(ArrayList<FieldFunction> AL_FF) {
        VECTOR_SELECTED_LIST.clear();
        for (FieldFunction FF : AL_FF) {
            VECTOR_SELECTED_LIST.add(FF);
        }
        checkUsage();
        Collections.sort(VECTOR_SELECTED_LIST, new Comparator<FieldFunction>() {

            @Override
            public int compare(FieldFunction a, FieldFunction b) {
                return a.getPresentationName().compareTo(b.getPresentationName());
            }
        });
        dataChanged(Configuration_Tool.onChange_Vector);
    }

    public ArrayList<FieldFunction> getVectorsSelectedList() {
        return VECTOR_SELECTED_LIST;
    }

    public boolean isBoundarySelectionChangeable() {
        return BoundarySelectionChangeable;
    }

    public void setBoundarySelectionChangeable(boolean BoundarySelectionChangeable) {
        this.BoundarySelectionChangeable = BoundarySelectionChangeable;
    }

    public File getExportPath() {
        return ExportPath;
    }

    public void setExportPath(File ExportPath) {
        this.ExportPath = ExportPath;
        dataChanged(Configuration_Tool.onChange_ExportPath);
    }

    public void addListener(Interface_EnsightExport_DataChangedListener toAdd) {
        listeners.add(toAdd);
    }

    public void dataChanged(String e) {
        // Notify everybody that may be interested.
        if (e != null) {
            if (e.equals(Configuration_Tool.onChange_Region)) {
                for (Interface_EnsightExport_DataChangedListener dcl : listeners) {

                    dcl.RegionSelectionChanged();
                }
            }
            if (e.equals(Configuration_Tool.onChange_Boundary)) {
                for (Interface_EnsightExport_DataChangedListener dcl : listeners) {

                    dcl.BoundarySelectionChanged();
                }
            }
            if (e.equals(Configuration_Tool.onChange_Scalar)) {
                for (Interface_EnsightExport_DataChangedListener dcl : listeners) {
                    dcl.ScalarsSelectionChanged();
                }
            }
            if (e.equals(Configuration_Tool.onChange_Vector)) {
                for (Interface_EnsightExport_DataChangedListener dcl : listeners) {
                    dcl.VectorsSelectionChanged();
                }
            }
            if (e.equals(Configuration_Tool.onChange_ExportPath)) {
                for (Interface_EnsightExport_DataChangedListener dcl : listeners) {
                    dcl.EnsightExportPathChanged();
                }
            }
            if (e.equals(Configuration_Tool.onChange_AppendToFile)) {
                for (Interface_EnsightExport_DataChangedListener dcl : listeners) {
                    dcl.AppendToExistingFileChanged();
                }
            }
            if (e.equals(Configuration_Tool.onChange_ExportOnVertices)) {
                for (Interface_EnsightExport_DataChangedListener dcl : listeners) {
                    dcl.ExportonVerticesChangedChanged();
                }
            }
        }

    }

    private void checkUsage() {

        HashMap<String, Construct> ConList = PC.getCNGDMan().getConMan().getConstructList();
        for (Entry<String, Construct> E : ConList.entrySet()) {
            Construct C = E.getValue();

            try {
                //Usage of Regions and Boundaries Geometry Export
                HashMap<Object, Integer> Parts = C.getParts();
                for (Entry<Object, Integer> P : Parts.entrySet()) {
                    if (P.getKey() instanceof Region) {
                        if (!this.REGIONS_SELECTED_LIST.contains((Region) P.getKey())) {
                            REGIONS_SELECTED_LIST.add((Region) P.getKey());
                        }

                    }
                    if (P.getKey() instanceof Boundary) {
                        if (!this.BOUNDARIES_SELECTED_LIST.contains((Boundary) P.getKey())) {
                            BOUNDARIES_SELECTED_LIST.add((Boundary) P.getKey());
                        }

                    }
                }

                //Usage of a Boundary as initial Surface Initial Surface
                if (C instanceof Construct_Streamline) {
                    if (!((Construct_Streamline) C).getInitialSurface().isEmpty()) {
                        for (Entry<Object, Integer> E2 : ((Construct_Streamline) C).getInitialSurface().
                                entrySet()) {
                            if (E2.getKey() instanceof Boundary) {
                                if (!this.BOUNDARIES_SELECTED_LIST.contains((Boundary) E2.getKey())) {
                                    BOUNDARIES_SELECTED_LIST.add((Boundary) E2.getKey());
                                }
                            }
                        }
                    }
                }

                //Usage of FieldFunctions
                FieldFunctionplusType FieldFunction = C.getFFplType();
                if (FieldFunction != null) {
                    if (FieldFunction.getType().equals(Configuration_Tool.DataType_vector)) {
                        if (!this.VECTOR_SELECTED_LIST.contains(FieldFunction.getFF())) {
                            VECTOR_SELECTED_LIST.add(FieldFunction.getFF());
                        }
                    }
                    if (FieldFunction.getType().equals(Configuration_Tool.DataType_scalar)) {
                        if (!this.SCALAR_SELECTED_LIST.contains(FieldFunction.getFF())) {
                            SCALAR_SELECTED_LIST.add(FieldFunction.getFF());
                        }
                    }
                }

            } catch (Exception Ex) {
                StringWriter sw = new StringWriter();
                Ex.printStackTrace(new PrintWriter(sw));
                new Error_Dialog(Configuration_GUI_Strings.Occourence + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.ErrMass + Ex.getMessage() + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.StackTrace + sw.toString());
            }
        }
    }

    public void addSelectedBoundary(Boundary B) {
        if (B != null) {
            if (!this.BOUNDARIES_SELECTED_LIST.contains(B) && this.PC.getSIM_DATA_MANGER().getAllBoundariesList().contains(B)) {
                this.BOUNDARIES_SELECTED_LIST.add(B);
                this.setSelectedBoundaries(new ArrayList<Boundary>(BOUNDARIES_SELECTED_LIST));
            }
        }
    }

    public void addSelectedRegion(Region R) {
        if (R != null) {
            if (!this.REGIONS_SELECTED_LIST.contains(R) && this.PC.getSIM_DATA_MANGER().getAllRegionsList().contains(R)) {
                this.REGIONS_SELECTED_LIST.add(R);
                this.setSelectedRegions(new ArrayList<Region>(REGIONS_SELECTED_LIST));
            }
        }
    }

    public void addSelectedScalar(FieldFunction FF) {
        if (FF != null) {
            if (!this.SCALAR_SELECTED_LIST.contains(FF) && this.PC.getSIM_DATA_MANGER().getAllFieldFunctionHashMap().containsValue(FF)) {
                this.SCALAR_SELECTED_LIST.add(FF);
                this.setScalarFieldFunctionSelection(new ArrayList<FieldFunction>(SCALAR_SELECTED_LIST));
            }
        }
    }

    public void addSelectedVector(FieldFunction FF) {
        if (FF != null) {
            if (!this.VECTOR_SELECTED_LIST.contains(FF) && this.PC.getSIM_DATA_MANGER().getAllFieldFunctionHashMap().containsValue(FF)) {
                this.VECTOR_SELECTED_LIST.add(FF);
                this.setVectorFieldFunctionSelection(new ArrayList<FieldFunction>(VECTOR_SELECTED_LIST));
            }
        }
    }
}
