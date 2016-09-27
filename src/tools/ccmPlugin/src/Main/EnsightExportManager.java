package Main;

import de.hlrs.starplugin.interfaces.Interface_EnsightExport_DataChangedListener;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import star.base.neo.NeoObjectVector;
import star.common.Boundary;
import star.common.FieldFunction;
import star.common.ImportManager;
import star.common.Region;

/**
 *  This class stears the Ensight Export over the StarCCM+ API
 * @author Weiss HLRS Stuttgart
 */
public class EnsightExportManager implements Interface_EnsightExport_DataChangedListener {

    private PluginContainer PC;
    private Collection<Region> Regions;
    private Collection<Boundary> Boundaries;
    private Collection<FieldFunction> Scalars;
    private Collection<FieldFunction> Vectors;
    private boolean AppendtoFile;
    private boolean ExportOnVertices;
    private File ExportPath;
    private Collection<FieldFunction> FieldFunctionstoexport;
    private HashMap<Object, Integer> PartsList = new HashMap<Object, Integer>();
    private HashMap<Object, Integer> RegionsList = new HashMap<Object, Integer>();
    private HashMap<Object, Integer> BoundariesList = new HashMap<Object, Integer>();
    private HashMap<Object, Integer> ScalarsList = new HashMap<Object, Integer>();
    private HashMap<Object, Integer> VectorsList = new HashMap<Object, Integer>();

    public EnsightExportManager(PluginContainer Pc) {

        this.PC = Pc;
        this.PC.getEEDMan().addListener(this);
        Regions = new NeoObjectVector(new Object[]{});
        Boundaries = new NeoObjectVector(new Object[]{});
        Scalars = new NeoObjectVector(new Object[]{});
        Vectors = new NeoObjectVector(new Object[]{});
        FieldFunctionstoexport = new NeoObjectVector(new Object[]{});
        AppendtoFile = PC.getEEDMan().isAppendtoFile();
        ExportOnVertices = PC.getEEDMan().isExportonVertices();
        ExportPath = PC.getEEDMan().getExportPath();
    }

    public void ExportSimulation() {
        // Clearing of the Lists
        Regions.clear();
        Boundaries.clear();
        Scalars.clear();
        Vectors.clear();
        FieldFunctionstoexport.clear();
        //Filling the List with uptoDate Data
        Regions.addAll(PC.getEEDMan().getRegionsSelectedList());
        Boundaries.addAll(PC.getEEDMan().getBoundariesSelectedList());
        Scalars.addAll(PC.getEEDMan().getScalarsSelectedList());
        Vectors.addAll(PC.getEEDMan().getVectorsSelectedList());
        FieldFunctionstoexport.addAll(Scalars);
        FieldFunctionstoexport.addAll(Vectors);

        AppendtoFile = PC.getEEDMan().isAppendtoFile();
        ExportOnVertices = PC.getEEDMan().isExportonVertices();
        ExportPath = PC.getEEDMan().getExportPath();

//Forward Information to StarCCM+
        ImportManager ExportManager = PC.getSim().getExportManager();
        ExportManager.setExportPath(ExportPath);
        ExportManager.setExportRegions(PC.getEEDMan().getRegionsSelectedList());
        ExportManager.setExportBoundaries(PC.getEEDMan().getBoundariesSelectedList());
        ExportManager.setExportScalars(PC.getEEDMan().getScalarsSelectedList());
        ExportManager.setExportVectors(PC.getEEDMan().getVectorsSelectedList());

        /**
         * Print to controll output
         *       PC.getSim().println("Regions");
         *       PC.getSim().println(Regions.toString());
         *       for (Region r : Regions) {
         *           PC.getSim().println(Arrays.toString(r.getBoundaryManager().getBoundaries().toArray()));
         *       }

         *       PC.getSim().println("Boundaries");
         *       PC.getSim().println(Boundaries.toString());
         *        PC.getSim().println(ExportManager.getExportBoundaries());
         *        PC.getSim().println(ExportManager.getExportRegions());

         *        PC.getSim().println("Scalar");
         *       PC.getSim().println(Arrays.toString(PC.getEEDMan().getScalarsSelectedList().toArray()));

         *        PC.getSim().println("Vector");
         *      PC.getSim().println(Arrays.toString(PC.getEEDMan().getVectorsSelectedList().toArray()));

        PC.getSim().println("Partzuordnung");
        printZuordnug();

        PC.getSim().println("Export on Vertices: " + ExportOnVertices);
        PC.getSim().println("Append to File: " + AppendtoFile);
         */
        ExportManager.export(ExportPath.getAbsolutePath(), Regions, Boundaries, new NeoObjectVector(
                new Object[]{}), FieldFunctionstoexport, AppendtoFile, ExportOnVertices);
    }
//for coding Use prints the mapping of parts to Partnumbers in Covise
    public void printZuordnug() {
        List<Object> PartList = new ArrayList<Object>();

//        PC.getSim().println(Arrays.toString(Regions.toArray()));

//        PC.getSim().println(Arrays.toString(Regions.toArray()));
        for (Region r : PC.getSim().getRegionManager().getRegions()) {
            if (Regions.contains(r)) {
                PartList.add(r);
                PartList.addAll(r.getBoundaryManager().getBoundaries());
            }
        }
        for (Boundary b : Boundaries) {
            if (!PartList.contains(b)) {
                PartList.add(b);
            }
        }
        int i = 1;
        for (Object o : PartList) {
//            PC.getSim().print("Part " + i + ": ");
//            PC.getSim().println(o);
            i++;
        }
//        PC.getSim().println("Scalar");
        for (Entry<Object, Integer> e : ScalarsList.entrySet()) {
//            PC.getSim().print("Part " + e.getValue() + ": ");
//            PC.getSim().println(e.getKey().toString());
        }
//        PC.getSim().println("Vector");
        for (Entry<Object, Integer> e : VectorsList.entrySet()) {
//            PC.getSim().print("Part " + e.getValue() + ": ");
//            PC.getSim().println(e.getKey().toString());
        }

    }
//regeneration of the GeometryPartMapping
    private void updatePartList() {
        int pa = 1;
        int re = 0;
        int bo = 0;
        PartsList.clear();
        List<Region> RegionList = new ArrayList<Region>();
        RegionList.addAll(PC.getEEDMan().getRegionsSelectedList());
        for (Region r : PC.getSim().getRegionManager().getRegions()) {
            if (RegionList.contains(r)) {
                PartsList.put(r, pa);
                RegionsList.put(r, re);
                re++;
                pa++;
                for (Boundary B : r.getBoundaryManager().getBoundaries()) {
                    PartsList.put(B, pa);
                    BoundariesList.put(B, bo);
                    bo++;
                    pa++;
                }
            }
        }
        for (Boundary B : PC.getEEDMan().getBoundariesSelectedList()) {
            if (!PartsList.containsKey(B)) {
                PartsList.put(B, pa);
                BoundariesList.put(B, bo);
                bo++;
                pa++;
            }
        }

    }
//regeneration of the ScalarPartMapping
    private void updateScalarsList() {
        int i = 2;
        ScalarsList.clear();
        for (FieldFunction FF : PC.getEEDMan().getScalarsSelectedList()) {
            ScalarsList.put(FF, i);
            i++;
        }

    }
//regeneration of the VecPartMapping
    private void updateVectorsList() {
        int i = 2;
        VectorsList.clear();
        for (FieldFunction FF : PC.getEEDMan().getVectorsSelectedList()) {
            VectorsList.put(FF, i);
            i++;
        }
    }

    public void RegionSelectionChanged() {
        this.updatePartList();
    }

    public void BoundarySelectionChanged() {
        this.updatePartList();
    }

    public void ScalarsSelectionChanged() {
        updateScalarsList();
    }

    public void VectorsSelectionChanged() {
        updateVectorsList();
    }

    public void EnsightExportPathChanged() {
        this.ExportPath = this.PC.getEEDMan().getExportPath();
    }

    public void ExportonVerticesChangedChanged() {
    }

    public void AppendToExistingFileChanged() {
    }

    public HashMap<Object, Integer> getPartsList() {
        return PartsList;
    }

    public HashMap<Object, Integer> getBoundariesList() {
        return BoundariesList;
    }

    public HashMap<Object, Integer> getRegionsList() {
        return RegionsList;
    }

    public HashMap<Object, Integer> getScalarsList() {
        return ScalarsList;
    }

    public HashMap<Object, Integer> getVectorsList() {
        return VectorsList;
    }
}
