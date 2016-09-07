package de.hlrs.starplugin.load_save;

import Main.PluginContainer;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_CuttingSurface;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_CuttingSurfaceSeries;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_GeometryVisualization;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_IsoSurface;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_Streamline;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import star.common.Boundary;
import star.common.FieldFunction;
import star.common.Region;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Saver {

    public static void SaveState(PluginContainer PC, File ExportFilePath) throws FileNotFoundException, IOException {
        
            PC.getEEDMan();
            PC.getCNGDMan().getConMan();

            Serializable_DataContainer DC = new Serializable_DataContainer();
            DC.setExportPath_EnsightExport(PC.getEEDMan().getExportPath());
            DC.setExportPath_CviseNetGeneration(PC.getCNGDMan().getExportPath());
            DC.setAppendToFile(PC.getEEDMan().isAppendtoFile());
            DC.setExportOnVertices(PC.getEEDMan().isExportonVertices());


            DC.setSelected_Boundaries(getArrayListB(PC.getEEDMan().getBoundariesSelectedList()));
            DC.setSelected_Regions(getArrayListR(PC.getEEDMan().getRegionsSelectedList()));
            DC.setSelected_Scalars(getArrayListFF(PC.getEEDMan().getScalarsSelectedList()));
            DC.setSelected_Vectors(getArrayListFF(PC.getEEDMan().getVectorsSelectedList()));
            DC.setConstructList(getConstructList(new ArrayList<Construct>(PC.getCNGDMan().getConMan().getConstructList().
                    values())));
            FileOutputStream fos = new FileOutputStream(ExportFilePath);
            ObjectOutputStream osw = new ObjectOutputStream(fos);
            osw.writeObject(DC);

            osw.flush();
            osw.close();
    }

    private static ArrayList<String> getArrayListB(ArrayList<Boundary> ALO) {
        ArrayList<String> ALS = new ArrayList<String>();
        for (Boundary B : ALO) {
            ALS.add(B.toString());
        }
        return ALS;
    }

    private static ArrayList<String> getArrayListR(ArrayList<Region> ALR) {
        ArrayList<String> ALS = new ArrayList<String>();
        for (Region R : ALR) {
            ALS.add(R.toString());
        }
        return ALS;
    }

    private static ArrayList<String> getArrayListFF(ArrayList<FieldFunction> ARO) {
        ArrayList<String> ALS = new ArrayList<String>();
        for (FieldFunction FF : ARO) {
            ALS.add(FF.getPresentationName());
        }
        return ALS;
    }

    private static ArrayList<Serializable_Construct> getConstructList(ArrayList<Construct> ConList) {
        ArrayList<Serializable_Construct> newAL = new ArrayList<Serializable_Construct>();
        for (Construct c : ConList) {
            if (c instanceof Construct_Streamline) {
                newAL.add(new Serializable_Construct_Streamline((Construct_Streamline) c));
            }
            if (c instanceof Construct_IsoSurface) {
                newAL.add(new Serializable_Construct_IsoSurface((Construct_IsoSurface) c));
            }
            if (c instanceof Construct_GeometryVisualization) {
                newAL.add(
                        new Serializable_Construct_GeometryVisualization((Construct_GeometryVisualization) c));
            }
            if (c instanceof Construct_CuttingSurface && !(c instanceof Construct_CuttingSurfaceSeries)) {
                newAL.add(
                        new Serializable_Construct_CuttingSurface((Construct_CuttingSurface) c));
            }
            if (c instanceof Construct_CuttingSurfaceSeries) {
                newAL.add(new Serializable_Construct_CuttingSurfaceSeries((Construct_CuttingSurfaceSeries) c));
            }

        }
        return newAL;
    }
}
