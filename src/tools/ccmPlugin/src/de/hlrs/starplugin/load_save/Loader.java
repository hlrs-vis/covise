package de.hlrs.starplugin.load_save;

import Main.PluginContainer;
import de.hlrs.starplugin.configuration.Configuration_GUI_Strings;
import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_CuttingSurface;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_CuttingSurfaceSeries;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_GeometryVisualization;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_IsoSurface;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_Streamline;
import de.hlrs.starplugin.util.FieldFunctionplusType;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;
import star.common.Boundary;
import star.common.FieldFunction;
import star.common.Region;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Loader {

    public static Message_Load LoadState(PluginContainer PC, File FilePath) throws FileNotFoundException, IOException, ClassNotFoundException {

        Message_Load Message = new Message_Load();
        FileInputStream fos = new FileInputStream(FilePath);
        ObjectInputStream ois = new ObjectInputStream(fos);
        Serializable_DataContainer SDC = (Serializable_DataContainer) ois.readObject();
        System.out.print("test");
        DataContainer DC = new DataContainer();

        DC.setExportPath_EnsightExport(SDC.getExportPath_EnsightExport());
        DC.setExportPath_CviseNetGeneration(SDC.getExportPath_CviseNetGeneration());
        DC.setAppendToFile(SDC.isAppendToFile());
        DC.setExportOnVertices(SDC.isExportOnVertices());


        DC.setSelected_Boundaries(getArrayListB(SDC.getSelected_Boundaries(), PC, Message));
        DC.setSelected_Regions(getArrayListR(SDC.getSelected_Regions(), PC, Message));
        DC.setSelected_Scalars(getArrayListFF(SDC.getSelected_Scalars(), PC, Message));
        DC.setSelected_Vectors(getArrayListFF(SDC.getSelected_Vectors(), PC, Message));
        PC.setState_EnsightExport(DC);
        DC.setConstructList(getConstructList(SDC.getConstructList(), PC, Message));
        PC.setState_CoviseNetGeneration(DC);
        return Message;


    }

    private static ArrayList<Boundary> getArrayListB(ArrayList<String> AL_String, PluginContainer PC, Message_Load Message) {
        ArrayList<Boundary> AL_Boundary = new ArrayList<Boundary>();
        for (String S : AL_String) {
            if (PC.getSIM_DATA_MANGER().getAllBoundariesHashMap().containsKey(S)) {
                AL_Boundary.add(PC.getSIM_DATA_MANGER().getAllBoundariesHashMap().get(S));
            } else {
                Message.setTrouble(true);
                Message.addBoundary(S);
            }
        }
        return AL_Boundary;

    }

    private static ArrayList<Region> getArrayListR(ArrayList<String> AL_String, PluginContainer PC, Message_Load Message) {
        ArrayList<Region> AL_Region = new ArrayList<Region>();
        for (String S : AL_String) {
            if (PC.getSIM_DATA_MANGER().getAllRegionsHashMap().containsKey(S)) {
                AL_Region.add(PC.getSIM_DATA_MANGER().getAllRegionsHashMap().get(S));
            } else {
                Message.setTrouble(true);
                Message.addRegion(S);
            }
        }
        return AL_Region;

    }

    private static ArrayList<FieldFunction> getArrayListFF(ArrayList<String> AL_String, PluginContainer PC, Message_Load Message) {
        ArrayList<FieldFunction> AL_FieldFunction = new ArrayList<FieldFunction>();
        for (String S : AL_String) {
            if (PC.getSIM_DATA_MANGER().getAllFieldFunctionHashMap().containsKey(S)) {
                AL_FieldFunction.add(PC.getSIM_DATA_MANGER().getAllFieldFunctionHashMap().get(S));
            } else {
                Message.setTrouble(true);
                Message.addFF(S);
            }
        }
        return AL_FieldFunction;
    }

    private static ArrayList<Construct> getConstructList(ArrayList<Serializable_Construct> SCL, PluginContainer PC, Message_Load Message) {

        ArrayList<Construct> AL_Construct = new ArrayList<Construct>();

        //FieldFunctions
        HashMap<String, FieldFunctionplusType> HM_FFpT = new HashMap<String, FieldFunctionplusType>();
        for (FieldFunction FF : PC.getSIM_DATA_MANGER().getAllScalarFieldFunctionsList()) {
            HM_FFpT.put(FF.getPresentationName(), new FieldFunctionplusType(FF,
                    Configuration_Tool.DataType_scalar));
        }
        for (FieldFunction FF : PC.getSIM_DATA_MANGER().getAllVectorFieldFunctionsList()) {
            HM_FFpT.put(FF.getPresentationName(), new FieldFunctionplusType(FF,
                    Configuration_Tool.DataType_vector));
        }

        //Parts
        HashMap<String, Entry<Object, Integer>> HM_Parts = new HashMap<String, Entry<Object, Integer>>();
        for (Entry<Object, Integer> e : PC.getEEMan().getPartsList().entrySet()) {
            HM_Parts.put(e.getKey().toString(), e);
        }

        for (Serializable_Construct SC : SCL) {
            //
            boolean TeilFehlt = false;
            //FF plus Type
            FieldFunctionplusType FFpT = null;
            if (SC.getFF() != null) {
                if (HM_FFpT.containsKey(SC.getFF())) {
                    FFpT = HM_FFpT.get(SC.getFF());
                } else {
                    TeilFehlt = true;
                }
            }

            //Parts of Construct
            HashMap<Object, Integer> Parts = new HashMap<Object, Integer>();
            for (String S : SC.getParts()) {
                if (HM_Parts.containsKey(S)) {
                    Entry<Object, Integer> e = HM_Parts.get(S);
                    Parts.put(e.getKey(), e.getValue());
                } else {
                    TeilFehlt = true;
                }
            }
            if (!TeilFehlt) {
                if (SC instanceof Serializable_Construct_Streamline) {
                    HashMap<Object, Integer> InitialSurface = null;
                    if (((Serializable_Construct_Streamline) SC).getInitialSurface() != null) {
                        InitialSurface = new HashMap<Object, Integer>();
                        for (String S : ((Serializable_Construct_Streamline) SC).getInitialSurface()) {
                            if (HM_Parts.containsKey(S)) {
                                Entry<Object, Integer> E = HM_Parts.get(S);
                                InitialSurface.put(E.getKey(), PC.getEEMan().getBoundariesList().get(E.getKey()));
                            }
                        }
                    } else {
                        InitialSurface = null;
                        //TODO mistake
                    }
                    AL_Construct.add(new Construct_Streamline((Serializable_Construct_Streamline) SC,
                            FFpT, Parts, InitialSurface));
                }
                if (SC instanceof Serializable_Construct_IsoSurface) {
                    AL_Construct.add(new Construct_IsoSurface((Serializable_Construct_IsoSurface) SC,
                            FFpT, Parts));
                }
                if (SC instanceof Serializable_Construct_GeometryVisualization) {
                    AL_Construct.add(new Construct_GeometryVisualization(
                            (Serializable_Construct_GeometryVisualization) SC, FFpT, Parts));
                }
                if (SC instanceof Serializable_Construct_CuttingSurface && !(SC instanceof Serializable_Construct_CuttingSurfaceSeries)) {
                    AL_Construct.add(new Construct_CuttingSurface((Serializable_Construct_CuttingSurface) SC,
                            FFpT,
                            Parts));
                }
                if (SC instanceof Serializable_Construct_CuttingSurfaceSeries) {
                    AL_Construct.add(new Construct_CuttingSurfaceSeries(
                            (Serializable_Construct_CuttingSurfaceSeries) SC, FFpT, Parts));
                }

            } else {
                Message.setTrouble(true);
                Message.addCon("Name: " + SC.Name + Configuration_GUI_Strings.eol + "Typ:" + SC.Typ);
            }
        }
        return AL_Construct;

    }
}

