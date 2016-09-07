package Main;

import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.util.FieldFunctionplusType;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import star.common.AbstractReferenceFrame;
import star.common.Boundary;
import star.common.CoordinateSystem;
import star.common.FieldFunction;
import star.common.FieldFunctionManager;
import star.common.LabCoordinateSystem;
import star.common.LocalCoordinateSystem;
import star.common.Region;
import star.motion.ReferenceFrameManager;

/**
 *This Classs holds the Data about the active Simulation
 * 
 * @author Weiss HLRS Stuttgart
 */
public class SimDataManager {

    private PluginContainer PC;
    private ArrayList<Region> AllRegionsList;
    private ArrayList<Boundary> AllBoundariesList;
    private ArrayList<FieldFunction> AllScalarFieldFunctionsList;
    private ArrayList<FieldFunction> AllVectorFieldFunctionsList;
    private ArrayList<FieldFunction> AllFieldFunctionsList;
    private ArrayList<CoordinateSystem> CoordinateSystemsList;
    private ArrayList<AbstractReferenceFrame> ReferenceFramesList;
    //Load and Save HashMaps
    private HashMap<String, FieldFunction> AllFieldFunctionHashMap;
    private HashMap<String, FieldFunctionplusType> AllFieldFunctionplusTypeHashMap;
    private HashMap<String, Region> AllRegionsHashMap;
    private HashMap<String, Boundary> AllBoundariesHashMap;

    public SimDataManager(PluginContainer PC) {
        this.PC = PC;
        Collection<Region> RegionCollection = PC.getSim().getRegionManager().getRegions();

        FieldFunctionManager FFM = PC.getSim().getFieldFunctionManager();

        Collection<FieldFunction> CVFF = FFM.getVectorFieldFunctions();
        Collection<FieldFunction> CSFF = FFM.getScalarFieldFunctions();

        //Zuerst alle Regions und Boundaries in eine Liste Füllen

        Iterator<Region> RegionIterator = RegionCollection.iterator();
        //Listen initialisieren
        AllRegionsList = new ArrayList<Region>();
        AllBoundariesList = new ArrayList<Boundary>();




        //Region Liste füllen
        while (RegionIterator.hasNext()) {
            Region region = RegionIterator.next();
            AllRegionsList.add(region);

        }



        //Wenn Region Liste nicht leer boundary Liste erstellen und Füllen
        if (!AllRegionsList.isEmpty()) {
            //über RegionListe Iterieren
            for (Region region : AllRegionsList) {
                //Boundary Collection aus Region extrahieren
                Collection<Boundary> TmpBoundaryCollection = region.getBoundaryManager().getBoundaries();
                //Und zur Boundary Liste hinzufügen
                Iterator<Boundary> BoundaryIterator = TmpBoundaryCollection.iterator();
                while (BoundaryIterator.hasNext()) {
                    Boundary boundary = BoundaryIterator.next();
                    AllBoundariesList.add(boundary);
                }
            }
        }


        //ArrayList für Scalar Field functions Füllen
        AllScalarFieldFunctionsList = new ArrayList<FieldFunction>();

        for (FieldFunction FF : CSFF) {
            AllScalarFieldFunctionsList.add(FF);
        }

        //ArrayList für Vector Field functions Füllen
        AllVectorFieldFunctionsList = new ArrayList<FieldFunction>();

        for (FieldFunction FF : CVFF) {
            AllVectorFieldFunctionsList.add(FF);
        }

        //ArrayList für alle FieldFunctons
        Collection<FieldFunction> AllFieldFunctions = CSFF;
        AllFieldFunctions.addAll(CVFF);
        AllFieldFunctionsList = new ArrayList<FieldFunction>();
        for (FieldFunction FF : AllFieldFunctions) {
            AllFieldFunctionsList.add(FF);
        }

        //ReferenceFrames
        ReferenceFrameManager RFM = PC.getSim().get(ReferenceFrameManager.class);
        Collection<AbstractReferenceFrame> ARFC = RFM.getObjects();
        ReferenceFramesList = new ArrayList<AbstractReferenceFrame>();
        for (AbstractReferenceFrame ARF : ARFC) {
            ReferenceFramesList.add(ARF);
        }

        //Coordinate Systems
        Collection<CoordinateSystem> CSC = PC.getSim().getCoordinateSystemManager().getAllObjects();
        CoordinateSystemsList = new ArrayList<CoordinateSystem>();
        for (CoordinateSystem CS : CSC) {
            if (CS instanceof LocalCoordinateSystem || CS instanceof LabCoordinateSystem) {
                CoordinateSystemsList.add(CS);
            }
        }

        //Load Save HashMaps
        AllFieldFunctionHashMap = new HashMap<String, FieldFunction>();
        AllFieldFunctionplusTypeHashMap = new HashMap<String, FieldFunctionplusType>();
        AllRegionsHashMap = new HashMap<String, Region>();
        AllBoundariesHashMap = new HashMap<String, Boundary>();
        initiateRegionBoundaryHashMap();
        initiateScalarVectorHashMap();
    }

    public ArrayList<Boundary> getAllBoundariesList() {
        return AllBoundariesList;
    }

    public void setAllBoundariesList(ArrayList<Boundary> AllBoundariesList) {
        this.AllBoundariesList = AllBoundariesList;
    }

    public ArrayList<FieldFunction> getAllFieldFunctionsList() {
        return AllFieldFunctionsList;
    }

    public void setAllFieldFunctionsList(ArrayList<FieldFunction> AllFieldFunctionsList) {
        this.AllFieldFunctionsList = AllFieldFunctionsList;
    }

    public ArrayList<Region> getAllRegionsList() {
        return AllRegionsList;
    }

    public void setAllRegionsList(ArrayList<Region> AllRegionsList) {
        this.AllRegionsList = AllRegionsList;
    }

    public ArrayList<FieldFunction> getAllScalarFieldFunctionsList() {
        return AllScalarFieldFunctionsList;
    }

    public void setAllScalarFieldFunctionsList(ArrayList<FieldFunction> AllScalarFieldFunctionsList) {
        this.AllScalarFieldFunctionsList = AllScalarFieldFunctionsList;
    }

    public ArrayList<FieldFunction> getAllVectorFieldFunctionsList() {
        return AllVectorFieldFunctionsList;
    }

    public void setAllVectorFieldFunctionsList(ArrayList<FieldFunction> AllVectorFieldFunctionsList) {
        this.AllVectorFieldFunctionsList = AllVectorFieldFunctionsList;
    }

    public ArrayList<CoordinateSystem> getCoordinateSystemsList() {
        return CoordinateSystemsList;
    }

    public ArrayList<AbstractReferenceFrame> getReferenceFramesList() {
        return ReferenceFramesList;
    }

    public HashMap<String, Boundary> getAllBoundariesHashMap() {
        return AllBoundariesHashMap;
    }

    public void setAllBoundariesHashMap(HashMap<String, Boundary> AllBoundariesHashMap) {
        this.AllBoundariesHashMap = AllBoundariesHashMap;
    }

    public HashMap<String, FieldFunction> getAllFieldFunctionHashMap() {
        return AllFieldFunctionHashMap;
    }

    public void setAllFieldFunctionHashMap(HashMap<String, FieldFunction> AllFieldFunctionHashMap) {
        this.AllFieldFunctionHashMap = AllFieldFunctionHashMap;
    }

    public HashMap<String, FieldFunctionplusType> getAllFieldFunctionplusTypeHashMap() {
        return AllFieldFunctionplusTypeHashMap;
    }
    

    public HashMap<String, Region> getAllRegionsHashMap() {
        return AllRegionsHashMap;
    }

    public void setAllRegionsHashMap(HashMap<String, Region> AllRegionsHashMap) {
        this.AllRegionsHashMap = AllRegionsHashMap;
    }

    private void initiateRegionBoundaryHashMap() {
        for (Region r : this.AllRegionsList) {
            AllRegionsHashMap.put(r.toString(), r);

        }
        for (Boundary b : this.AllBoundariesList) {
            AllBoundariesHashMap.put(b.toString(), b);
        }
    }

    private void initiateScalarVectorHashMap() {
        //Scalars and Chosen Scalars Nodes

        //Scalars from Scalars
        for (FieldFunction FF : this.AllScalarFieldFunctionsList) {
            if (FF.getPresentationName().equals("Total Pressure")) {

                for (AbstractReferenceFrame ARF : this.ReferenceFramesList) {
                    this.AllFieldFunctionHashMap.put(FF.getFunctionInReferenceFrame(ARF).getPresentationName(),
                            FF.getFunctionInReferenceFrame(ARF));
                    this.AllFieldFunctionplusTypeHashMap.put(FF.getFunctionInReferenceFrame(ARF).getPresentationName(),
                            new FieldFunctionplusType(FF.getFunctionInReferenceFrame(ARF), Configuration_Tool.DataType_scalar));
                }
            } else {
                this.AllFieldFunctionHashMap.put(FF.getPresentationName(), FF);
                this.AllFieldFunctionplusTypeHashMap.put(FF.getPresentationName(),
                        new FieldFunctionplusType(FF, Configuration_Tool.DataType_scalar));
            }
        }

        //Scalars from Vectors
        for (FieldFunction FF : this.AllVectorFieldFunctionsList) {
            if (FF.getPresentationName().equals(Configuration_Tool.FunctionName_Velocity)) {
                for (AbstractReferenceFrame ARF : this.ReferenceFramesList) {
                    //Magnitude
                    this.AllFieldFunctionHashMap.put(FF.getFunctionInReferenceFrame(
                            ARF).getMagnitudeFunction().getPresentationName(), FF.getFunctionInReferenceFrame(
                            ARF).getMagnitudeFunction());
                    this.AllFieldFunctionplusTypeHashMap.put(FF.getFunctionInReferenceFrame(
                            ARF).getMagnitudeFunction().getPresentationName(),
                            new FieldFunctionplusType(FF.getFunctionInReferenceFrame(
                            ARF).getMagnitudeFunction(), Configuration_Tool.DataType_scalar));
                    //Component Functions
                    for (CoordinateSystem CS : this.CoordinateSystemsList) {
                        for (int i = 0; i < 3; i++) {
                            this.AllFieldFunctionHashMap.put(FF.getFunctionInReferenceFrame(ARF).
                                    getFunctionInCoordinateSystem(CS).getComponentFunction(i).
                                    getPresentationName(), FF.getFunctionInReferenceFrame(ARF).
                                    getFunctionInCoordinateSystem(CS).getComponentFunction(i));
                            this.AllFieldFunctionplusTypeHashMap.put(FF.getFunctionInReferenceFrame(ARF).
                                    getFunctionInCoordinateSystem(CS).getComponentFunction(i).
                                    getPresentationName(), new FieldFunctionplusType(FF.getFunctionInReferenceFrame(ARF).
                                    getFunctionInCoordinateSystem(CS).getComponentFunction(i), Configuration_Tool.DataType_scalar));


                        }
                    }
                }

            } else {
                //Magnitude
                this.AllFieldFunctionHashMap.put(FF.getMagnitudeFunction().getPresentationName(), FF.getMagnitudeFunction());
                this.AllFieldFunctionplusTypeHashMap.put(FF.getMagnitudeFunction().getPresentationName(),
                        new FieldFunctionplusType(FF.getMagnitudeFunction(), Configuration_Tool.DataType_scalar));
                //Component Functions
                for (CoordinateSystem CS : this.CoordinateSystemsList) {

                    for (int i = 0; i < 3; i++) {
                        this.AllFieldFunctionHashMap.put(FF.getFunctionInCoordinateSystem(CS).
                                getComponentFunction(i).getPresentationName(), FF.getFunctionInCoordinateSystem(CS).getComponentFunction(i));
                        this.AllFieldFunctionplusTypeHashMap.put(FF.getFunctionInCoordinateSystem(CS).
                                getComponentFunction(i).getPresentationName(),
                                new FieldFunctionplusType(FF.getFunctionInCoordinateSystem(CS).getComponentFunction(i),
                                Configuration_Tool.DataType_scalar));
                    }
                }
            }
        }
        //Vectors
        for (FieldFunction FF : this.AllVectorFieldFunctionsList) {
            if (!FF.getFunctionName().equals(Configuration_Tool.FunctionName_Velocity)) {
                this.AllFieldFunctionHashMap.put(FF.getPresentationName(), FF);
                this.AllFieldFunctionplusTypeHashMap.put(FF.getPresentationName(),
                        new FieldFunctionplusType(FF,
                        Configuration_Tool.DataType_vector));
            } else {
                for (AbstractReferenceFrame ARF : this.ReferenceFramesList) {
                    this.AllFieldFunctionHashMap.put(FF.getFunctionInReferenceFrame(ARF).getPresentationName(),
                            FF.getFunctionInReferenceFrame(ARF));
                    this.AllFieldFunctionplusTypeHashMap.put(FF.getFunctionInReferenceFrame(ARF).getPresentationName(),
                            new FieldFunctionplusType(FF.getFunctionInReferenceFrame(ARF), Configuration_Tool.DataType_vector));

                }
            }
        }
    }
}
