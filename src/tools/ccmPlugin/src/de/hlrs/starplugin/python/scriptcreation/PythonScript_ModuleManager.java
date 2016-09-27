package de.hlrs.starplugin.python.scriptcreation;

import de.hlrs.starplugin.python.scriptcreation.modules.Connection;
import de.hlrs.starplugin.configuration.Configuration_ModuleManager;
import Main.PluginContainer;
import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_CuttingSurface;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_CuttingSurfaceSeries;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_GeometryVisualization;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_IsoSurface;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_Streamline;
import de.hlrs.starplugin.configuration.Configuration_Module;
import de.hlrs.starplugin.python.scriptcreation.modules.Module;
import de.hlrs.starplugin.python.scriptcreation.modules.Module_Collect;
import de.hlrs.starplugin.python.scriptcreation.modules.Module_Colors;
import de.hlrs.starplugin.python.scriptcreation.modules.Module_CuttingSurface;
import de.hlrs.starplugin.python.scriptcreation.modules.Module_GetSubset;
import de.hlrs.starplugin.python.scriptcreation.modules.Module_IsoSurface;
import de.hlrs.starplugin.python.scriptcreation.modules.Module_Material;
import de.hlrs.starplugin.python.scriptcreation.modules.Module_MinMax;
import de.hlrs.starplugin.python.scriptcreation.modules.Module_OpenCover;
import de.hlrs.starplugin.python.scriptcreation.modules.Module_ReadEnsight;
import de.hlrs.starplugin.python.scriptcreation.modules.Module_Renderer;
import de.hlrs.starplugin.python.scriptcreation.modules.Module_SimplifySurface;
import de.hlrs.starplugin.python.scriptcreation.modules.Module_Tracer;
import de.hlrs.starplugin.python.scriptcreation.modules.Module_Tube;
import de.hlrs.starplugin.util.FieldFunctionplusType;
import de.hlrs.starplugin.util.Vec;
import de.hlrs.starplugin.util.Wrapper_ReadEnsightModule_Port;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import star.common.Boundary;
import star.common.FieldFunction;
import star.common.Region;

/**
 *
 *@author Weiss HLRS Stuttgart
 */
public class PythonScript_ModuleManager {
    
    private ArrayList<Module_ReadEnsight> ReadEnsightList = new ArrayList<Module_ReadEnsight>();
    private ArrayList<Module_Renderer> RendererList = new ArrayList<Module_Renderer>();
    private ArrayList<Module_GetSubset> GetSubset_List = new ArrayList<Module_GetSubset>();
    private ArrayList<Module_Colors> Colors_List = new ArrayList<Module_Colors>();
    private ArrayList<Module_Collect> Collect_List = new ArrayList<Module_Collect>();
    private ArrayList<Module_CuttingSurface> CuttingSurface_List = new ArrayList<Module_CuttingSurface>();
    private ArrayList<Module_IsoSurface> IsoSurface_List = new ArrayList<Module_IsoSurface>();
    private ArrayList<Module_Tube> Tube_List = new ArrayList<Module_Tube>();
    private ArrayList<Module_Tracer> Tracer_List = new ArrayList<Module_Tracer>();
    private ArrayList<Module_SimplifySurface> SimplifySurface_List = new ArrayList<Module_SimplifySurface>();
    private ArrayList<Module_MinMax> MinMax_List = new ArrayList<Module_MinMax>();
    private ArrayList<Module_Material> Material_List = new ArrayList<Module_Material>();
    private ArrayList<Connection> ConnectionList = new ArrayList<Connection>();
    private HashMap<FieldFunction, Wrapper_ReadEnsightModule_Port> FieldFunctionPortHashMap = new HashMap<FieldFunction, Wrapper_ReadEnsightModule_Port>();
    private HashMap<FieldFunction, Module_Colors> ColormapsHashMap = new HashMap<FieldFunction, Module_Colors>();
    private PluginContainer PC;
    private int column = 0;

    public PythonScript_ModuleManager(PluginContainer PC) {
        this.PC = PC;

        RendererList.add(new Module_OpenCover("Renderer", Configuration_ModuleManager.XPosition_Distance * RendererList.size(),
                Configuration_ModuleManager.YPosition_Renderer));
        ReadEnsightList.add(new Module_ReadEnsight(Configuration_Module.Typ_ReadEnsight + "_0", Configuration_ModuleManager.ReadEnsight_XPosition * ReadEnsightList.size(), Configuration_ModuleManager.YPosition_ReadEnsight, this.PC.getEEDMan().getExportPath().
                getPath()));
    }

    public void createNet() {
        for (Entry<String, Construct> e : PC.getCNGDMan().getConMan().getConstructList().entrySet()) {
            if (e.getValue().checkValid()) {
                String Typ = e.getValue().getTyp();
                if (Typ.equals(Configuration_Tool.VisualizationType_Geometry)) {
                    this.addGeometryConstruct((Construct_GeometryVisualization) e.getValue());
                }
                if (Typ.equals(Configuration_Tool.VisualizationType_CuttingSurface)) {
                    this.addCuttingSurfaceConstruct((Construct_CuttingSurface) e.getValue());
                }
                if (Typ.equals(Configuration_Tool.VisualizationType_CuttingSurface_Series)) {
                    this.addCuttingSurfaceSeriesConstruct((Construct_CuttingSurfaceSeries) e.getValue());
                }
                if (Typ.equals(Configuration_Tool.VisualizationType_Streamline)) {
                    this.addStreamlineConstruct((Construct_Streamline) e.getValue());
                }
                if (Typ.equals(Configuration_Tool.VisualizationType_IsoSurface)) {
                    this.addIsoSurfaceConstruct((Construct_IsoSurface) e.getValue());
                }
            } else {
                //TODO if not Valid
            }
        }
    }

    private void addGeometryConstruct(Construct_GeometryVisualization Con) {

        if (this.is2DData(Con)) {
            String Selection = get2DSelection(Con);
            Module_GetSubset MGetSubset = addModule_GetSubsetBoundaries(Selection);

            if (Con.isShowdata()) {
                addConnection_REtoGETSUB_FF(Con, MGetSubset, Configuration_ModuleManager.D_2D);
                addColors_Collect_Modules(MGetSubset, Configuration_ModuleManager.D_2D, Con.getFFplType().
                        getFF());
            } else {
                Module_Material MMaterial = new Module_Material(Configuration_Module.Typ_Material + Configuration_Module.Underscore + this.Material_List.size(),
                        Configuration_ModuleManager.XPosition_Distance * this.getCollectionsMax(),
                        Configuration_ModuleManager.YPosition_Material);
                MMaterial.setMaterial(Con.getColor(), Con.getTransparency());
                Material_List.add(MMaterial);

                ConnectionList.add(new Connection(MGetSubset, Configuration_Module.OutputPorts_GetSubset[0],
                        MMaterial, Configuration_Module.InputPorts_Material[0]));



                this.addRendererConnection(MMaterial, Configuration_Module.OutputPorts_Material[0]);
            }
        }


        if (this.is3DData(Con)) {
            String Selection = get3DSelection(Con);
            Module_GetSubset MGetSubset = addModule_GetSubsetRegions(Selection);

            if (Con.isShowdata()) {
                addConnection_REtoGETSUB_FF(Con, MGetSubset, Configuration_ModuleManager.D_3D);
                addColors_Collect_Modules(MGetSubset, Configuration_ModuleManager.D_3D, Con.getFFplType().
                        getFF());
            } else {

                this.addRendererConnection(MGetSubset, Configuration_Module.OutputPorts_GetSubset[0]);
            }

        }

    }

    private void addCuttingSurfaceConstruct(Construct_CuttingSurface Con) {


        String Selection = this.get3DSelection(Con);
        Module_GetSubset MGetSubset = this.addModule_GetSubsetRegions(Selection);
        this.addConnection_REtoGETSUB_FF(Con, MGetSubset, Configuration_ModuleManager.D_3D);
        Module_CuttingSurface MCuttingSurface = addCuttingSurfaceModule(Con.getVertex(), Con.getPoint(), Con.getDistance(), MGetSubset);

        this.addColors_Collect_Modules(MCuttingSurface, Configuration_ModuleManager.D_2D, Con.getFFplType().
                getFF());




    }

    private void addCuttingSurfaceSeriesConstruct(Construct_CuttingSurfaceSeries Con) {
        String Selection = this.get3DSelection(Con);
        Module_GetSubset MGetSubset = this.addModule_GetSubsetRegions(Selection);
        this.addConnection_REtoGETSUB_FF(Con, MGetSubset, Configuration_ModuleManager.D_3D);
        for (int i = 0; i < Con.getAmount(); i++) {
            Vec NormalE = calcNormalE(Con.getVertex());
            Vec point2 = new Vec();

            point2.x = Con.getPoint().x + i * NormalE.x * Con.getDistanceBetween();
            point2.y = Con.getPoint().y + i * NormalE.y * Con.getDistanceBetween();
            point2.z = Con.getPoint().z + i * NormalE.z * Con.getDistanceBetween();
            //TODO als nächstes: Color module colormap verbinden module list connections clearen
            Module_CuttingSurface MCuttingSurface = addCuttingSurfaceModule(Con.getVertex(), point2, clacDistance(Con.getVertex(), point2), MGetSubset);
            this.addColors_Collect_Modules(MCuttingSurface, Configuration_ModuleManager.D_2D, Con.getFFplType().
                    getFF());
        }


    }

    private void addStreamlineConstruct(Construct_Streamline Con) {
        String Selection = this.get3DSelection(Con);
        Module_GetSubset MGetSubset = this.addModule_GetSubsetRegions(Selection);
        this.addConnection_REtoGETSUB_FF(Con, MGetSubset, Configuration_ModuleManager.D_3D);

        Module_Tracer MTracer = new Module_Tracer(
                Configuration_Module.Typ_Tracer + Configuration_Module.Underscore + this.Tracer_List.size(),
                Configuration_ModuleManager.XPosition_Distance * this.getCollectionsMax(),
                Configuration_ModuleManager.YPosition_Tracer);
        MTracer.setMaxOutOfDomain(Con.getMax_out_of_domain());
        MTracer.setTrace_len(Con.getTrace_length());
        MTracer.setTdirection(Con.getDirection());
        this.Tracer_List.add(MTracer);

        ConnectionList.add(new Connection(MGetSubset,
                Configuration_Module.OutputPorts_GetSubset[0], MTracer,
                Configuration_Module.InputPorts_Tracer[0]));
        ConnectionList.add(new Connection(MGetSubset,
                Configuration_Module.OutputPorts_GetSubset[1], MTracer,
                Configuration_Module.InputPorts_Tracer[1]));



        Module_Tube Mtube = new Module_Tube(Configuration_Module.Typ_Tube + Configuration_Module.Underscore + this.Tube_List.size(),
                Configuration_ModuleManager.XPosition_Distance * this.getCollectionsMax(),
                Configuration_ModuleManager.YPosition_Tube);
        Mtube.setRadius(Con.getTube_Radius());
        this.Tube_List.add(Mtube);

        ConnectionList.add(new Connection(MTracer,
                Configuration_Module.OutputPorts_Tracer[0], Mtube,
                Configuration_Module.InputPorts_Tube[0]));
        ConnectionList.add(new Connection(MTracer,
                Configuration_Module.OutputPorts_Tracer[1], Mtube,
                Configuration_Module.InputPorts_Tube[1]));

        this.addColors_Collect_Modules(Mtube, Configuration_ModuleManager.D_3D, Con.getFFplType().getFF());


        Module_GetSubset MGetSubsetInitialSurface = this.addModule_GetSubsetBoundaries(this.get2DInitialSurfaceSelection(Con));
        this.addConnection_REtoGETSUB_FF(Con, MGetSubsetInitialSurface, Configuration_ModuleManager.D_2D);

        Module_SimplifySurface MSimplifySurface = new Module_SimplifySurface(Configuration_Module.Typ_SimplifySurface + Configuration_Module.Underscore + this.SimplifySurface_List.size(),
                Configuration_ModuleManager.XPosition_Distance * this.getCollectionsMax(),
                Configuration_ModuleManager.YPosition_Tube);
        MSimplifySurface.setDivisions(Con.getDivisions());
        this.SimplifySurface_List.add(MSimplifySurface);

        ConnectionList.add(new Connection(MGetSubsetInitialSurface,
                Configuration_Module.OutputPorts_GetSubset[0], MSimplifySurface,
                Configuration_Module.InputPorts_SimplifySurface[0]));
        ConnectionList.add(new Connection(MSimplifySurface,
                Configuration_Module.OutputPorts_SimplifySurface[0], MTracer,
                Configuration_Module.InputPorts_Tracer[2]));
        this.column++;



    }

    private void addIsoSurfaceConstruct(Construct_IsoSurface Con) {
        String Selection = this.get3DSelection(Con);
        Module_GetSubset MGetSubset = this.addModule_GetSubsetRegions(Selection);
        this.addConnection_REtoGETSUB_FF(Con, MGetSubset, Configuration_ModuleManager.D_3D);

        Module_IsoSurface MIsoSurface = new Module_IsoSurface(Configuration_Module.Typ_IsoSurface + Configuration_Module.Underscore + this.IsoSurface_List.size(),
                Configuration_ModuleManager.XPosition_Distance * this.getCollectionsMax(),
                Configuration_ModuleManager.YPosition_IsoSurface);
        MIsoSurface.setIsoValue(new Vec(Con.getIsoValue(), Con.getIsoValue(), Con.getIsoValue()));


        ConnectionList.add(new Connection(MGetSubset,
                Configuration_Module.OutputPorts_GetSubset[0], MIsoSurface,
                Configuration_Module.InputPorts_IsoSurface[0]));

        ConnectionList.add(new Connection(MGetSubset,
                Configuration_Module.OutputPorts_GetSubset[1], MIsoSurface,
                Configuration_Module.InputPorts_IsoSurface[1]));

        this.IsoSurface_List.add(MIsoSurface);

        this.addColors_Collect_Modules(MIsoSurface, Configuration_ModuleManager.D_3D,
                Con.getFFplType().getFF());
    }

    public void add(Module_ReadEnsight Module) {
        ReadEnsightList.add(Module);
    }

    public void addRendererConnection(Module m, String Port) {
        ConnectionList.add(new Connection(m, Port, RendererList.get(0),
                Configuration_Module.InputPorts_OpenCOVER[0]));
        column++;

    }

    public ArrayList<Module_ReadEnsight> getReadEnsightList() {
        return ReadEnsightList;
    }

    public ArrayList<Module> getAllModules() {
        ArrayList<Module> tmpList = new ArrayList<Module>();
        tmpList.addAll(ReadEnsightList);
        tmpList.addAll(RendererList);
        tmpList.addAll(GetSubset_List);
        tmpList.addAll(Colors_List);
        tmpList.addAll(Collect_List);
        tmpList.addAll(CuttingSurface_List);
        tmpList.addAll(this.Tracer_List);
        tmpList.addAll(this.Tube_List);
        tmpList.addAll(this.SimplifySurface_List);
        tmpList.addAll(this.IsoSurface_List);
        tmpList.addAll(this.MinMax_List);
        tmpList.addAll(this.Material_List);


        return tmpList;
    }

    public ArrayList<Connection> getConnectionList() {
        return ConnectionList;
    }

    private void addFieldFunctiontoReadEnsight(FieldFunctionplusType fFplType) {
        String Type = fFplType.getType();
        FieldFunction FF = fFplType.getFF();
        boolean done = false;
        if (Type.equals(Configuration_Tool.DataType_scalar)) {
            for (Module_ReadEnsight MRES : ReadEnsightList) {
                if (!done && MRES.getMetaData().getScalarFieldFunctionCount() < 3) {
                    MRES.getMetaData().addScalarFieldFunction(FF, MRES.getMetaData().
                            getScalarFieldFunctionCount(), PC.getEEMan().getScalarsList().get(FF));
                    int[] Port = new int[3];
                    switch (MRES.getMetaData().getScalarFieldFunctionCount()) {
                        case 1:
                            Port[0] = 1;
                            Port[1] = 7;
                            break;

                        case 2:
                            Port[0] = 2;
                            Port[1] = 8;
                            break;
                        case 3:
                            Port[0] = 3;
                            Port[1] = 9;
                            break;
                    }
                    FieldFunctionPortHashMap.put(FF,
                            new Wrapper_ReadEnsightModule_Port(MRES, Port[0], Port[1]));

                    done = true;
                }
            }
            if (!done) {

                Module_ReadEnsight newMRES = new Module_ReadEnsight(Configuration_Module.Typ_ReadEnsight + Configuration_Module.Underscore + ReadEnsightList.size(), Configuration_ModuleManager.ReadEnsight_XPosition * ReadEnsightList.size(),
                        Configuration_ModuleManager.YPosition_ReadEnsight, this.PC.getEEDMan().
                        getExportPath().
                        getPath());
                newMRES.getMetaData().addScalarFieldFunction(FF, newMRES.getMetaData().
                        getScalarFieldFunctionCount(), PC.getEEMan().getScalarsList().get(FF));
                ReadEnsightList.add(newMRES);
                FieldFunctionPortHashMap.put(FF, new Wrapper_ReadEnsightModule_Port(newMRES, 1, 7));//TODO Hardcoded
                done = true;
            }

        }
        if (Type.equals(Configuration_Tool.DataType_vector)) {
            for (Module_ReadEnsight MRES : ReadEnsightList) {
                if (!done && MRES.getMetaData().getVectorFieldFunctionCount() < 2) {
                    MRES.getMetaData().addVectorFieldFunction(FF, MRES.getMetaData().
                            getVectorFieldFunctionCount(), PC.getEEMan().getVectorsList().get(FF));
                    int[] Port = new int[2];
                    switch (MRES.getMetaData().getVectorFieldFunctionCount()) {
                        case 1:
                            Port[0] = 4;
                            Port[1] = 10;
                            break;

                        case 2:
                            Port[0] = 5;
                            Port[1] = 11;
                            break;
                    }
                    FieldFunctionPortHashMap.put(FF,
                            new Wrapper_ReadEnsightModule_Port(MRES, Port[0], Port[1]));
                    done = true;
                }
            }
            if (!done) {
                Module_ReadEnsight newMRES = new Module_ReadEnsight(Configuration_Module.Typ_ReadEnsight + Configuration_Module.Underscore + ReadEnsightList.size(), Configuration_ModuleManager.ReadEnsight_XPosition * ReadEnsightList.size(),
                        Configuration_ModuleManager.YPosition_ReadEnsight, this.PC.getEEDMan().
                        getExportPath().
                        getPath());
                newMRES.getMetaData().addVectorFieldFunction(FF, newMRES.getMetaData().
                        getVectorFieldFunctionCount(), PC.getEEMan().getVectorsList().get(FF));
                ReadEnsightList.add(newMRES);
                FieldFunctionPortHashMap.put(FF, new Wrapper_ReadEnsightModule_Port(newMRES, 4, 10));//TODO Hardcoded
                done = true;
            }
        }
        this.addMinMaxColorMap(FF);
    }

    private void addMinMaxColorMap(FieldFunction FF) {

        Module_MinMax MMinMax = new Module_MinMax(Configuration_Module.Typ_MinMax + Configuration_Module.Underscore + this.MinMax_List.size(),
                Configuration_ModuleManager.XPosition_MinMax + ((this.MinMax_List.size() % 3) * -1 * Configuration_ModuleManager.XPosition_MinMax / 3),
                Configuration_ModuleManager.YPosition_MinMax * (((this.MinMax_List.size() - this.MinMax_List.size() % 3)) / 3));
        Module_Colors MColors = new Module_Colors(Configuration_Module.Typ_Colors + Configuration_Module.Underscore + this.Colors_List.size(),
                Configuration_ModuleManager.XPosition_MinMax + ((this.MinMax_List.size() % 3) * -1 * Configuration_ModuleManager.XPosition_MinMax / 3),
                Configuration_ModuleManager.YPosition_MinMax * (((this.MinMax_List.size() - this.MinMax_List.size() % 3)) / 3) + Configuration_ModuleManager.YPosition_MinMax / 2);
        MColors.setAnnotation(FF.getPresentationName());

        this.MinMax_List.add(MMinMax);
        this.Colors_List.add(MColors);
        this.ConnectionList.add(new Connection(MMinMax, Configuration_Module.OutputPorts_MinMax[2], MColors,
                Configuration_Module.InputPorts_Colors[3]));

        Wrapper_ReadEnsightModule_Port W_RE_P = FieldFunctionPortHashMap.get(FF);
        ConnectionList.add(new Connection(W_RE_P.getRE_Module(), Configuration_Module.OutputPorts_ReadEnsight[W_RE_P.getPort()[0]], MMinMax, Configuration_Module.InputPorts_MinMax[0]));
        this.ColormapsHashMap.put(FF, MColors);

    }

    private int getCollectionsMax() {
        return column;
    }

    private boolean is3DData(Construct Con) {
        ArrayList<Integer> parts = new ArrayList<Integer>();
        for (Entry<Object, Integer> e : Con.getParts().entrySet()) {
            parts.add(e.getValue());
            if (e.getKey() instanceof Region) {
                return true;
            }
        }
        return false;
    }

    private boolean is2DData(Construct Con) {
        ArrayList<Integer> parts = new ArrayList<Integer>();
        for (Entry<Object, Integer> e : Con.getParts().entrySet()) {
            parts.add(e.getValue());
            if (e.getKey() instanceof Boundary) {
                return true;
            }
        }
        return false;
    }

    private void addColors_Collect_Modules(Module Module, Integer Dimension, FieldFunction FF) {
        Module_Colors MColor = new Module_Colors(
                Configuration_Module.Typ_Colors + Configuration_Module.Underscore + this.Colors_List.size(),
                Configuration_ModuleManager.XPosition_Distance * this.getCollectionsMax(),
                Configuration_ModuleManager.YPosition_Colors);
        Colors_List.add(MColor);
        Module_Collect MCollect = new Module_Collect(Configuration_Module.Typ_Collect + Configuration_Module.Underscore + this.Collect_List.size(), Configuration_ModuleManager.XPosition_Distance * this.getCollectionsMax(),
                Configuration_ModuleManager.YPosition_Collect);
        Collect_List.add(MCollect);
        if (Module instanceof Module_GetSubset) {
            ConnectionList.add(new Connection(Module,
                    Configuration_Module.OutputPorts_GetSubset[0], MCollect,
                    Configuration_Module.InputPorts_Collect[0]));
            ConnectionList.add(new Connection(Module,
                    Configuration_Module.OutputPorts_GetSubset[1], MColor,
                    Configuration_Module.InputPorts_Colors[0]));
        }
        if (Module instanceof Module_CuttingSurface) {
            ConnectionList.add(new Connection(Module,
                    Configuration_Module.OutputPorts_CuttingSurface[0], MCollect,
                    Configuration_Module.InputPorts_Collect[0]));
            ConnectionList.add(new Connection(Module,
                    Configuration_Module.OutputPorts_CuttingSurface[1], MColor,
                    Configuration_Module.InputPorts_Colors[0]));
        }
        if (Module instanceof Module_Tube) {
            ConnectionList.add(new Connection(Module,
                    Configuration_Module.OutputPorts_Tube[0], MCollect,
                    Configuration_Module.InputPorts_Collect[0]));
            ConnectionList.add(new Connection(Module,
                    Configuration_Module.OutputPorts_Tube[1], MColor,
                    Configuration_Module.InputPorts_Colors[0]));

        }
        if (Module instanceof Module_IsoSurface) {
            ConnectionList.add(new Connection(Module,
                    Configuration_Module.OutputPorts_IsoSurface[0], MCollect,
                    Configuration_Module.InputPorts_Collect[0]));
            ConnectionList.add(new Connection(Module,
                    Configuration_Module.OutputPorts_IsoSurface[1], MColor,
                    Configuration_Module.InputPorts_Colors[0]));

        }

        if (Dimension.equals(Configuration_ModuleManager.D_2D)) {
            ConnectionList.add(new Connection(MColor,
                    Configuration_Module.OutputPorts_Colors[1], MCollect,
                    Configuration_Module.InputPorts_Collect[3]));
        }

        if (Dimension.equals(Configuration_ModuleManager.D_3D)) {
            ConnectionList.add(new Connection(MColor,
                    Configuration_Module.OutputPorts_Colors[0], MCollect,
                    Configuration_Module.InputPorts_Collect[1]));
        }
        this.ConnectionList.add(new Connection(this.ColormapsHashMap.get(FF),
                Configuration_Module.OutputPorts_Colors[2], MColor, Configuration_Module.InputPorts_Colors[3]));

        this.addRendererConnection(MCollect,
                Configuration_Module.OutputPorts_Collect[0]);
    }

    private void addConnection_REtoGETSUB_FF(Construct Con, Module_GetSubset MGetSubset, Integer Dimension) {

        if (Dimension.equals(Configuration_ModuleManager.D_3D)) {
            if (FieldFunctionPortHashMap.containsKey(Con.getFFplType().getFF())) {
                Wrapper_ReadEnsightModule_Port W_RE_P = FieldFunctionPortHashMap.get(Con.getFFplType().getFF());
                ConnectionList.add(new Connection(W_RE_P.getRE_Module(), Configuration_Module.OutputPorts_ReadEnsight[W_RE_P.getPort()[0]], MGetSubset, Configuration_Module.InputPorts_GetSubset[1]));
            } else {
                addFieldFunctiontoReadEnsight(Con.getFFplType());
                Wrapper_ReadEnsightModule_Port W_RE_P = FieldFunctionPortHashMap.get(Con.getFFplType().getFF());
                ConnectionList.add(new Connection(W_RE_P.getRE_Module(),
                        Configuration_Module.OutputPorts_ReadEnsight[W_RE_P.getPort()[0]], MGetSubset,
                        Configuration_Module.InputPorts_GetSubset[1]));
            }
        }
        if (Dimension.equals(Configuration_ModuleManager.D_2D)) {
            if (FieldFunctionPortHashMap.containsKey(Con.getFFplType().getFF())) {
                Wrapper_ReadEnsightModule_Port W_RE_P = FieldFunctionPortHashMap.get(Con.getFFplType().getFF());
                ConnectionList.add(new Connection(W_RE_P.getRE_Module(), Configuration_Module.OutputPorts_ReadEnsight[W_RE_P.getPort()[1]], MGetSubset, Configuration_Module.InputPorts_GetSubset[1]));
            } else {
                addFieldFunctiontoReadEnsight(Con.getFFplType());
                Wrapper_ReadEnsightModule_Port W_RE_P = FieldFunctionPortHashMap.get(Con.getFFplType().
                        getFF());
                ConnectionList.add(new Connection(W_RE_P.getRE_Module(), Configuration_Module.OutputPorts_ReadEnsight[W_RE_P.getPort()[1]], MGetSubset, Configuration_Module.InputPorts_GetSubset[1]));

            }
        }
    }

    private Module_GetSubset addModule_GetSubsetRegions(String Selection) {
        Module_GetSubset MGetSubset = new Module_GetSubset(Configuration_Module.Typ_GetSubset + Configuration_Module.Underscore + this.GetSubset_List.size(),
                Configuration_ModuleManager.XPosition_Distance * this.getCollectionsMax(),
                Configuration_ModuleManager.YPosition_GetSubset, Selection);
        GetSubset_List.add(MGetSubset);
        ConnectionList.add(new Connection(ReadEnsightList.get(0),
                Configuration_Module.OutputPorts_ReadEnsight[0], MGetSubset,
                Configuration_Module.InputPorts_GetSubset[0]));
        return MGetSubset;
    }

    private Module_GetSubset addModule_GetSubsetBoundaries(String Selection) {
        Module_GetSubset MGetSubset = new Module_GetSubset(Configuration_Module.Typ_GetSubset + Configuration_Module.Underscore + this.GetSubset_List.size(),
                Configuration_ModuleManager.XPosition_Distance * this.getCollectionsMax(),
                Configuration_ModuleManager.YPosition_GetSubset, Selection);
        GetSubset_List.add(MGetSubset);
        ConnectionList.add(new Connection(ReadEnsightList.get(0),
                Configuration_Module.OutputPorts_ReadEnsight[6], MGetSubset,
                Configuration_Module.InputPorts_GetSubset[0]));
        return MGetSubset;
    }

    private String get2DSelection(Construct Con) {
        String Selection = "";
        for (Entry<Object, Integer> e : Con.getParts().entrySet()) {
            if (e.getKey() instanceof Boundary) {
                Selection = Selection.concat(
                        PC.getEEMan().getBoundariesList().get(e.getKey()).toString() + ";");
            }
        }
        return Selection;
    }
    private String get2DInitialSurfaceSelection(Construct_Streamline Con){
                String Selection = "";
        for (Entry<Object, Integer> e : Con.getInitialSurface().entrySet()) {
            if (e.getKey() instanceof Boundary) {
                Selection = Selection.concat(
                        PC.getEEMan().getBoundariesList().get(e.getKey()).toString() + ";");
            }
        }
        return Selection;
    }

    private String get3DSelection(Construct Con) {
        String Selection = "";
        for (Entry<Object, Integer> e : Con.getParts().entrySet()) {
            if (e.getKey() instanceof Region) {
                Selection = Selection.concat(PC.getEEMan().getRegionsList().get(e.getKey()).toString() + ";");
            }
        }
        return Selection;
    }

    private Module_CuttingSurface addCuttingSurfaceModule(Vec Vertex, Vec point, float Distance, Module_GetSubset MGetSubset) {
        Module_CuttingSurface MCuttingSurface = new Module_CuttingSurface(Configuration_Module.Typ_CuttingSurface + Configuration_Module.Underscore + this.CuttingSurface_List.size(),
                Configuration_ModuleManager.XPosition_Distance * this.getCollectionsMax(),
                Configuration_ModuleManager.YPosition_CuttingSurface, Vertex, point,
                Distance);
        CuttingSurface_List.add(MCuttingSurface);
        ConnectionList.add(new Connection(MGetSubset,
                Configuration_Module.OutputPorts_GetSubset[0], MCuttingSurface,
                Configuration_Module.InputPorts_CuttingSurface[0]));
        ConnectionList.add(new Connection(MGetSubset,
                Configuration_Module.OutputPorts_GetSubset[1], MCuttingSurface,
                Configuration_Module.InputPorts_CuttingSurface[1]));
        return MCuttingSurface;
    }

    private float clacDistance(Vec Normal, Vec point) {

        float Nenner = -(-point.x * Normal.x - point.y * Normal.y - point.z * Normal.z);
        float Zaehler = (float) Math.sqrt(
                Math.pow(Normal.x, 2) + Math.pow(Normal.y, 2) + Math.pow(Normal.z, 2));

        return Nenner / Zaehler;
    }

    private Vec calcNormalE(Vec v) {
        float Betrag = (float) Math.sqrt(Math.pow(v.x, 2) + Math.pow(v.y, 2) + Math.pow(v.z, 2));
        Vec a = new Vec(0, 0, 0);
        a.x = v.x / Betrag;
        a.y = v.y / Betrag;
        a.z = v.z / Betrag;
        return a;
    }
}
