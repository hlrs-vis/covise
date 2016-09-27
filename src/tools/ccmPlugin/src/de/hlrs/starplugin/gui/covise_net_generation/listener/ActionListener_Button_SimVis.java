package de.hlrs.starplugin.gui.covise_net_generation.listener;

import Main.PluginContainer;
import de.hlrs.starplugin.configuration.Configuration_GUI_Strings;
import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_CuttingSurface;
import de.hlrs.starplugin.covise_net_generation.constructs.Construct_Streamline;
import de.hlrs.starplugin.gui.dialogs.Error_Dialog;
import de.hlrs.starplugin.util.FieldFunctionplusType;
import de.hlrs.starplugin.util.Vec;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Collection;
import java.util.Iterator;
import star.base.neo.DoubleVector;
import star.base.neo.IntVector;
import star.base.neo.NamedObject;
import star.common.Boundary;
import star.common.FieldFunction;
import star.common.Part;
import star.common.PartGroup;
import star.common.Region;
import star.common.Simulation;
import star.vis.Displayer;
import star.vis.DisplayerManager;
import star.vis.PlaneSection;
import star.vis.RK2;
import star.vis.ScalarDisplayer;
import star.vis.Scene;
import star.vis.SceneManager;
import star.vis.StreamPart;

/**
 *
 *@author Weiss HLRS Stuttgart
 */
public class ActionListener_Button_SimVis implements ActionListener {

    private PluginContainer PC;

    public ActionListener_Button_SimVis(PluginContainer pC) {
        super();
        this.PC = pC;

    }

    public void actionPerformed(ActionEvent e) {
        Simulation Sim = PC.getSim();
        SceneManager SM = Sim.getSceneManager();
        Collection<Scene> CS = SM.getScenes();
        Iterator<Scene> IS = CS.iterator();
        while (IS.hasNext()) {
            DisplayerManager DM = IS.next().getDisplayerManager();
            Collection<Displayer> CD = DM.getNonDummyObjects();
            Iterator<Displayer> ID = CD.iterator();
            while (ID.hasNext()) {
                Displayer D = ID.next();
                if (D instanceof ScalarDisplayer) {
                    ScalarDisplayer SD = (ScalarDisplayer) D;
                    // Field Function
                    FieldFunctionplusType FFpT;
                    FieldFunction FF = SD.getScalarDisplayQuantity().getFieldFunction();
                    if (PC.getSIM_DATA_MANGER().getAllFieldFunctionplusTypeHashMap().containsKey(
                            FF.getPresentationName())) {
                        FFpT = new FieldFunctionplusType(FF, PC.getSIM_DATA_MANGER().getAllFieldFunctionplusTypeHashMap().
                                get(FF.getPresentationName()).getType());

                        if (FFpT.getType().equals(Configuration_Tool.DataType_scalar) && !PC.getEEDMan().
                                getScalarsSelectedList().contains(FF)) {
                            PC.getEEDMan().addSelectedScalar(FF);
                        }
                        if (FFpT.getType().equals(Configuration_Tool.DataType_vector) && !PC.getEEDMan().
                                getVectorsSelectedList().contains(FF)) {
                            PC.getEEDMan().addSelectedVector(FF);
                        }
                    } else {
                        FFpT = null;
                    }

                    //Derived Parts
                    PartGroup PG = D.getInputParts();
                    Collection<NamedObject> CP = PG.getParts();
                    Iterator<NamedObject> I = CP.iterator();
                    while (I.hasNext()) {
                        NamedObject NO = I.next();
                        //Schnittflächen
                        if (NO instanceof Part) {
                            Part P = (Part) NO;
                            if (P instanceof PlaneSection) {
                                try {
                                    PlaneSection PS = (PlaneSection) P;
                                    Construct_CuttingSurface Con_CS = addCuttingSurface(PS, FFpT);
                                    PC.getCNGDMan().getConMan().addConsturct(Con_CS);
                                } catch (Exception ex) {
                                }
                            }
                            if (P instanceof StreamPart) {
                                try {
                                    StreamPart SP = (StreamPart) P;
                                    Construct_Streamline Con_SL = addStreamline(SP);
                                    PC.getCNGDMan().getConMan().addConsturct(Con_SL);
                                } catch (Exception ex) {
                                }
                            }

                        }
                    }
                }
            }
        }

    }

    private Construct_CuttingSurface addCuttingSurface(PlaneSection PS, FieldFunctionplusType FFpT) throws Exception {
        try {
            DoubleVector DV1 = PS.getOrigin();
            Vec VOrigin = new Vec(DV1.get(0), DV1.get(1), DV1.get(2));
            PS.getOriginCoordinate();
            DoubleVector Normal = PS.getOrientation();
            Vec VNormal = new Vec(Normal.get(0), Normal.get(1), Normal.get(2));

            Vec NormalE = calcNormalE(VNormal);
            float Distance = calcDistance(NormalE, VOrigin);
            Construct_CuttingSurface Con_CS = new Construct_CuttingSurface(VNormal, VOrigin, Distance);
            Con_CS.setName(PS.getPresentationName());
            Con_CS.setTyp(Configuration_Tool.VisualizationType_CuttingSurface);
            Con_CS.setFFplType(FFpT);

            //Parts of Construct
            Collection<NamedObject> CPSF = PS.getInputParts().getObjects();
            Iterator<NamedObject> I_CPSF = CPSF.iterator();
            while (I_CPSF.hasNext()) {
                Object O = I_CPSF.next();
                if (O instanceof Region) {
                    Region R = (Region) O;
                    if (!PC.getEEDMan().getRegionsSelectedList().contains(R)) {
                        PC.getEEDMan().addSelectedRegion(R);
                    }

                    Con_CS.addPart(R, PC.getEEMan().getPartsList().get(R));


                }
                if (O instanceof Boundary) {
                    Boundary B = (Boundary) O;
                    if (!PC.getEEDMan().getBoundariesSelectedList().contains(B)) {
                        PC.getEEDMan().addSelectedBoundary(B);
                    }
                    Con_CS.addPart(B, PC.getEEMan().getPartsList().get(B));

                }
            }
            //Add Construct to ConMan
            return Con_CS;
        } catch (Exception Ex) {
            StringWriter sw = new StringWriter();
            Ex.printStackTrace(new PrintWriter(sw));
            new Error_Dialog(Configuration_GUI_Strings.Occourence + Configuration_GUI_Strings.eol
                    + Configuration_GUI_Strings.ErrMass + Ex.getMessage() + Configuration_GUI_Strings.eol
                    + Configuration_GUI_Strings.StackTrace + sw.toString());
            throw Ex;
        }
    }

    private Construct_Streamline addStreamline(StreamPart SP) throws Exception {
        try {
            Construct_Streamline Con_SL = new Construct_Streamline();
            Con_SL.setName(SP.getPresentationName());
            Con_SL.setTyp(Configuration_Tool.VisualizationType_Streamline);

            //    FieldFunction
            FieldFunctionplusType FFpT;
            FieldFunction FF = SP.getFieldFunction();
            if (PC.getSIM_DATA_MANGER().getAllFieldFunctionplusTypeHashMap().containsKey(FF.getPresentationName())) {
                FFpT = new FieldFunctionplusType(FF, PC.getSIM_DATA_MANGER().getAllFieldFunctionplusTypeHashMap().get(FF.
                        getPresentationName()).getType());

                if (FFpT.getType().equals(Configuration_Tool.DataType_scalar) && !PC.getEEDMan().getScalarsSelectedList().
                        contains(FF)) {
                    PC.getEEDMan().addSelectedScalar(FF);
                }
                if (FFpT.getType().equals(Configuration_Tool.DataType_vector) && !PC.getEEDMan().getVectorsSelectedList().
                        contains(FF)) {
                    PC.getEEDMan().addSelectedVector(FF);
                }
            } else {
                FFpT = null;
            }
            Con_SL.setFFplType(FFpT);

            //Divisions
            IntVector IV = SP.getSourceSeed().getNGridPoints();
            int size = IV.size();
            Vec Divisions = new Vec(1, 1, 1);
            for (int i = 0; i < size && i < 3; i++) {
                if (i == 0) {
                    Divisions.x = IV.get(i);
                }
                if (i == 1) {
                    Divisions.y = IV.get(i);
                }
                if (i == 2) {
                    Divisions.z = IV.get(i);
                }
                Con_SL.setDivisions(Divisions);
            }
            Con_SL.setDivisions(Divisions);
            //Direction
            RK2 rk2 = SP.getSecondOrderIntegrator();
            if (rk2.getDirection() == 0) {
                Con_SL.setDirection(Configuration_Tool.forward);
            }
            if (rk2.getDirection() == 0) {
                Con_SL.setDirection(Configuration_Tool.back);
            }
            if (rk2.getDirection() == 0) {
                Con_SL.setDirection(Configuration_Tool.both);
            }

            //Parts of Construct
            Collection<NamedObject> CPSF = SP.getInputParts().getObjects();
            Iterator<NamedObject> I_CPSF = CPSF.iterator();
            while (I_CPSF.hasNext()) {
                Object O = I_CPSF.next();
                if (O instanceof Region) {
                    Region R = (Region) O;
                    if (!PC.getEEDMan().getRegionsSelectedList().contains(R)) {
                        PC.getEEDMan().addSelectedRegion(R);
                    }

                    Con_SL.addPart(R, PC.getEEMan().getPartsList().get(R));
                }
                if (O instanceof Boundary) {
                    Boundary B = (Boundary) O;
                    if (!PC.getEEDMan().getBoundariesSelectedList().contains(B)) {
                        PC.getEEDMan().addSelectedBoundary(B);
                    }
                    Con_SL.addPart(B, PC.getEEMan().getPartsList().get(B));
                }
            }
            //Initial Surface over Seed Part
            Collection<NamedObject> SPO = SP.getSourceSeed().getSeedParts().getObjects();
            Iterator<NamedObject> I_SPO = SPO.iterator();
            while (I_SPO.hasNext()) {
                Object O = I_SPO.next();

                if (O instanceof Boundary) {
                    Boundary B = (Boundary) O;
                    if (!PC.getEEDMan().getBoundariesSelectedList().contains(B)) {
                        PC.getEEDMan().addSelectedBoundary(B);
                    }
                    if (PC.getEEMan().getPartsList().containsKey(B)) {
                        Con_SL.addInitialSurface(B, PC.getEEMan().getPartsList().get(B));
                    }
                }
            }


            //Add Construct to ConMan
            return Con_SL;
        } catch (Exception Ex) {

            StringWriter sw = new StringWriter();
            Ex.printStackTrace(new PrintWriter(sw));
            new Error_Dialog(Configuration_GUI_Strings.Occourence + Configuration_GUI_Strings.eol
                    + Configuration_GUI_Strings.ErrMass + Ex.getMessage() + Configuration_GUI_Strings.eol
                    + Configuration_GUI_Strings.StackTrace + sw.toString());
            throw Ex;
        }

    }

    private float calcDistance(Vec Normal, Vec point) {

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
