package de.hlrs.starplugin.covise_net_generation.constructs;

import de.hlrs.starplugin.configuration.Configuration_GUI_Strings;
import de.hlrs.starplugin.covise_net_generation.CoviseNetGeneration_DataManager;
import de.hlrs.starplugin.gui.dialogs.Error_Dialog;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.HashMap;

/**
 *
 *  @author Weiss HLRS Stuttgart
 */
public class Construct_Manager {

    private CoviseNetGeneration_DataManager CNGDMan;
    //Covise Netz Generierung
    private HashMap<String, Construct> ConstructList;
    private Construct SelectedConstruct;
    //Funktionalität
    private int GeometryCount = 0;
    private int CuttingSurfaceCount = 0;
    private int CuttingSurfaceSeriesCount = 0;
    private int StreamlineCount = 0;
    private int IsoSurfaceCount = 0;
    private boolean SelectionChangeing = false;

    public Construct_Manager(CoviseNetGeneration_DataManager cngdman) {
        this.CNGDMan = cngdman;
        this.ConstructList = new HashMap<String, Construct>();
    }

    public void addConsturct(Construct construct) {
        if (construct instanceof Construct_GeometryVisualization) {
            GeometryCount++;
        }
        if (construct instanceof Construct_CuttingSurface && !(construct instanceof Construct_CuttingSurfaceSeries)) {
            CuttingSurfaceCount++;
        }
        if (construct instanceof Construct_CuttingSurfaceSeries) {
            CuttingSurfaceSeriesCount++;
        }
        if (construct instanceof Construct_Streamline) {
            StreamlineCount++;
        }
        if (construct instanceof Construct_IsoSurface) {
            IsoSurfaceCount++;
        }
        this.addConstructToConstructList(construct);
        this.CNGDMan.onChange("add");
    }

    public void deleteConstruct(String Key) {
        ConstructList.remove(Key);
        this.CNGDMan.onChange("delete");
    }

    public int getGeometryCount() {
        return GeometryCount;
    }

    public int getCuttingSurfaceCount() {
        return CuttingSurfaceCount;
    }

    public int getCuttingSurfaceSeriesCount() {
        return CuttingSurfaceSeriesCount;
    }

    public int getIsoSurfaceCount() {
        return IsoSurfaceCount;
    }

    public int getStreamlineCount() {
        return StreamlineCount;
    }

    public Construct getSelectedConstruct() {
        return SelectedConstruct;
    }

    public void setSelectedConstruct(Construct Construct) {
        if (!SelectionChangeing) {
            try {
                SelectionChangeing = true;
//                this.CNGDMan.getPC().getSim().print(SelectionChangeing);
                this.CNGDMan.getPC().disableEventManager();
                this.SelectedConstruct = Construct;
                this.CNGDMan.onChange("Selection");
                this.CNGDMan.getPC().enableEventManager();
                SelectionChangeing = false;
//                this.CNGDMan.getPC().getSim().println(SelectionChangeing);
            } catch (Exception Ex) {
                StringWriter sw = new StringWriter();
                Ex.printStackTrace(new PrintWriter(sw));
                new Error_Dialog(Configuration_GUI_Strings.Occourence + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.ErrMass + Ex.getMessage() + Configuration_GUI_Strings.eol + Configuration_GUI_Strings.StackTrace + sw.toString());
                this.CNGDMan.getPC().enableEventManager();
            }
        }


    }

    public void clearConstructs() {
        ConstructList.clear();
        GeometryCount = 0;
        CuttingSurfaceCount = 0;
        CuttingSurfaceSeriesCount = 0;
        StreamlineCount = 0;
        IsoSurfaceCount = 0;
        SelectedConstruct = null;
        this.CNGDMan.onChange("delete");
    }

    public HashMap<String, Construct> getConstructList() {
        return ConstructList;
    }

    private void addConstructToConstructList(Construct Con) {
        int i = 1;
        while (ConstructList.containsKey(Con.toString())) {
            String newName = Con.getName() + "(" + i + ")";
            Con.setName(newName);
            i++;
        }
        this.ConstructList.put(Con.toString(), Con);
    }
}


