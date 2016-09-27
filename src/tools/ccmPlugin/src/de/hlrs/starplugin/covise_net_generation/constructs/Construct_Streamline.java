package de.hlrs.starplugin.covise_net_generation.constructs;

import de.hlrs.starplugin.load_save.Serializable_Construct_Streamline;
import de.hlrs.starplugin.util.FieldFunctionplusType;
import de.hlrs.starplugin.util.Vec;
import java.util.HashMap;
import star.common.Boundary;

/**
 *
 *  @author Weiss HLRS Stuttgart
 */
public class Construct_Streamline extends Construct {

    private Vec Divisions;
    private float Tube_Radius;
    private float trace_length;
    private float max_out_of_domain;
    private String direction;
    private HashMap<Object, Integer> InitialSurface = new HashMap<Object, Integer>();

    public Construct_Streamline() {
        this.Divisions = new Vec(2, 2, 2);
        this.Tube_Radius = 0.001f;
        this.trace_length = 100;
        this.max_out_of_domain = 0.5f;
        this.direction = "1";
    }

    public Vec getDivisions() {
        return Divisions;
    }

    public void setDivisions(Vec Divisions) {
        this.Divisions = Divisions;
    }

    public HashMap<Object, Integer> getInitialSurface() {
        return InitialSurface;
    }

    public void setInitialSurface(HashMap<Object, Integer> InitialSurface) {
        this.InitialSurface = InitialSurface;
    }

    public void addInitialSurface(Boundary B, Integer i) {
        if (InitialSurface == null) {
            InitialSurface = new HashMap<Object, Integer>();
        }
        this.InitialSurface.put(B, i);
    }

    public void setInitialSurface(Boundary B, Integer i) {
        this.InitialSurface.clear();
        this.InitialSurface.put(B, i);
    }

    public float getTube_Radius() {
        return Tube_Radius;
    }

    public void setTube_Radius(float Tube_Radius) {
        this.Tube_Radius = Tube_Radius;
    }

    public float getMax_out_of_domain() {
        return max_out_of_domain;
    }

    public void setMax_out_of_domain(float max_out_of_domain) {
        this.max_out_of_domain = max_out_of_domain;
    }

    public float getTrace_length() {
        return trace_length;
    }

    public void setTrace_length(float trace_length) {
        this.trace_length = trace_length;
    }

    public String getDirection() {
        return direction;
    }

    public void setDirection(String direction) {
        this.direction = direction;
    }

    public void modify(Construct_Streamline Con) {

        this.Divisions = Con.getDivisions();
        this.Tube_Radius = Con.getTube_Radius();
        this.trace_length = Con.getTrace_length();
        this.max_out_of_domain = Con.getMax_out_of_domain();
        this.direction = Con.getDirection();
        this.InitialSurface = new HashMap<Object, Integer>(Con.getInitialSurface());
        this.Parts = new HashMap<Object, Integer>(Con.getParts());
        this.FFplType = Con.getFFplType();
    }

    public Construct_Streamline(Serializable_Construct_Streamline SCon, FieldFunctionplusType FFpT, HashMap<Object, Integer> sParts, HashMap<Object, Integer> HM_InitialSurface) {
        super(SCon, FFpT, sParts);
        Divisions = SCon.getDivisions();
        Tube_Radius = SCon.getTube_Radius();
        trace_length = SCon.getTrace_length();
        max_out_of_domain = SCon.getMax_out_of_domain();
        direction = SCon.getDirection();
        if (HM_InitialSurface != null) {
            if (!HM_InitialSurface.isEmpty()) {
                this.InitialSurface = new HashMap<Object, Integer>();
                this.InitialSurface.putAll(HM_InitialSurface);
            } else {
                this.InitialSurface = new HashMap<Object, Integer>();
            }
        }
    }

    @Override
    public boolean checkValid() {
        if (!super.checkValid()) {
            return false;
        }
        if (this.InitialSurface.isEmpty()) {
            return false;
        }
        return true;
    }
}
