package de.hlrs.starplugin.load_save;

import de.hlrs.starplugin.covise_net_generation.constructs.Construct_Streamline;
import de.hlrs.starplugin.util.Vec;
import java.util.ArrayList;
import java.util.Map.Entry;
import star.common.Boundary;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Serializable_Construct_Streamline extends Serializable_Construct {

    private static final long serialVersionUID = 1310248353023656286L;
    private Vec Divisions;
    private float Tube_Radius;
    private float trace_length;
    private float max_out_of_domain;
    private String direction;
    private ArrayList<String> InitialSurface;

    public Serializable_Construct_Streamline() {
    }

    public Serializable_Construct_Streamline(Vec Divisions, float Tube_Radius, float trace_length, int max_Points, float max_out_of_domain, String direction, ArrayList<String>  InitialSurface) {
        this.Divisions = Divisions;
        this.Tube_Radius = Tube_Radius;
        this.trace_length = trace_length;
        this.max_out_of_domain = max_out_of_domain;
        this.direction = direction;
        this.InitialSurface = InitialSurface;
    }

    public Vec getDivisions() {
        return Divisions;
    }

    public void setDivisions(Vec Divisions) {
        this.Divisions = Divisions;
    }

    public ArrayList<String> getInitialSurface() {
        return InitialSurface;
    }

    public void setInitialSurface(ArrayList<String> InitialSurface) {
        this.InitialSurface = InitialSurface;
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


    

    public Serializable_Construct_Streamline(Construct_Streamline Con) {
        super(Con);
        this.Divisions = Con.getDivisions();
        this.Tube_Radius = Con.getTube_Radius();
        this.trace_length = Con.getTrace_length();
        this.max_out_of_domain = Con.getMax_out_of_domain();
        this.direction=Con.getDirection();
        if (Con.getInitialSurface() != null && !Con.getInitialSurface().isEmpty()) {
            this.InitialSurface = new ArrayList<String>();
            for (Entry<Object, Integer> E : Con.getInitialSurface().entrySet()) {
                if (E.getKey() instanceof Boundary) {
                    InitialSurface.add((((Boundary) E.getKey()).toString()));
                }
            }
        } else {
            this.InitialSurface = new ArrayList<String>();
        }
    }
}
