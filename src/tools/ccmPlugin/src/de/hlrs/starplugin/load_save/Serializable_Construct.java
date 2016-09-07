package de.hlrs.starplugin.load_save;

import de.hlrs.starplugin.covise_net_generation.constructs.Construct;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Map.Entry;
import star.common.Boundary;
import star.common.Region;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Serializable_Construct implements Serializable {

    private static final long serialVersionUID = -838943527720495884L;
    protected String Name;
    protected String Typ;
    protected String FF;
    protected ArrayList<String> Parts;

    public Serializable_Construct() {
    }

    public Serializable_Construct(String Name, String Typ, String FFplType, ArrayList<String> Parts) {
        this.Name = Name;
        this.Typ = Typ;
        this.FF = FFplType;
        this.Parts = Parts;
    }

    public String getFF() {
        return FF;
    }

    public void setFF(String FF) {
        this.FF = FF;
    }

    public String getName() {
        return Name;
    }

    public void setName(String Name) {
        this.Name = Name;
    }

    public ArrayList<String> getParts() {
        return Parts;
    }

    public void setParts(ArrayList<String> Parts) {
        this.Parts = Parts;
    }

    public String getTyp() {
        return Typ;
    }

    public void setTyp(String Typ) {
        this.Typ = Typ;
    }

    public Serializable_Construct(Construct Con) {


        this.Name = Con.getName();
        this.Typ = Con.getTyp();
        if (Con.getFFplType() != null) {
            this.FF = Con.getFFplType().getFF().getPresentationName();
        } else {
            this.FF = null;
        }
        if (Con.getParts() != null) {
            this.Parts = new ArrayList<String>();
            for (Entry<Object, Integer> E : Con.getParts().entrySet()) {
                if (E.getKey() instanceof Region) {
                    Parts.add((((Region) E.getKey()).toString()));
                }
                if (E.getKey() instanceof Boundary) {
                    Parts.add((((Boundary) E.getKey()).toString()));
                }
            }
        } else {
            this.Parts = new ArrayList<String>();
        }


    }
}
