package de.hlrs.starplugin.covise_net_generation.constructs;

import de.hlrs.starplugin.configuration.Configuration_GUI_Strings;
import de.hlrs.starplugin.load_save.Serializable_Construct;
import de.hlrs.starplugin.util.FieldFunctionplusType;
import java.util.HashMap;

/**
 *Parent class for Visualization Constucts
 * @author Weiss HLRS Stuttgart
 */
public abstract class Construct {

    protected String Name;
    protected String Typ;
    protected FieldFunctionplusType FFplType;
    protected HashMap<Object, Integer> Parts = new HashMap<Object, Integer>();

    @Override
    public String toString() {
        return this.Name;
    }

    public Construct() {
        this.Name = Configuration_GUI_Strings.Default;
    }

    public void setName(String Name) {
        this.Name = Name;
    }

    public String getName() {
        return Name;
    }

    public String getTyp() {
        return Typ;
    }

    public void setTyp(String Typ) {
        this.Typ = Typ;
    }

    public FieldFunctionplusType getFFplType() {
        return FFplType;
    }

    public void setFFplType(FieldFunctionplusType FFplType) {
        this.FFplType = FFplType;
    }

    public HashMap<Object, Integer> getParts() {
        return Parts;
    }

    public void addPart(Object Key, int PartNumber) {
        Parts.put(Key, PartNumber);
    }

    public boolean checkValid(){
        if(this.Name==null){
            return false;
        }
        if(this.Typ==null){
            return false;
        }
        if(this.Parts.isEmpty()){
            return false;
        }
        return true;
    }

    public Construct(Serializable_Construct SCon, FieldFunctionplusType FFpT, HashMap<Object, Integer> sParts) {
        this.Name = SCon.getName();
        this.Typ = SCon.getTyp();
        this.FFplType = FFpT;
        this.Parts = sParts;
    }
}
