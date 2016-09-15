package de.hlrs.starplugin.load_save;

import de.hlrs.starplugin.covise_net_generation.constructs.Construct_IsoSurface;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Serializable_Construct_IsoSurface extends Serializable_Construct {

    private static final long serialVersionUID =-7247360987155255770L;
    private float IsoValue;

    public Serializable_Construct_IsoSurface() {
    }

    public Serializable_Construct_IsoSurface(float IsoValue) {
        this.IsoValue = IsoValue;
    }
    

    public float getIsoValue() {
        return IsoValue;
    }

    public void setIsoValue(float IsoValue) {
        this.IsoValue = IsoValue;
    }

    public Serializable_Construct_IsoSurface(Construct_IsoSurface Con) {
        super(Con);
        this.IsoValue=Con.getIsoValue();
    }
    
}
