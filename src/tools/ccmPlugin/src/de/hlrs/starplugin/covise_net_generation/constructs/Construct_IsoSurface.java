package de.hlrs.starplugin.covise_net_generation.constructs;

import de.hlrs.starplugin.load_save.Serializable_Construct_IsoSurface;
import de.hlrs.starplugin.util.FieldFunctionplusType;
import java.util.HashMap;

/**
 *
 *  @author Weiss HLRS Stuttgart
 */
public class Construct_IsoSurface extends Construct {

    private float IsoValue = 0f;

    public Construct_IsoSurface() {
    }

    public float getIsoValue() {
        return IsoValue;
    }

    public void setIsoValue(float IsoValue) {
        this.IsoValue = IsoValue;
    }

    public void modify(Construct_IsoSurface ConMod) {
        this.IsoValue = ConMod.getIsoValue();
        this.Parts = new HashMap<Object, Integer>(ConMod.getParts());
        this.FFplType = ConMod.getFFplType();
    }

    public Construct_IsoSurface(Serializable_Construct_IsoSurface SCon, FieldFunctionplusType FFpT, HashMap<Object, Integer> sParts) {
        super(SCon, FFpT, sParts);
        this.IsoValue = SCon.getIsoValue();
    }
}
