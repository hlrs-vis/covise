package de.hlrs.starplugin.covise_net_generation.constructs;

import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.load_save.Serializable_Construct_CuttingSurfaceSeries;
import de.hlrs.starplugin.util.FieldFunctionplusType;
import de.hlrs.starplugin.util.Vec;
import java.util.HashMap;

/**
 *
 *  @author Weiss HLRS Stuttgart
 */
public class Construct_CuttingSurfaceSeries extends Construct_CuttingSurface {

    private float DistanceBetween;
    private int amount;

    public Construct_CuttingSurfaceSeries() {
        this.Distance = 0f;
        this.Direction = Configuration_Tool.RadioButtonActionCommand_X_Direction;
        this.vertex = new Vec(1, 0, 0);
        this.point = new Vec(0, 0, 0);
        this.amount = 2;
        this.DistanceBetween = 1;
    }

    public float getDistanceBetween() {
        return DistanceBetween;
    }

    public void setDistanceBetween(float DistanceBetween) {
        this.DistanceBetween = DistanceBetween;
    }

    public int getAmount() {
        return amount;
    }

    public void setAmount(int amount) {

        this.amount = amount;
    }

    public void modify(Construct_CuttingSurfaceSeries ConMod) {
        this.Direction = ConMod.getDirection();
        this.Distance = ConMod.getDistance();
        this.FFplType = ConMod.getFFplType();
        this.Parts = new HashMap<Object, Integer>(ConMod.getParts());
        this.point = ConMod.getPoint();
        this.vertex = ConMod.getVertex();
        this.DistanceBetween = ConMod.getDistanceBetween();
        this.amount = ConMod.getAmount();
    }

    public Construct_CuttingSurfaceSeries(Serializable_Construct_CuttingSurfaceSeries SCon, FieldFunctionplusType FFpT, HashMap<Object, Integer> sParts) {
        super(SCon, FFpT, sParts);
        DistanceBetween = SCon.getDistanceBetween();
        amount = SCon.getAmount();
    }


}
