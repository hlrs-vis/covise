package de.hlrs.starplugin.load_save;

import de.hlrs.starplugin.covise_net_generation.constructs.Construct_CuttingSurfaceSeries;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Serializable_Construct_CuttingSurfaceSeries extends Serializable_Construct_CuttingSurface {

    private static final long serialVersionUID = -6990566493975617449L;
    private float DistanceBetween;
    private int amount;

    public Serializable_Construct_CuttingSurfaceSeries() {
    }

    public Serializable_Construct_CuttingSurfaceSeries(float DistanceBetween, int amount) {
        this.DistanceBetween = DistanceBetween;
        this.amount = amount;
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

    public Serializable_Construct_CuttingSurfaceSeries(Construct_CuttingSurfaceSeries Con) {
        super(Con);
        this.DistanceBetween=Con.getDistanceBetween();
        this.amount=Con.getAmount();
    }

}
