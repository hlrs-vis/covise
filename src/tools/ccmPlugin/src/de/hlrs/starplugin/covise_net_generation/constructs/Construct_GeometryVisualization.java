package de.hlrs.starplugin.covise_net_generation.constructs;

import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.load_save.Serializable_Construct_GeometryVisualization;
import de.hlrs.starplugin.util.FieldFunctionplusType;
import java.util.HashMap;

/**
 *
 *  @author Weiss HLRS Stuttgart
 */
public class Construct_GeometryVisualization extends Construct {

    private boolean showdata = false;
    private String Color = Configuration_Tool.Color_grey;
    private float Transparency = 0;

    public Construct_GeometryVisualization() {
    }

    public boolean isShowdata() {
        return showdata;
    }

    public void setShowdata(boolean showdata) {
        this.showdata = showdata;
    }

    public void modify(Construct_GeometryVisualization ConMod) {
        this.FFplType = ConMod.getFFplType();
        this.Parts = new HashMap<Object, Integer>(ConMod.getParts());
        this.showdata = ConMod.isShowdata();
        this.Transparency = ConMod.getTransparency();
        this.Color = ConMod.getColor();
    }

    public String getColor() {
        return Color;
    }

    public float getTransparency() {
        return Transparency;
    }

    public void setColor(String Color) {
        this.Color = Color;
    }

    public void setTransparency(float Transparency) {
        this.Transparency = Transparency;
    }

    @Override
    public void setFFplType(FieldFunctionplusType FFplType) {

        this.FFplType = FFplType;
        if (FFplType != null) {
            this.showdata = true;
        } else {
            this.showdata = false;
        }
    }

    public Construct_GeometryVisualization(Serializable_Construct_GeometryVisualization SCon, FieldFunctionplusType FFpT, HashMap<Object, Integer> sParts) {
        super(SCon, FFpT, sParts);
        showdata = SCon.isShowdata();
        Color = SCon.getColor();
        Transparency = SCon.getTransparency();
    }

}
