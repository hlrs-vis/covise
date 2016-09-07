package de.hlrs.starplugin.load_save;

import de.hlrs.starplugin.covise_net_generation.constructs.Construct_GeometryVisualization;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Serializable_Construct_GeometryVisualization extends Serializable_Construct {

    private static final long serialVersionUID =6169208943263789879L;
    private boolean showdata;
    private String Color;
    private float Transparency;

    public Serializable_Construct_GeometryVisualization() {
    }

    public Serializable_Construct_GeometryVisualization(boolean showdata, String Color, float Transparency) {
        this.showdata = showdata;
        this.Color = Color;
        this.Transparency = Transparency;
    }
    

    public String getColor() {
        return Color;
    }

    public void setColor(String Color) {
        this.Color = Color;
    }

    public float getTransparency() {
        return Transparency;
    }

    public void setTransparency(float Transparency) {
        this.Transparency = Transparency;
    }

    public boolean isShowdata() {
        return showdata;
    }

    public void setShowdata(boolean showdata) {
        this.showdata = showdata;
    }

    public Serializable_Construct_GeometryVisualization(Construct_GeometryVisualization Con) {
        super(Con);
        this.showdata=Con.isShowdata();
        this.Color=Con.getColor();
        this.Transparency=Con.getTransparency();
    }


    
}
