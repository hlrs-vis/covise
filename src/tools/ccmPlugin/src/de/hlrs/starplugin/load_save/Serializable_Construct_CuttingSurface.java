package de.hlrs.starplugin.load_save;

import de.hlrs.starplugin.covise_net_generation.constructs.Construct_CuttingSurface;
import de.hlrs.starplugin.util.Vec;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Serializable_Construct_CuttingSurface extends Serializable_Construct {

    private static final long serialVersionUID = -3066158746993700041L;
    protected Vec vertex;
    protected Vec point;
    protected float Distance;
    protected String Direction;

    public Serializable_Construct_CuttingSurface() {
    }

    public Serializable_Construct_CuttingSurface(Vec vertex, Vec point, float Distance, String Direction) {
        this.vertex = vertex;
        this.point = point;
        this.Distance = Distance;
        this.Direction = Direction;
    }

    public String getDirection() {
        return Direction;
    }

    public void setDirection(String Direction) {
        this.Direction = Direction;
    }

    public float getDistance() {
        return Distance;
    }

    public void setDistance(float Distance) {
        this.Distance = Distance;
    }

    public Vec getPoint() {
        return point;
    }

    public void setPoint(Vec point) {
        this.point = point;
    }

    public Vec getVertex() {
        return vertex;
    }

    public void setVertex(Vec vertex) {
        this.vertex = vertex;
    }

    public Serializable_Construct_CuttingSurface(Construct_CuttingSurface Con) {
        super(Con);
        this.vertex = Con.getVertex();
        this.point = Con.getPoint();
        this.Distance = Con.getDistance();
        this.Direction = Con.getDirection();
    }
}
