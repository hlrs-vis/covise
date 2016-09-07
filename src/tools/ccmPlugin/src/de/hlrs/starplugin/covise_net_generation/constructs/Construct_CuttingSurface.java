/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package de.hlrs.starplugin.covise_net_generation.constructs;

import de.hlrs.starplugin.configuration.Configuration_Tool;
import de.hlrs.starplugin.load_save.Serializable_Construct_CuttingSurface;
import de.hlrs.starplugin.util.FieldFunctionplusType;
import de.hlrs.starplugin.util.Vec;
import java.util.HashMap;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Construct_CuttingSurface extends Construct {

    protected Vec vertex;
    protected Vec point;
    protected float Distance;
    protected String Direction;

    public Construct_CuttingSurface() {
        this.Distance = 0f;
        this.Direction = Configuration_Tool.RadioButtonActionCommand_X_Direction;
        this.vertex = new Vec(1, 0, 0);
        this.point = new Vec(0, 0, 0);
    }

    public Construct_CuttingSurface(Vec ver, Vec poi, float dis){

        this.Direction = Configuration_Tool.RadioButtonActionCommand_notKart_Direction;
        this.vertex =ver;
        this.point = poi;
        this.Distance=dis;

    }

    public Vec getVertex() {
        return vertex;
    }

    public void setVertex(Vec vertex) {
        this.vertex = vertex;
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

    public String getDirection() {
        return Direction;
    }

    public void setDirection(String Direction) {
        this.Direction = Direction;
    }

    public void modify(Construct_CuttingSurface ConMod) {
        this.Direction = ConMod.getDirection();
        this.Distance = ConMod.getDistance();
        this.FFplType = ConMod.getFFplType();
        this.Parts = new HashMap<Object, Integer>(ConMod.getParts());
        this.point = ConMod.getPoint();
        this.vertex = ConMod.getVertex();
    }

    public Construct_CuttingSurface(Serializable_Construct_CuttingSurface SCon, FieldFunctionplusType FFpT, HashMap<Object, Integer> sParts) {
        super(SCon, FFpT, sParts);
        this.Direction = SCon.getDirection();
        this.Distance = SCon.getDistance();
        this.point = SCon.getPoint();
        this.vertex = SCon.getVertex();
    }

}
