package de.hlrs.starplugin.util;

import java.io.Serializable;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Vec implements Serializable {

    private static final long serialVersionUID = -592932512502979473L;
    public float x;
    public float y;
    public float z;

    public Vec() {
        this.x = 0;
        this.y = 0;
        this.z = 0;
    }

    public Vec(float x, float y, float z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public Vec(Double x, Double y, Double z) {
        this.x = x.floatValue();
        this.y = y.floatValue();
        this.z = z.floatValue();
    }

    public float getX() {
        return x;
    }

    public void setX(float x) {
        this.x = x;
    }

    public float getY() {
        return y;
    }

    public void setY(float y) {
        this.y = y;
    }

    public float getZ() {
        return z;
    }

    public void setZ(float z) {
        this.z = z;
    }
}
