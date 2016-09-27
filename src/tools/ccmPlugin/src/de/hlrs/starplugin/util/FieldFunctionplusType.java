package de.hlrs.starplugin.util;

import star.common.FieldFunction;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class FieldFunctionplusType {

    private FieldFunction FF;
    private String Type;

    public FieldFunctionplusType(FieldFunction FF, String Type) {
        this.FF = FF;
        this.Type = Type;
    }

    public void setFF(FieldFunction FF) {
        this.FF = FF;
    }

    public void setType(String Type) {
        this.Type = Type;
    }

    public FieldFunction getFF() {
        return FF;
    }

    public String getType() {
        return Type;
    }

    public String print() {
        if (FF != null) {
            return FF.getFunctionName();
        } else {
            return "null";
        }


    }
}
