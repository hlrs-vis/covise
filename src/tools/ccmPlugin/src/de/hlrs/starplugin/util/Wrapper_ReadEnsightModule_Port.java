package de.hlrs.starplugin.util;

import de.hlrs.starplugin.python.scriptcreation.modules.Module_ReadEnsight;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Wrapper_ReadEnsightModule_Port {

    private Module_ReadEnsight RE_Module;
    private int[] Port;

    public Wrapper_ReadEnsightModule_Port(Module_ReadEnsight RE_Module, int Port_3D, int Port_2D) {
        this.RE_Module = RE_Module;
        Port = new int[2];


        this.Port[0] = Port_3D;
        this.Port[1] = Port_2D;
    }

    public int[] getPort() {
        return Port;
    }

    public Module_ReadEnsight getRE_Module() {
        return RE_Module;
    }
}
