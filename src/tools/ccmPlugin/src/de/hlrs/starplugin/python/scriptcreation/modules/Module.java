package de.hlrs.starplugin.python.scriptcreation.modules;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public abstract class Module {

        protected String Name;
    protected String Type;
    protected int param_pos_x;
    protected int param_pos_y;

    public Module(String Type, String Name, int param_pos_x, int param_pos_y) {
        this.Name = Name;
        this.Type = Type;
        this.param_pos_x = param_pos_x;
        this.param_pos_y = param_pos_y;
    }

    public abstract String[] addtoscript();

    public String getName() {
        return Name;
    }



}
