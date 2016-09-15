package de.hlrs.starplugin.python.scriptcreation.modules;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Connection {

    private Module mod1;
    private String mod1_portname;
    private Module mod2;
    private String mod2_portname;

    public Connection(Module mod1, String mod1_portname, Module mod2, String mod2_portname) {
        this.mod1 = mod1;
        this.mod1_portname = mod1_portname;
        this.mod2 = mod2;
        this.mod2_portname = mod2_portname;
    }

    public String addtoscript() {
        String ExportLine;
        ExportLine = "network.connect(" + this.mod1.getName() + ",\"" + this.mod1_portname + "\"," + this.mod2.getName() + ",\"" + this.mod2_portname + "\")";
        return ExportLine;
    }
}
