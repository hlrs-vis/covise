package de.hlrs.starplugin.python.scriptcreation;

import Main.PluginContainer;
import java.io.IOException;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class PythonScript_Creator {

    private PluginContainer PC;
    private PythonScript_Writer PS_Writer;
    private PythonScript_ModuleManager PS_ModuleManager;

    public PythonScript_Creator(PluginContainer PC) {
        this.PC = PC;
        PS_ModuleManager = new PythonScript_ModuleManager(PC);
        PS_Writer = new PythonScript_Writer();

    }

    public void createScript() throws IOException {
        PS_ModuleManager.createNet();
        PS_Writer.addallModule(PS_ModuleManager.getAllModules());
        PS_Writer.addAllConnection(PS_ModuleManager.getConnectionList());
        PS_Writer.createPythonFile(PC.getCNGDMan().getExportPath());
        PC.getSim().println("Python Script saved: " + PC.getCNGDMan().getExportPath());
    }
}
