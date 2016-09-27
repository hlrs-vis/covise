package de.hlrs.starplugin.python.scriptcreation;

import de.hlrs.starplugin.python.scriptcreation.modules.Connection;
import de.hlrs.starplugin.python.scriptcreation.modules.Module;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collection;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class PythonScript_Writer {

    private ArrayList<Module> ModuleList = new ArrayList<Module>();
    private ArrayList<Connection> ConnectionList = new ArrayList<Connection>();

    public PythonScript_Writer() {
    }

    public void addModule(Module m) {
        ModuleList.add(m);
    }

    public void addallModule(Collection<Module> ModColl) {
        ModuleList.addAll(ModColl);
    }

    public void addConnecton(Connection Con) {
        ConnectionList.add(Con);
    }

    public void addAllConnection(Collection<Connection> ConCol) {
        ConnectionList.addAll(ConCol);
    }

    public void createPythonFile(File Path) throws IOException {
        
            FileOutputStream fos = new FileOutputStream(Path.getPath());
            OutputStreamWriter osw = new OutputStreamWriter(fos);

            osw.write(StringArraytoString(fileStart()));
            osw.flush();
            //Module schreiben
            for (Module m : ModuleList) {
                osw.write(StringArraytoString(m.addtoscript()));
            }
            osw.flush();

            //Connections Schreiben
            if (!ConnectionList.isEmpty()) {
                osw.write(StringArraytoString(ConnectionHeadline()));
                for (Connection con : ConnectionList) {
                    osw.write(con.addtoscript() + System.lineSeparator());
                }
            }

            //FileAbschluss schreiben
            osw.write(StringArraytoString(fileEnd()));
            osw.close();

    }

    public String[] fileStart() {
        String[] CreateNetwork = new String[4];
        CreateNetwork[0] = "#";
        CreateNetwork[1] = "# create global net";
        CreateNetwork[2] = "#";
        CreateNetwork[3] = "network=net()";
        return CreateNetwork;
    }

    public String[] fileEnd() {
        String[] SaveasNetFile = new String[9];
        SaveasNetFile[0] = "#";
        SaveasNetFile[1] = "# save as Covise Net File";
        SaveasNetFile[2] = "#";
        SaveasNetFile[3] = "#network.save(\"CoviseNetFile.net\")";
        SaveasNetFile[4] = "#";
        SaveasNetFile[5] = "# uncomment the following line if you want your script to be executed after loading";
        SaveasNetFile[6] = "#";
        SaveasNetFile[7] = "runMap()";
        SaveasNetFile[8] = "#";
        return SaveasNetFile;
    }

    public String StringArraytoString(String[] Array) {
        String result = "";

        for (int i = 0; i < Array.length; i++) {
            result = result.concat(Array[i] + System.lineSeparator());
        }
        return result;
    }

    public String[] ConnectionHeadline() {
        String[] ConnectionHeadline = new String[3];
        ConnectionHeadline[0] = "#";
        ConnectionHeadline[1] = "# CONNECTIONS";
        ConnectionHeadline[2] = "#";

        return ConnectionHeadline;
    }
}


