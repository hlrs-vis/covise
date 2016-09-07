package de.hlrs.starplugin.load_save;

import de.hlrs.starplugin.configuration.Configuration_GUI_Strings;
import java.util.ArrayList;

/**
 *
 * @author Weiss HLRS Stuttgart
 */
public class Message_Load {

    private boolean trouble;
    private ArrayList<String> BList;
    private ArrayList<String> RList;
    private ArrayList<String> FFList;
    private ArrayList<String> ConList;

    public Message_Load() {
        trouble = false;
        BList = new ArrayList<String>();
        RList = new ArrayList<String>();
        FFList = new ArrayList<String>();
        ConList = new ArrayList<String>();

    }

    public boolean isTrouble() {
        return trouble;
    }

    public void setTrouble(boolean trouble) {
        this.trouble = trouble;
    }

    public void addBoundary(String B) {
        BList.add(B);
    }

    public void addRegion(String R) {
        RList.add(R);
    }

    public void addFF(String FF) {
        FFList.add(FF);
    }
    public void addCon(String Con){
        ConList.add(Con);
    }
    public String getResultMessage(){
        String S="";
        S=S.concat(Configuration_GUI_Strings.eol+ "Boundaries: "+BList.size());
        S=S.concat(Configuration_GUI_Strings.eol+ "Regions: "+RList.size());
        S=S.concat(Configuration_GUI_Strings.eol+ "FieldFunctions: "+FFList.size());
        S=S.concat(Configuration_GUI_Strings.eol+Configuration_GUI_Strings.eol+ "Constructs: "+ConList.size());
        return  S;
    }
}
