import java.applet.*;
import java.awt.*;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import java.net.*;
import java.io.*;

import vrml.external.*;
import vrml.external.field.*;
import vrml.external.exception.*;


public class PlugIn extends Object implements EventOutObserver {

  private static ClientApplet client_applet;
  private static UserInterface	user_interface;
  private static Browser browser;

  private int current_timesteps = -1;
  private int no_switch;

  private Node root;
  private Node[] scene;
  private Node[] scene_view; 
  private Node view_covise;
  private Node view_default;
  private Node ssc, stepper, telepointer;
  private Node[] telep;  
  private Node telep_text,rot_v,trans_v;

  private EventInMFNode root_addChildren;
  private EventInMFNode root_rmvChildren;

  private EventInMFNode telep_addChildren;
  private EventInMFNode telep_rmvChildren;
  private EventInSFVec3f telep_setTrans;
  private EventInSFVec3f telep_setTrans_v;
  private EventInSFRotation telep_setRot_v;
  private EventInMFString telep_setText;

  private EventInSFVec3f set_position;
  private EventInSFRotation set_orientation;
  private EventInSFFloat set_fieldOfView;
  private EventInSFBool set_bind;
  

  // stepper 
  private EventOutSFInt32 current_changed;
  private EventInSFInt32  set_numTimesteps;
  private EventInSFInt32  stepper_on;
  private EventInSFInt32  set_current; 


  private final static float defaultFieldOfView = 0.785398f;
  // container for Nodes

  private EventInMFNode addChildren;
  private EventInMFNode rmvChildren;

  private Node[] parent;
  private Node[] child; 
  private EventInSFInt32 set_whichChoice;


  float or_x,or_y,or_z,or_delta;
  float tr_x,tr_y,tr_z;


  public PlugIn()
  {
     ClientApplet.plug_in = this;
  }


 //***********************************************************************


  public void init() 
  {

     this.client_applet = ClientApplet.client_applet;
     this.user_interface = ClientApplet.user_interface;

     browser = null;

     root = null;
     telepointer = null;
     telep = null;
     rot_v = null;
     trans_v = null;
     telep_text = null;
     scene = null;
     scene_view = null;
     view_covise = null;
     view_default = null;

     root_addChildren = null;
     root_rmvChildren = null;

     telep_addChildren = null;
     telep_rmvChildren = null;
     telep_setTrans = null;  
     telep_setText = null; 
     telep_setTrans_v = null;
     telep_setRot_v = null;   
 
     set_position = null;
     set_orientation = null;
     set_fieldOfView = null;
     set_bind = null;

     no_switch = 0;

     or_x = or_y = or_z = or_delta = 0;
     tr_x = tr_y = tr_z = 0;

 
     for (int count = 0; count < 10; count++) 
     {
        try {
           try { Thread.sleep(150); } catch (InterruptedException ignored) { }
           browser = Browser.getBrowser(client_applet);
        }
        catch (Exception e) { }  // getBrowser() can throw NullBrowserException
        if (browser != null) break;
        Browser.print("browser was null, trying again...");
     }
     if (browser == null) {
        throw new Error("Failed to get the browser after 10 tries!");
     }
     Browser.print("Got browser!");

     // get different Nodes

     try {
        telepointer = browser.getNode("TELE_POINTER");
        telep_text = browser.getNode("TELE_TEXT");
        rot_v = browser.getNode("ROT_V");
        trans_v = browser.getNode("TRANS_V");
        ssc = browser.getNode("SSC");
        stepper = browser.getNode("STEPPER");
      //tp_position1 = browser.getNode("TRANS");
      //tp_position2 = browser.getNode("ROT");
      //scenesize = browser.getNode("SCENESIZE"); 
        root = browser.getNode("ROOT");
        //view_covise = browser.getNode("DEFAULT");
        //view_all = browser.getNode("VIEWALL");
        //handles = browser.getNode("HANDLES");
        //boundingbox = browser.getNode("BOUNDINGBOX");
        //scale = browser.getNode("SCALE");
        //center = browser.getNode("CENTER");
        //position = browser.getNode("POSITION");
        //move = browser.getNode("MOVE");
        //handlesposition = browser.getNode("HANDLESPOSITION");	  
     }  catch(InvalidNodeException e) {
        System.err.println("\n PlugIn.init(): " + e);
     }

     // get fields
     try {

        // root

        telep_rmvChildren = (EventInMFNode)telepointer.getEventIn("removeChildren");
        telep_addChildren = (EventInMFNode)telepointer.getEventIn("addChildren");
        telep_setTrans = (EventInSFVec3f)telepointer.getEventIn("set_translation");
        telep_setText = (EventInMFString)telep_text.getEventIn("set_string");
        telep_setTrans_v = (EventInSFVec3f)trans_v.getEventIn("set_translation");        
        telep_setRot_v = (EventInSFRotation)rot_v.getEventIn("set_rotation");
  

        root_rmvChildren = (EventInMFNode)root.getEventIn("removeChildren");
        root_addChildren = (EventInMFNode)root.getEventIn("addChildren");
          
        // stepper 
    
        current_changed = (EventOutSFInt32)ssc.getEventOut("current_changed");
        set_numTimesteps = (EventInSFInt32)ssc.getEventIn("set_numTimesteps");
        stepper_on = (EventInSFInt32)stepper.getEventIn("set_whichChoice");
        set_current = (EventInSFInt32)ssc.getEventIn("set_current");

     } catch (InvalidEventInException e) {
        System.err.println("\n PlugIn.init(): " + e);
     }

  }



 //***********************************************************************



  public void callback(EventOut event, double ts, Object type)
  {
   //if ((String)type=="viewer_orientation") {
   //   tmp = ((EventOutSFRotation)event).getValue();
   //   actual_position.putOrientation(tmp);
   // } else if ((String)type=="viewer_position"){
   //   tmp = ((EventOutSFVec3f)event).getValue();
   //   actual_position.putTranslation(tmp);
   // } else if ((String)type=="tp_orientation"){
   //   tmp = ((EventOutSFRotation)event).getValue();
   //   actual_tp_position.putOrientation(tmp);
   // } else if((String)type=="tp_position"){
   //   tmp = ((EventOutSFVec3f)event).getValue();
   //   actual_tp_position.putTranslation(tmp);
   // }
  }


 // ** get ROOT from plug-in **
 //***********************************************************************

  public Node getRootNode() { return root; };


 //***********************************************************************

  public void init_scene()
  {
   //System.err.println("--- init_scene()---"); 
     try {
        if(scene_view != null) root_rmvChildren.setValue(scene_view);
        if(scene != null) root_rmvChildren.setValue(scene);
     } catch (IllegalArgumentException e) {
        System.err.println("\n PlugIn.init_scene(): " + e);
     }  
     scene_view = null;
     view_covise = null;
     scene = null;
     current_timesteps = -1;
     no_switch = 0;
     stepper_on.setValue(-1); 
  }

 //***********************************************************************

  public void decr_no_sw()
  {
     no_switch--;
     if(no_switch==0) stepper_on.setValue(-1);  
  }

 //***********************************************************************


  public void load_URL(String id)
  {
     String[] req = new String[2];

     req[0]  = "single_obj.wrl"; //+ id + ".cgi-rnd";
     req[1]  = "";
     
     //System.err.println("$$$$$ - PlugIn.load_URL - " + req[0]); 

     //browser.loadURL(req, null);
  }



 //***********************************************************************


  public void init_viewpoint(String vrml_string)
  { 
   //System.err.println("--- init_viewpoint()---"); 
     try {
        scene_view = browser.createVrmlFromString(vrml_string);
     } catch (InvalidVrmlException e) {
        System.err.println("\n PlugIn.init_viewpoint() : " + e);
     } 

     if(scene_view != null) 
     {
        view_covise = scene_view[0]; 
        try {
           set_position = (EventInSFVec3f)view_covise.getEventIn("set_position"); 
           set_orientation = (EventInSFRotation)view_covise.getEventIn("set_orientation");
           set_fieldOfView = (EventInSFFloat)view_covise.getEventIn("set_fieldOfView");
           set_bind = (EventInSFBool)view_covise.getEventIn("set_bind");
        } catch (InvalidEventInException e) {
           System.err.println("\n PlugIn.init_viewpoint(): " + e);
        }
        try {
           root_addChildren.setValue(scene_view);
        } catch (IllegalArgumentException e) {
           System.err.println("\n PlugIn.init_viewpoint(): " + e);
        }         
     }

     if(view_covise != null)
     {
      //System.err.println("\n get Viewpoint node - success \n"); 
     }


  }


 //***********************************************************************

  public void set_viewpoint(String view)
  {
     String position;
     String[] position_el;
     String orientation;
     String[] orientation_el;
     String fieldOfView;
     int begin_pos,pos,i;
     Float tmp_f;
     float[] pos_val = null;
     float[] or_val = null;
     float fv_val = -1;

     if(view_covise == null) 
     {
        init_viewpoint(view);
        bind_view();
        return;
     }

     //System.err.println("--- set_viewpoint node without init"); 


     //user_interface.addInfo(view);
     begin_pos = view.indexOf("position");
     if(begin_pos>0) 
     {
        position_el = new String[4];
        pos_val = new float[3];
        position = view.substring(begin_pos);

        //pos = position.indexOf('\n');
        //position_str = position.substring(0,pos+1);
        //System.err.println("---position_str = " +  position_str);

        begin_pos = 0;
        //user_interface.addInfo("\n position_el: ");
        pos = position.indexOf('\t',begin_pos);
        //position_el[0] = position.substring(begin_pos,pos);
        //user_interface.addInfo(position_el[0]);
        //user_interface.addInfo("|");
        begin_pos = pos+1; 
        for(i=0;i<3;i++)
        {
           pos = position.indexOf(" ",begin_pos);
           position_el[i] = position.substring(begin_pos,pos);
           tmp_f = Float.valueOf(position_el[i]);
           pos_val[i] = tmp_f.floatValue();
           //user_interface.addInfo(pos_val[i]);
           //user_interface.addInfo("|");
           begin_pos = pos+1; 
        }
     }
     begin_pos = view.indexOf("orientation");     
     if(begin_pos>0) 
     {
        orientation_el = new String[5];
        or_val = new float[4];
        orientation = view.substring(begin_pos);

        //pos = orientation.indexOf('\n');
        //orientation_str = orientation.substring(0,pos+1);
        //System.err.println("---orientation_str = " +  orientation_str);

        begin_pos = 0;
        //user_interface.addInfo("\n orientation_el: ");
        pos = orientation.indexOf('\t',begin_pos);
        //orientation_el[0] = orientation.substring(begin_pos,pos);
        //user_interface.addInfo(orientation_el[0]);
        //user_interface.addInfo("|");
        begin_pos = pos+1; 
        for(i=0;i<3;i++)
        {
           pos = orientation.indexOf(" ",begin_pos);
           orientation_el[i] = orientation.substring(begin_pos,pos);
           tmp_f = Float.valueOf(orientation_el[i]);
           or_val[i] = tmp_f.floatValue();
           //user_interface.addInfo(or_val[i]);
           //user_interface.addInfo("|");
           begin_pos = pos+1;
        }
        pos = orientation.indexOf(" ",begin_pos+1);
        orientation_el[3] = orientation.substring(begin_pos+1,pos);
        tmp_f = Float.valueOf(orientation_el[3]);
        or_val[3] = tmp_f.floatValue();
        //user_interface.addInfo(or_val[3]);
        //user_interface.addInfo("|");
        begin_pos = pos+1;
     }
     begin_pos = view.indexOf("fieldOf");
     if(begin_pos>0) 
     {  
        fieldOfView = view.substring(begin_pos);
        begin_pos = fieldOfView.indexOf(" ");
        pos = fieldOfView.indexOf(" ",begin_pos+1);
        tmp_f = Float.valueOf(fieldOfView.substring(begin_pos,pos));
        fv_val = tmp_f.floatValue();
        //user_interface.addInfo("\nfieldOfView = ");
        //user_interface.addInfo(fv_val);
        //user_interface.addInfo("|\n"); 
     }
     //user_interface.addInfo("\n======================\n"); 
     set_viewpoint(pos_val,or_val,fv_val);
     bind_view();
  }


 //***********************************************************************



  public void set_viewpoint(float[] pos, float[] or, float fv)
  {
     int i;

     if(view_covise == null)
     {
        System.err.println("\n Plug-in set_viewpoint - node null !! \n"); 
        return;
     } 
     if(pos != null)
     {
        tr_x = pos[0];
        tr_y = pos[1];
        tr_z = pos[2];

        set_position.setValue(pos); 
        //user_interface.addInfo("\n Position update !!!\n");
        //for(i=0;i<3;i++)
        //{
        //   user_interface.addInfo(pos[i]);
        //   user_interface.addInfo("|");
        //}       
     }  
     
     if(or != null)
     {
        or_x = or[0];
        or_y = or[1];
        or_z = or[2];
        or_delta = or[3];

        set_orientation.setValue(or);        
        //user_interface.addInfo("\n Orientation update !!!\n");
        //for(i=0;i<4;i++)
        //{
        //   user_interface.addInfo(or[i]);
        //   user_interface.addInfo("|");
        //} 
     }

     if(fv>0) set_fieldOfView.setValue(fv);

     //set_bind.setValue(true);

  }

 //***********************************************************************


  public void bind_view()
  {
     if(set_bind != null)
     {
        try {
           set_bind.setValue(true);
        } catch (IllegalArgumentException e) {
           System.err.println("\n PlugIn.bind_view(): " + e);
        } 
     }      
  }

 //***********************************************************************


  public void set_timestep(int timestep)
  {
   //System.err.println("---PlugIn.set_timestep("+timestep+")");
     if(stepper_on != null) set_current.setValue(timestep);
  } 

 //***********************************************************************

  public void set_telepointer(String content)
  {
     String telep_str;
     String trans;
     String[] obj = {"Test"};

     int begin_pos,pos;

     Float tmp_f;
     float[] pos_val = null;
     float[] trans_val = null;
     float[] rot_val = null;
     String tmp_str;
      
     //if(telep_addChildren != null)
     if(telep_setTrans != null)
     {
        pos_val = new float[3];
        trans_val = new float[3];
        rot_val = new float[4]; 

        pos = content.indexOf("translation");
        begin_pos = content.indexOf(' ',pos);
        pos = content.indexOf(' ',begin_pos+1);
        tmp_str = content.substring(begin_pos+1,pos);
        tmp_f = Float.valueOf(tmp_str);
        pos_val[0] = tmp_f.floatValue();
        begin_pos = pos+1;

        pos = content.indexOf(' ',begin_pos);
        tmp_str = content.substring(begin_pos,pos);
        tmp_f = Float.valueOf(tmp_str);
        pos_val[1] = tmp_f.floatValue();
        begin_pos = pos+1;

        pos = content.indexOf('\n',begin_pos);
        tmp_str = content.substring(begin_pos,pos);
        tmp_f = Float.valueOf(tmp_str);
        pos_val[2] = tmp_f.floatValue();

        begin_pos =  content.indexOf('"');
        begin_pos++;
        pos = content.indexOf('"',begin_pos);
        obj[0] = content.substring(begin_pos,pos);

        trans_val[0] = -tr_x;
        trans_val[1] = -tr_y;
        trans_val[2] = -tr_z;

        rot_val[0] = -or_x;
        rot_val[1] = -or_y;
        rot_val[2] = -or_z;
        rot_val[2] = or_delta;

      /* 
        int begin_pos = content.indexOf('\n');
        begin_pos++;
        int pos = content.indexOf('\n',begin_pos);
        pos++; 
        trans = content.substring(begin_pos,pos);
        obj = content.substring(pos); 


        telep_str  = " Transform {\n ";
        telep_str += trans;

        telep_str += " children Transform { \n rotation ";
        telep_str += String.valueOf(or_x) + " " + String.valueOf(or_y) + " " + String.valueOf(or_z) + "  " + String.valueOf(or_delta) + " \n"; 

        telep_str += " children Transform { \n translation ";
        telep_str += String.valueOf(tr_x) + " " + String.valueOf(tr_y) + " " + String.valueOf(tr_z) + " \n";

        telep_str += obj;
        telep_str += "}\n}\n";
      */
        System.err.println("PlugIn.set_telepointer("+obj[0]+"): " +pos_val[0]+ ","+pos_val[1]+","+pos_val[2] );
        /* 
        try {
           telep = browser.createVrmlFromString(telep_str);   
           if(telep != null)
           {
              telep_addChildren.setValue(telep);
           }
        */
        try {
           //telep_setTrans_v.setValue(trans_val);
           telep_setTrans.setValue(pos_val);
           //telep_setRot_v.setValue(rot_val);
           telep_setText.setValue(obj);
        } catch (InvalidEventInException e) {
           System.err.println(" PlugIn.set_telepointer():" + e);
        } catch (InvalidVrmlException e) {
           System.err.println(" PlugIn.set_telepointer():" + e);
        }
     }
     

  }

 //***********************************************************************

  public void rst_telepointer()
  {
   /*
     if(telep_rmvChildren != null)
     {
        if(telep != null)
        {
           telep_rmvChildren.setValue(telep);
        }

     }
  */
     String[] obj = {""};
     if(telep_setText != null)
     {
        telep_setText.setValue(obj);
     }
 
  }



 //***********************************************************************
 // ** add geometry to VRML plug-in **

  public Node addGeometryToScene(Node parent_node, String vrml_string,  Object data, int timestep)
  {
   //Node[] geo=null;
     //Node[] ts=null;
     EventInMFNode ac;

     //System.err.println("PlugIn.addGeometryToScene parent_node = " + parent_node.getType());
     SceneGraph.SceneGeometryNode sgn = (SceneGraph.SceneGeometryNode)data;
     try {
        child = browser.createVrmlFromString(vrml_string);  //geo
        //ts  = browser.createVrmlFromString("TouchSensor{}");
        //child  = browser.createVrmlFromString("Group{}");
     } catch (InvalidVrmlException e) {
        System.err.println(" PlugIn.addGeometryToScene()1: " + e);
     } 

     if(child==null) System.err.println(" PlugIn.addGeometryToScene(): child==null");

     try {
      //EventInMFNode ac = (EventInMFNode)child[0].getEventIn("addChildren");
      //ac.setValue(geo);
        //ac.setValue(ts);
        //EventInSFBool set_enabled = (EventInSFBool)ts[0].getEventIn("set_enabled");
        //EventOutSFBool isActive   = (EventOutSFBool)ts[0].getEventOut("isActive");
        //sgn.set_enabled = set_enabled;
        sgn.vrml_node = child[0];
        //sgn.isActive = isActive;
        //isActive.advise(scene_graph, data);

        if(timestep==-1) 
        {
         //System.err.println("PlugIn.addGeometryToScene(): addchildren(-1)");
	   ac = (EventInMFNode)parent_node.getEventIn("addChildren");
           ac.setValue(child);
        } 
        else 
        {
         //System.err.println("PlugIn.addGeometryToScene(): addchildren number " + timestep);
	   ac = (EventInMFNode)parent_node.getEventIn("set_choice");
	   ac.set1Value(timestep, child[0]);
        }
     } catch (InvalidEventInException e) {
        System.err.println(" PlugIn.addGeometryToScene()3: " + e);
     } catch (IllegalArgumentException e) {
        System.err.println(" PlugIn.addGeometryToScene()4: " + e);
     }
    return child[0];
  }


 //***********************************************************************
 // ** add geometry directly to VRML plug-in **

  public void addGeometryDirectly(Node parent_node, String vrml_string, int timestep)
  {
     EventInMFNode ac;

     //System.err.println("PlugIn.addGeometryDirectly parent_node = " + parent_node.getType());
     try {
        child = browser.createVrmlFromString(vrml_string);  //geo
     } catch (InvalidVrmlException e) {
        System.err.println(" PlugIn.addGeometryDirectly()1: " + e);
     } 

     if(child==null) System.err.println(" PlugIn.addGeometryDirectly(): child==null");

     try {
        if(timestep==-1) 
        {
         //System.err.println("PlugIn.addGeometryDirectly(): addchildren(-1)");
	   ac = (EventInMFNode)parent_node.getEventIn("addChildren");
           ac.setValue(child);
        } 
        else 
        {
         //System.err.println("PlugIn.addGeometryDirectly(): addchildren number " + timestep);
	   ac = (EventInMFNode)parent_node.getEventIn("set_choice");
	   ac.set1Value(timestep, child[0]);
        }
     } catch (InvalidEventInException e) {
        System.err.println(" PlugIn.addGeometryToScene()3: " + e);
     } catch (IllegalArgumentException e) {
        System.err.println(" PlugIn.addGeometryToScene()4: " + e);
     }

  }



 //***********************************************************************
 // ** remove geometry from VRML plug-in **

  public void rmvGeometryFromScene(Node parent_node, Node node) 
  {
     rmvChildren = (EventInMFNode)parent_node.getEventIn("removeChildren");
     child[0]=node;
     rmvChildren.setValue(child);
  }

 //***********************************************************************
 // ** add route to VRML plug-in **

  public void addRouteToScene(Node fromNode, String fromEventOut,
                              Node toNode, String toEventIn)
  {
   //System.err.println("AddRoute");
     browser.addRoute(fromNode, fromEventOut, toNode, toEventIn);    
     //System.err.println("End AddRoute");
  };


 //***********************************************************************
 // ** remove route from VRML plug-in **

  public void deleteRouteFromScene(Node fromNode, String fromEventOut,
                                   Node toNode, String toEventIn)
  {
     browser.deleteRoute(fromNode, fromEventOut, toNode, toEventIn);    
  };




 //***********************************************************************
 // ** add group to VRML plug-in **

  public Node addGroupNode(Node parent, int timestep)
  {
     if(parent==null) return root;
     try {
        child = browser.createVrmlFromString("Group { }");
        if(timestep == -1) 
        {
         //System.err.println("PlugIn::adGroupNode - in group ");
           addChildren = (EventInMFNode)parent.getEventIn("addChildren"); 
           addChildren.setValue(child);
        } 
        else 
        { 
         //System.err.println("PlugIn::adGroupNode - in switch ");
           addChildren = (EventInMFNode)parent.getEventIn("set_choice");
           addChildren.set1Value(timestep, child[0]);
        }
    } catch (InvalidEventInException e) {
       System.err.println(" PlugIn.addGroupNode(): " + e);
    } catch (InvalidVrmlException e) {
       System.err.println(" PlugIn.addGroupNode(): " + e);
    }
    return child[0];
  };


 //***********************************************************************
 // ** add switch to VRML plug-in **

  public Node addSwitchNode(Node parent, int timesteps)
  {
     no_switch++;  
     stepper_on.setValue(0);
     if(current_timesteps<timesteps) set_numTimesteps.setValue(timesteps+1);
     try {
        child = browser.createVrmlFromString("Switch { whichChoice 0}");
	addChildren = (EventInMFNode)parent.getEventIn("addChildren");
     } catch (InvalidEventInException e) {
        System.err.println(" PlugIn.addSwitchNode(): " + e);
     } catch (InvalidVrmlException e) {
        System.err.println(" PlugIn.addSwitchNode(): " + e);
     }
     addChildren.setValue(child);
     addRouteToScene(ssc, "current_changed", child[0], "set_whichChoice");

     return child[0];
  };


 //***********************************************************************
 // ** replace scene in VRML plug-in **

  public void replaceScene(String vrml_string)
  {
   
     try {
        if(scene != null)
        {
           root_rmvChildren.setValue(scene);
        }
        scene = browser.createVrmlFromString(vrml_string);
        if(scene != null) 
        {
           root_addChildren.setValue(scene);
           System.err.println("adding new scene !!!");
        }
        /*   
        String typ = scene[0].getType();
        //user_interface.addInfo("\n Type of the first node =" + typ + "\n");
        if(typ.startsWith("Viewpoint"))
        {
           view_default = scene[0];
           set_position = (EventInSFVec3f)view_default.getEventIn("set_position"); 
           set_orientation = (EventInSFRotation)view_default.getEventIn("set_orientation");
           set_fieldOfView = (EventInSFFloat)view_default.getEventIn("set_fieldOfView");
           set_bind = (EventInSFBool)view_default.getEventIn("set_bind");
        } 
        */
     } catch (InvalidEventInException e) {
        System.err.println("\n PlugIn.replaceScene() : " + e);
     } catch (InvalidVrmlException e) {
        System.err.println("\n PlugIn.replaceScene() : " + e);
     }       
   
     if(scene == null) System.err.println("\n scene = null \n");
           
     //scene = null;
     //Runtime.getRuntime().gc(); // force immediate gc
 
     

  }



}  // end of PlugIn class












