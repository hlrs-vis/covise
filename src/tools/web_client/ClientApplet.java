import java.applet.*;
import java.awt.*;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import java.net.*;
import java.io.*;

//import vrml.external.*;
//import vrml.external.field.*;
//import vrml.external.exception.*;


public class ClientApplet extends Applet {
  public static  boolean is_master = false;
  public static  boolean is_slave = true;

  public static  ClientApplet	client_applet; 
  public static  UserInterface	user_interface;
  public static  WWWConnection  www_conn;
  public static  PlugIn         plug_in; 
  public static  SceneGraph     scene_graph;
  public static  AppletContext  appletContext;
  public static  int id;
  public static  int dyn_usr;
  public static  int dyn_view;

  public static String hostsrv;
  public static int http_port;

  private TextArea status, objects;

  public void init() {
     is_master = false;
     dyn_usr = 0;
     dyn_view = 0;
     id = (int)System.currentTimeMillis();

     ClientApplet.client_applet = this;

     appletContext = getAppletContext();  
  
     hostsrv = getParameter("HOSTSRV");
     
     String http_portString = getParameter("HTTP_PORT");
     if (http_portString != null) 
     {
        try {
           Integer port = Integer.valueOf(http_portString);
           http_port = port.intValue();
        } catch (NumberFormatException e) {
           http_port = 50000;
        }
     }


     URL applet_url = this.getCodeBase();

     String host = applet_url.getHost();
     if(host != null) hostsrv = host;

     int port = applet_url.getPort();
     if(port > 0) http_port = port; 

     user_interface = new UserInterface();
     user_interface.init();

     
    
  }

  public void start()
  {
     plug_in = new PlugIn();
     plug_in.init(); 

     scene_graph = new SceneGraph(); 
     scene_graph.init();

     www_conn = new WWWConnection();
     www_conn.init();

     update_users();
     set_dynamic_usr();

     //if(hostsrv != null) user_interface.printInfo(hostsrv);
     //user_interface.addInfo(http_port);
    
  }
  

  public void stop()
  {
     System.err.println("\nStop !!!\n");
     if(www_conn != null) www_conn.disconnect();
     www_conn = null; 
  }



  public void destroy()
  { 
     System.err.println("\nDestroy !!!\n");
   //if(www_conn != null) www_conn.disconnect();
   //www_conn = null;
  }
  
  public void printInfo(String info) {
   //System.err.println(info + "\n");
     user_interface.printInfo(info);
  }
 
  public void update_users()
  {
     send_get_req("/applet.cgi-reg");

   //user_interface.printInfo(" Start_of_List\n");

  }

  
  public void update_xxx()
  {
     plug_in.set_timestep(0);
  }


  public void init_scene()
  {
     scene_graph.ClearScene();
     plug_in.init_scene();
     
  }



  public void new_scene(String content)
  {
     int pos,begin_pos;
     String vrml_camera;
 
     vrml_camera = null;

     begin_pos = content.indexOf("DEF World");
     if(begin_pos>0)  // ViewPoint message present
     {
        pos = content.indexOf('}');
        vrml_camera = content.substring(begin_pos,pos+1);
        //content = content.substring(pos+1);
     } 
     replace_scene(content);
     if(vrml_camera != null) plug_in.set_viewpoint(vrml_camera);       
     plug_in.bind_view();   

  }

  public void load_URL(String id)
  {
     plug_in.load_URL(id);

  }

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
 
     //user_interface.addInfo(view);
     begin_pos = view.indexOf("position");     
     if(begin_pos>0) 
     {
        position_el = new String[4];
        pos_val = new float[3];
        position = view.substring(begin_pos);
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
        orientation_el[3] = orientation.substring(begin_pos,pos);
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
     plug_in.set_viewpoint(pos_val,or_val,fv_val);
  }
  public void set_dynamic_usr()
  {
     dyn_usr = 1;
     send_get_req("/applet.cgi-set_dyn_usr");
  }

  public void rmv_dynamic_usr()
  {
     dyn_usr = 0;
     send_get_req("/applet.cgi-rst_dyn_usr");
  }

  
  public void set_dynamic_view()
  {
     dyn_view = 1;
     send_get_req("/applet.cgi-set_dyn_view");
  }

  public void rmv_dynamic_view()
  {
     dyn_view = 0;
     send_get_req("/applet.cgi-rst_dyn_view");
  }

  public void send_get_req(String req)
  {
     www_conn.send_msg("GET " + req + " HTTP/1.0\r\n\r\n");
  }
  public void get_last_wrl()
  {
     www_conn.send_msg("GET /get_last_wrl HTTP/1.0\r\n\r\n");
     //user_interface.addInfo("\n Getting the last wrl !!!");

  }

  public void replace_scene(String scene_desc)
  {
     String clean_desc;  
     int crt = 0; 
     int pos;

     if(scene_desc == null) 
     {
       System.err.println("\nreplace_scene() scene_desc = null\n");
        return; 
     }

     int lng = scene_desc.length();

     //user_interface.addInfo("\n scene_desc length=");
     //user_interface.addInfo(lng);
     
     //user_interface.addInfo("\n---" + scene_desc.substring(0,5) + "---\n");
     //user_interface.addInfo("\n===" + scene_desc.substring(lng-6) + "===\n");
       
     plug_in.replaceScene(scene_desc);
     //user_interface.addInfo("!\n End of replace\n");

     scene_desc = null;
     //scene = null;

     //Runtime.getRuntime().gc(); // force immediate gc

  } 

}  // end of ClientApplet












