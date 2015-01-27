import java.awt.*;

import java.lang.StringBuffer;
import java.net.*;
import java.io.*;

public class WWWConnection {

   public static ClientApplet client_applet;
   public static UserInterface user_interface;
   public static PlugIn plug_in;
   public static SceneGraph scene_graph;

   public static  BufferedReader in;
   public static  PrintWriter out; 
   public static  Socket client_socket;
   public static  WWWReadThread read_thread; 
   
   public static  String req;

   public WWWConnection() 
   {
      ClientApplet.www_conn = this;
   }

   public void disconnect()
   {
      System.err.println("\nDisconnect !!!\n");
      read_thread = null;  
      try {
         in.close();
         out.close();
      } catch (IOException e) {               // close failed
         System.err.println("closing connections has failed");
      } 

   } 

   public void init() 
   {
      this.client_applet = ClientApplet.client_applet;
      this.user_interface = ClientApplet.user_interface;
      this.plug_in = ClientApplet.plug_in;
      this.scene_graph = ClientApplet.scene_graph;

      try {
          client_socket = new Socket(client_applet.hostsrv,client_applet.http_port);
          out = new PrintWriter(client_socket.getOutputStream(), true);
          in = new BufferedReader(new InputStreamReader(client_socket.getInputStream()));
      } catch (UnknownHostException e) {
         System.err.println("Don't know about host: sgi001.");
         client_applet.www_conn = null; 
         //System.exit(1);
      } catch (IOException e) {               // openConnection() failed
         System.err.println("Couldn't get I/O for the connection to: sgi001.");
         client_applet.www_conn = null; 
         //System.exit(1);
      }
      read_thread = new WWWReadThread();
      read_thread.init(this,in);
      read_thread.start();
   }

   

   public void send_msg(String msg)
   {
      out.print(msg);
      out.flush(); 

   }

   public void handle_wwwmsg(String msg)
   {
      int key,lng,pos,begin_pos;
      Integer i;
      String content,user;
      //String msg_scene;

      String parent_name;
      String object_name;
      String str_timestep;
      int is_timestep,min_timestep,max_timestep,timestep;
      
      //user_interface.addInfo("handle_wwwmsg\n");
        
      i = Integer.valueOf((msg.substring(0,3).trim()));
     
      key = i.intValue();
      
      content = msg.substring(3,msg.length()-1);
      //user_interface.addInfo(content);

      switch (key) {
         case 1 :    // Registered users
                 user_interface.remove_users();
                 lng = content.length();
                 //user_interface.addInfo("\n lng=");
                 //user_interface.addInfo(lng);
                 begin_pos = 0; 
                 while(begin_pos < lng)  //!= last char
                 { 
                    
                    pos = content.indexOf('\n',begin_pos);
                    if(pos>0)
                    {
                     //user_interface.addInfo("\n pos=");
                     //user_interface.addInfo(pos);
                       user = content.substring(begin_pos,pos);
                       //user_interface.addInfo("\n user=" + user);
                       user_interface.addObject(user);  
                       begin_pos = pos+1;  //skip `\n`
                    }
                    else begin_pos = lng; 
                 }     
                 break;
         case 2 :    // reponse from get_last_wrl
                 plug_in.replaceScene(content);
                 break;
         case 3 :    // add new usr
                 user_interface.addObject(content.substring(0,content.length()-1));
                 break;
         case 4 :    // remove usr
                 user_interface.deleteObject(content.substring(0,content.length()-1));
                 break;
         case 5 :    // VRML Camera msg
          //user_interface.printInfo(content);
          //System.err.println("VRML Camera :" + content);
                 plug_in.set_viewpoint(content);
                 break;
         case 6 :    // addGeometry
                 begin_pos = 0;
                 pos = content.indexOf('@',begin_pos);
                 if(pos>0)
                 {
                    object_name = content.substring(begin_pos,pos);
                    begin_pos = pos+1;
                    pos = content.indexOf('#',begin_pos);                                          if(pos>0)
                    { 
                       parent_name = content.substring(begin_pos,pos);
                       begin_pos = pos+1;
                       pos = content.indexOf('\n',begin_pos);
                       i = Integer.valueOf(content.substring(begin_pos,pos));
                       timestep = i.intValue();
                       scene_graph.addGeometryToScene(parent_name,object_name,content.substring(pos+1),timestep);   
                    }
                 }
                 break;
         case 7 :    // addGroup
                 begin_pos = 0;
                 pos = content.indexOf('@',begin_pos);
                 if(pos>0)
                 {
                    object_name = content.substring(begin_pos,pos);
                    begin_pos = pos+1;
                    pos = content.indexOf('#',begin_pos);                                          if(pos>0)
                    { 
                       parent_name = content.substring(begin_pos,pos);
                       begin_pos = pos+1;
                       pos = content.indexOf('\n',begin_pos);
                       i = Integer.valueOf(content.substring(begin_pos,pos));
                       timestep = i.intValue();
                       scene_graph.addGroupToScene(parent_name,object_name,0,0,0,timestep);   
                    }
                 }
                 break; 
         case 8 :    // addSwitch
                 begin_pos = 0;
                 pos = content.indexOf('@',begin_pos);
                 if(pos>0)
                 {
                    object_name = content.substring(begin_pos,pos);
                    begin_pos = pos+1;
                    pos = content.indexOf('#',begin_pos);                                          if(pos>0)
                    { 
                       parent_name = content.substring(begin_pos,pos);
                       begin_pos = pos+1;
                       pos = content.indexOf('#',begin_pos);
                       i = Integer.valueOf(content.substring(begin_pos,pos));
                       min_timestep = i.intValue();
                       begin_pos = pos+1;
                       pos = content.indexOf('\n',begin_pos);
                       i = Integer.valueOf(content.substring(begin_pos,pos));
                       max_timestep = i.intValue();
                       scene_graph.addGroupToScene(parent_name,object_name,1,min_timestep,max_timestep,-1);   
                    }
                 }
                 break;
         case 9 :    // rmv Object
                 scene_graph.rmvGeometryFromScene(content);
                 
                 break;
         case 10:    // set timestep
                 i = Integer.valueOf(content);
                 timestep = i.intValue();
                 System.err.println(" timestep message :" + timestep);
                 plug_in.set_timestep(timestep);
                 break;  
         case 11:    // activate telepointer
          //System.err.println("--- Telepointer message :" + content);
                 plug_in.set_telepointer(content);
                 break; 
         case 12:    // deactivate telepointer
          //System.err.println("--- Telepointer message :" + content);
                 plug_in.rst_telepointer();
                 break;
      }
      //user_interface.addInfo("end handle_wwwmsg \n");

      msg = null;
      content = null;
      //Runtime.getRuntime().gc(); // force immediate gc

   } 



}



