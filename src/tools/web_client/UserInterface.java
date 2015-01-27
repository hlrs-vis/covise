import java.util.Hashtable;
import java.net.*;

import java.awt.*;
import java.awt.event.ActionListener;
import java.awt.event.ItemListener;
import java.awt.event.ActionEvent;
import java.awt.event.ItemEvent;

import java.lang.StringBuffer;

public class UserInterface {

   ClientApplet client_applet;

 //protected Hashtable renderer_table;


   Panel upperPanel;
   Panel rightPanel, leftPanel;    

   Panel messagePanel;

   Button connectButton, clearButton;
   ActionListener bp; 

   Checkbox dview_chk;
   ItemListener chk;

   TextArea messageArea;
   List usersList;
   int nb_items;


   public static boolean dyn_active;
   
   private static String crt_renderer;


   public UserInterface() 
   {
      ClientApplet.user_interface = this;
   }

   public void init() 
   {  
      System.err.println("userInterface");
      this.client_applet = ClientApplet.client_applet;

      //renderer_table = new Hashtable();

 
      nb_items = 0;
      dyn_active = false;
      crt_renderer = null; 
 
      GridBagLayout gridbag = new GridBagLayout();
      GridBagConstraints c = new GridBagConstraints();

      client_applet.setBackground(Color.lightGray);
      client_applet.setLayout(gridbag);
      
      bp = new ButtonPushed();
      if(bp==null) System.err.println("ActionListener Problems");
      
      chk = new CheckboxSel();
      if(chk==null) System.err.println("ItemListener Problems");

      // creating upperPanel


      // upperPanel.leftPanel

      Label lb = new Label("Registered Renderers");
      c.weightx = 0.5;
      c.weighty = 0.1;
      c.gridx = 0;
      c.gridy = 0;
      c.insets = new Insets(2,1,0,1);  //left+right padding
      gridbag.setConstraints(lb, c);
      client_applet.add(lb);

      usersList = new List(10,false);
      c.weightx = 1.0;
      c.weighty = 0.4;
      c.gridx = 0;
      c.gridy = 1;
      c.gridheight = 5;
      c.gridwidth = 3;
      c.insets = new Insets(0,0,0,0);  //left+right padding
      gridbag.setConstraints(usersList, c);
      client_applet.add(usersList);
 
      // upperPanel.rightPanel


      connectButton = new Button("Connect");
      connectButton.setActionCommand("Connect");
      connectButton.addActionListener(bp);
      c.weightx = 0.1;
      c.weighty = 0.3;
      c.gridx = 3;
      c.gridy = 0;
      c.gridheight = 1;
      c.gridwidth = 1;
      c.insets = new Insets(0,1,0,1);  //left+right padding
      gridbag.setConstraints(connectButton, c);      
      client_applet.add(connectButton);
      
      dview_chk = new Checkbox("Dynamic view update", false);
      dview_chk.addItemListener(chk);
      c.weightx = 0.1;
      c.weighty = 0.3;
      c.gridx = 3;
      c.gridy = 1;
      c.insets = new Insets(0,1,0,1);  //left+right padding
      gridbag.setConstraints(dview_chk, c);      
      client_applet.add(dview_chk);

      clearButton = new Button("Clear msg area");
      clearButton.setActionCommand("Clear");
      clearButton.addActionListener(bp);
      c.weightx = 0.1;
      c.weighty = 0.3;
      c.gridx = 3;
      c.gridy = 2;
      c.insets = new Insets(0,0,0,5);  //left+right padding
      gridbag.setConstraints(clearButton, c);      
      client_applet.add(clearButton);
 

      // creating messagePanel
      //messageArea = new TextArea("",4,60,TextArea.SCROLLBARS_VERTICAL_ONLY);
      messageArea = new TextArea("Messages Area \n",4, 60);
      c.weightx = 1.0;
      c.weighty = 0.5;
      c.gridx = 0;
      c.gridy = 6;
      c.gridwidth = 4; 
      c.insets = new Insets(0,1,0,1);  // padding
      gridbag.setConstraints(messageArea, c);     
      client_applet.add(messageArea);

      client_applet.validate();
    }

    public void printInfo(String info) 
    {
       messageArea.setText(info);

    }

    public void addInfo(String info) 
    {
       messageArea.append(info);

    }

    public void addInfo(int info) 
    {
        
       messageArea.append(String.valueOf(info));

    }

    public void addInfo(float info) 
    {
        
       messageArea.append(String.valueOf(info));

    }
    

 //    public void paint(Graphics g) 
 //    {
 //
 //       Color bg = client_applet.getBackground();           
 //       g.setColor(bg);
 //	Dimension d = client_applet.getSize();
 //       g.draw3DRect(0, 0, d.width-2, d.height-2, false);
 //       g.setColor(new Color(0, 0, 0));
 //       g.drawLine(0, 1, d.width - 4, 1);
 //       g.drawLine(1, 0, 1, d.height-3);
 //   }

    public void addObject(String object_name) 
    {
       int pos;
       String obj;
       String port = null;
       
       pos = object_name.indexOf(')');
       if(pos>0)               // removing "/" end ".cgi-rnd"   
       {
          obj = object_name.substring(1,pos+1);
          //pos = object_name.indexOf('_',pos);
          //if(pos>0) port = object_name.substring(pos+1);
          //else System.err.println("ERROR: UserInterface.addObject incorrect renderer id "+object_name);     
       }  
       else obj = object_name;
 
       if(nb_items == 1) 
       {
          String first_item = usersList.getItem(0);
          if(first_item.startsWith("No")) 
          {
             usersList.removeAll();
             //printInfo("");
             nb_items = 0;
          }
       }
       if(obj.startsWith("VRML")) 
       {
          addInfo("\n New renderer has been registered : " + obj);
          //renderer_table.put(obj,port);
       }       
       usersList.add(obj);
       nb_items++;

    }

    public void deleteObject(String object_name) 
    {
       int pos;
       String obj,port;
       
       pos = object_name.indexOf(')');
       if(pos>0)          // removing "/" end ".cgi-rnd"   
       {
          obj = object_name.substring(1,pos+1);
          //pos = object_name.indexOf('_',pos);
          //if(pos>0) port = object_name.substring(pos+1);
          //else System.err.println("ERROR: UserInterface.deleteObject incorrect renderer id " +object_name );
          //renderer_table.remove(obj); 
       }
       else obj = object_name;

       if(obj.equals(crt_renderer))
       {
          addInfo("\nConnection lost -> " + crt_renderer);
          crt_renderer = null; 
       }  
       usersList.remove(obj);
       nb_items--;
       if(nb_items == 0) 
       {
          usersList.add("No VRML_Renderer has been registered !");
          nb_items = 1;
       } 
    }

    public void remove_users()
    {
       usersList.removeAll(); 
       //renderer_table.clear();
       nb_items = 0;
    }

    public void select(String object_name) 
    {

       String[] l = usersList.getItems();
       int n = 0, i = 0;
       for(i=0; i<l.length; i++) 
       {
          if(l[i]==object_name) { n = i; break; }
       }
       if(i<l.length) usersList.select(i);
       else 
       { 
          usersList.deselect(usersList.getSelectedIndex());
       }

    }

 /*
    public void start_new_conn(String name)
    {
      int begin_pos,pos;
      String port = null;

      port = (String)renderer_table.get(name);
      if(port!=null) 
      {
         String hostname;
         begin_pos = name.indexOf('(');
         pos = name.indexOf(')');
         hostname = name.substring(begin_pos+1,pos); 
         String urlString = "http://"+hostname+":"+port+"/single_obj.wrl";
         URL url = null;
         try {
            url = new URL(urlString);
         } catch (MalformedURLException e) {
            System.err.println("ERROR: UserInterface.start_new_conn " + e);
            return;
         }
         client_applet.appletContext.showDocument(url,name);
      } 

    }
 */



public class ButtonPushed implements ActionListener 
{

   public void actionPerformed(ActionEvent e) 
   {

      System.out.println("Action Performed " + e.getActionCommand());
      String command = e.getActionCommand();
      if(command=="Connect") 
      {
       
         String message = new String("Button Connect Pressed");
         //client_applet.printInfo(message);
         String user = usersList.getSelectedItem();
         if(user != null)
         {
            if(user.startsWith("No"))
            {
               printInfo("No users has been registered !!!");  
            } 
            else
            {
               client_applet.init_scene(); 
               if(dyn_active) client_applet.rmv_dynamic_view();
               crt_renderer = user;
               printInfo("Connection with " + crt_renderer);
               client_applet.send_get_req("/"+user+".cgi-rnd");
               //client_applet.load_URL(user);
               if(dyn_active) client_applet.set_dynamic_view();
               //start_new_conn(user);
            }              
         }
         else
         {
            printInfo("Please select an user !!!\n");
         }          
         //client_applet.get_last_wrl();  

      }
      else
      { 
         if(command=="Clear") 
         {
            String message = new String("Button Clear pressed");
            printInfo("");
            //client_applet.printInfo(message);
            //client_applet.update_xxx();           
         }


      }     
   }
    
}


public class CheckboxSel implements ItemListener 
{
   public void itemStateChanged(ItemEvent e)  
   {
      if(e.getStateChange() == ItemEvent.SELECTED) 
      {
         System.err.println("Dynamic view update selected");
         dyn_active = true;
         client_applet.set_dynamic_view();
      } 
      else 
      {
         System.err.println("Dynamic view update deselected");
         dyn_active = false;
         client_applet.rmv_dynamic_view();
      }
   }
}


}
