import java.util.Hashtable;
import java.util.Vector;
import java.util.Enumeration;
import java.util.StringTokenizer;

import vrml.external.Node;
import vrml.external.Browser;
import vrml.external.field.*;
import vrml.external.exception.*;

import java.awt.Checkbox;

public class SceneGraph {

  public final static int GROUP    = 0;
  public final static int SWITCH   = 1;
  public final static int GEOMETRY = 2;

  protected Hashtable node_table;
  
  protected ClientApplet client_applet;
  protected UserInterface user_interface;
  protected PlugIn plug_in;

  public static int crt_timestep;


  //***************************
  //** initialize scenegraph **
  //***************************

  public SceneGraph() 
  {
    ClientApplet.scene_graph = this;
  }

  public void init() {
    this.client_applet  = ClientApplet.client_applet;
    this.user_interface = ClientApplet.user_interface;
    this.plug_in = ClientApplet.plug_in;

    crt_timestep = -1;
    node_table = new Hashtable();

    node_table.put("ROOT", new SceneGroupNode(null, "ROOT", GROUP, -1, -1));
    
  };
  
  //********************************
  //** add geometry to scenegraph **
  //********************************

  public void addGeometryToScene(String parent_name, String object_name, String vrml_string, int timestep)
  {

   //System.err.println(" addGeometryToScene("+object_name+"@"+parent_name+")"); 

     SceneGroupNode scene_group_node = (SceneGroupNode)node_table.get(parent_name);
     if(scene_group_node!=null)
     {  
      //SceneGeometryNode new_node;
      //new_node = new SceneGeometryNode(scene_group_node, object_name, vrml_string, GEOMETRY, timestep);            
        if(parent_name.startsWith("ROOT"))
        {
           // add node to hashtable
         //SceneGeometryNode new_node;
         //new_node = new SceneGeometryNode(scene_group_node, object_name, vrml_string, GEOMETRY, timestep); 
           node_table.put(object_name,new SceneGeometryNode(scene_group_node, object_name, vrml_string, GEOMETRY, timestep) );
        }
        else 
        {
           int ts;
           if( scene_group_node.getObjectType() == SWITCH) ts = timestep;
           else ts = -1;
  
           plug_in.addGeometryDirectly(scene_group_node.getVRMLNode(), vrml_string, ts);
           //new_node = null;
        } 
     }
     //Runtime.getRuntime().gc(); // force immediate gc
     //System.err.println("\n\n\n"); 
     //printNode("ROOT");
  };

  //*****************************
  //** add group to scenegraph **
  //*****************************

  public void addGroupToScene(String parent_name, String group_name,
                int is_timestep, int min_timestep, int max_timestep, int timestep)  
  {

     int ts;
     user_interface.printInfo("Add Group " + group_name + " to " + parent_name + "\n");
     SceneGroupNode scene_group_node = (SceneGroupNode)node_table.get(parent_name);
     SceneGroupNode scene_node = (SceneGroupNode)node_table.get(group_name);

     if((scene_group_node!=null) && (scene_node==null))
     {
	int type; 
		
	if(is_timestep==1)  
        { 
           type = SWITCH; 
           ts = -1; 
        }
	else 
        {
           type = GROUP; 
           ts = timestep; 
        }

        // add node to hashtable
        node_table.put(group_name, new SceneGroupNode(scene_group_node, group_name, type, max_timestep, ts));
     }
     //System.err.println("\n\n\n"); 
     //printNode("ROOT");
  };

  //***********************************
  //** remove branch from scenegraph **
  //***********************************

  public void rmvGeometryFromScene(String object_name) 
  {
   //System.err.println("--- SceneGraph.rmvGeometryFromScene : "+object_name);
     SceneNode scene_node = (SceneNode)node_table.get(object_name);
     if(scene_node!=null) 
     {

 	int type = scene_node.getObjectType();   
        if((type==GROUP) || (type==SWITCH)) 
	{
           if(type==SWITCH) plug_in.decr_no_sw();
           SceneGroupNode scene_group_node = (SceneGroupNode)scene_node;
           scene_group_node.delete();
	}
        else 
	{
           SceneGeometryNode scene_geometry_node=(SceneGeometryNode)scene_node;
           scene_geometry_node.delete();
	}      
        node_table.remove(object_name);
     }
     //System.err.println("\n\n\n"); 
     //printNode("ROOT");
  };


  public void ClearScene()
  {
     System.err.println("---ClearScene---"); 
     SceneGroupNode r = (SceneGroupNode)node_table.get("ROOT");

     int s=r.child_nodes.size();
     int i = 0;
     while(i<s)
     {
        String child_name = (String)r.child_nodes.elementAt(0); 
        if(child_name != null) rmvGeometryFromScene(child_name); 
        i++;
     }
     /*
     Enumeration e = node_table.keys();
     while(e.hasMoreElements())
     {
        String key = (String)e.nextElement();
        if(!key.startsWith("ROOT"))
        {
           rmvGeometryFromScene(key);
        }
     }
     */
     //SceneGroupNode r = (SceneGroupNode)node_table.get("ROOT");
     //r.initChilds();  
     
     printNode("ROOT");

  }


  //************************
  //***  node definition ***
  //************************

  public class SceneNode
  {
    protected   String object_name;
    protected   int    object_type;
    protected   SceneGroupNode parent_node;
    public  Node vrml_node;
    
    public SceneNode(SceneGroupNode parent_node, String object_name, int object_type)
    {
       this.object_name = object_name;
       this.object_type = object_type;
       this.parent_node = parent_node;
       if(parent_node!=null) parent_node.addChildEntry(object_name);
       //user_interface.addObject(object_name);

    };
 
    public int getObjectType()
    {
       return object_type;
    };

    public Node getVRMLNode()
    {
       return vrml_node;
    };

    public String getObjectName()
    {
       return object_name;
    };
    
    public void delete()
    {
      if(parent_node!=null) parent_node.rmvChildEntry(object_name);
      //user_interface.deleteObject(object_name);
  
    }
  };

  //***************************
  //** Group node definition **
  //***************************

  public class SceneGroupNode extends SceneNode
  {

    private Vector child_nodes;

    public SceneGroupNode(SceneGroupNode parent_node, String object_name, int object_type, int timesteps, int timestep)
    {
      super(parent_node, object_name, object_type);	
      child_nodes = new Vector(5, 5);

      if(parent_node!=null) 
      {
         //+++++++++++++++++++++++++++++      
         // add switch or group to COSMO player
         if(object_type==SWITCH)
         {
            vrml_node = plug_in.addSwitchNode(parent_node.getVRMLNode(), timesteps);
         }      
         else if(object_type==GROUP)
         {
            if(parent_node.getObjectType() == GROUP)
            {
               vrml_node = plug_in.addGroupNode(parent_node.getVRMLNode(), -1);
            }
            else
            {
               vrml_node = plug_in.addGroupNode(parent_node.getVRMLNode(), timestep);
            }
         } 
      }
      else 
      {
         vrml_node = plug_in.getRootNode();
      }

    };
    
    public void addChildEntry(String child_name)
    {
      child_nodes.addElement(child_name);
    };

    public void rmvChildEntry(String child_name)
    {
      child_nodes.removeElement(child_name);
    };

    public void delete()
    {
       int s=child_nodes.size();
       int i = 0;
       if((object_type==GROUP)||(object_type==SWITCH)) 
       {
	  while(i<s)
	  {
             String child_name = (String)child_nodes.elementAt(0); 
             if(child_name != null) rmvGeometryFromScene(child_name); 
             i++;
	  }
       }
       if(parent_node.object_type!=SWITCH) plug_in.rmvGeometryFromScene(parent_node.getVRMLNode(), getVRMLNode());
	  
       super.delete();

    };

    public void initChilds()
    {
       child_nodes = new Vector(5, 5); 
    }

    public void print() 
    {
      int s=child_nodes.size(); 
      int i=0;

      while(i<s)
      {
         printNode((String)child_nodes.elementAt(i)); 
         i++;
      }
    };
  };
  //******************************
  //** Geometry node definition **
  //******************************

  public class SceneGeometryNode extends SceneNode
  {
   //public EventInSFBool set_enabled;
   //public EventOutSFBool isActive;
     public SceneGeometryNode(SceneGroupNode parent_node, String object_name,String vrml_string, int object_type, int timestep)
     {
        super(parent_node, object_name, object_type);

	//+++++++++++++++++++++++++++++
        // add geometry to COSMO player

        int ts;
        if( parent_node.getObjectType() == SWITCH) ts = timestep;
        else ts = -1;
  
        vrml_node = plug_in.addGeometryToScene(parent_node.getVRMLNode(), vrml_string,  (Object)this, ts);
     };        
       
     public void delete()
     {
        //++++++++++++++++++++++++++++++++++
        // remove geometry from COSMO player  
      //System.err.println("--- delete " + getObjectName());
      
        if(parent_node.getObjectType()==GROUP) 
        {
	   plug_in.rmvGeometryFromScene(parent_node.getVRMLNode(), getVRMLNode());
	}
        super.delete();
     };
  };

  private int space = 0;

  //*****************************
  //** print Scenegraph branch **
  //*****************************

  public void printNode(String object_name)
  {
    SceneNode sn = (SceneNode)node_table.get(object_name);

    for(int i=0; i<space; i++) System.err.print("  ");
    System.err.println(object_name);

    int type = sn.getObjectType();
    if((type==GROUP) || (type==SWITCH))
    {
      space++;
      SceneGroupNode sgrn = (SceneGroupNode)sn;
      sgrn.print();
      space--;
    }
  };

  //*****************************
  //** print Hashtable branch ***
  //*****************************

  public void printHashtable()
  {
    System.err.println("\n*** Print Hashtable ***\n");

    Enumeration e = node_table.elements();
    SceneNode sn;
    while(e.hasMoreElements()) 
    {
      sn = (SceneNode)e.nextElement();
      System.out.println(sn.getObjectName());
    }
  }

}




