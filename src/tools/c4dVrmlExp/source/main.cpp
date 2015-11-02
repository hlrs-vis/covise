#include "c4d.h"


// forward declarations

//OBJECT
Bool RegisterLOD();				//Registierung des LOD Plugin
Bool RegisterNavigationInfo();	//NavigationInfo Objekt
//TAG
//SCENESAVER
Bool RegisterVRML();			//Registierung des VRML-Exporters



Bool PluginStart(void)
{	
	//OBJECT
	if(!RegisterLOD()) return FALSE;
	if(!RegisterNavigationInfo()) return FALSE;
	//TAG
	//SCENESAVER
	if(!RegisterVRML()) return FALSE;	

    return TRUE;
}                          


void PluginEnd(void)
{
}


Bool PluginMessage(Int32 id, void *data)
{
    switch (id)
    {
        case C4DPL_INIT_SYS:
            if (!resource.Init()) return FALSE; // don't start plugin without resource
            return TRUE;
        case C4DMSG_PRIORITY: 
            return TRUE;
    }
    return FALSE;
}



