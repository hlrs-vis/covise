/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CTRL_GLOBAL_H_
#define CTRL_GLOBAL_H_

#include "control_process.h"
#include "control_netmod.h"
#include "control_modlist.h"
#include "control_object.h"

namespace covise
{

/**
 * CTRLGlobal singleton class:
 * Stores references to main data objects used by controller
 */
class CTRLGlobal
{
public:
    // reference to the controller object
    Controller *controller;

    // reference to the list of modules
    modulelist *moduleList;

    // reference to the list of data managers
    DM_list *dataManagerList;

    // reference to the list of user interfaces
    ui_list *userinterfaceList;

    // reference to the list of hosts
    rhost_list *hostList;

    // reference to the list of network modules
    net_module_list *netList;

    // reference to the list of data objects
    object_list *objectList;

    // reference to the list of ui objects
    modui_list *modUIList;

    /** Get the object reference for
       *  accessing the internal data
       *
       *  @return   object reference
       */
    static CTRLGlobal &get_handle();
    int s_nodeID;

    /// default constructor
    CTRLGlobal();

    /// Destructor
    ~CTRLGlobal();

protected:
private:
    // singleton object used to access
    // references to main controller objects
    static CTRLGlobal *m_global;

    /// copy constructor : assert(0)
    CTRLGlobal(const CTRLGlobal &);
};
}
#endif // _CTRL_GLOBAL_H_
