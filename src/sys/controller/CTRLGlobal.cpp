/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <assert.h>
#include <stdlib.h>
#include "CTRLGlobal.h"

// public section

using namespace covise;

CTRLGlobal &CTRLGlobal::get_handle()
{
    return *m_global;
}

CTRLGlobal::~CTRLGlobal()
{
    if (controller)
        delete controller;
    if (moduleList)
        delete moduleList;
    if (dataManagerList)
        delete dataManagerList;
    if (userinterfaceList)
        delete userinterfaceList;
    if (hostList)
        delete hostList;
    if (netList)
        delete netList;
    if (objectList)
        delete objectList;
    if (modUIList)
        delete modUIList;
}

// protected section

// private section

CTRLGlobal::CTRLGlobal()
{
    controller = new Controller((char *)"Controller");
    moduleList = new modulelist;
    dataManagerList = new DM_list;
    userinterfaceList = new ui_list;
    hostList = new rhost_list;
    netList = new net_module_list();
    objectList = new object_list;
    modUIList = new modui_list;
    if (m_global == NULL)
    {
        m_global = this;
    }
    else
    {
        print_comment(__LINE__, __FILE__, "already have a CTRLGlobal object");
    }
    s_nodeID = 0;
}

CTRLGlobal::CTRLGlobal(const CTRLGlobal &)
{
    assert(0);
}

CTRLGlobal *CTRLGlobal::m_global = NULL;
