/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <assert.h>
#include <stdlib.h>
#include "global.h"
#include "modui.h"

// public section

using namespace covise;
using namespace covise::controller;

CTRLGlobal *CTRLGlobal::m_global = nullptr;

CTRLGlobal *CTRLGlobal::getInstance()
{
    if (m_global == NULL)
    {
        m_global = new CTRLGlobal();
    }
    return m_global;
}



// protected section

// private section

CTRLGlobal::CTRLGlobal()
: controller(new Controller((char *)"Controller"))
, objectList(new object_list)
, modUIList(new modui_list)
, s_nodeID(0)
{
}

CTRLGlobal::CTRLGlobal(const CTRLGlobal &)
{
    assert(0);
}

