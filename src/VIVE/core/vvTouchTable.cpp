/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *									*
 *          								*
 *                            (C) 2002					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			TouchTable.cpp 				*
 *									*
 *	Description		TouchTable optical tracking system interface class				*
 *									*
 *	Author			Uwe Woessner				*
 *									*
 *	Date			July 2002				*
 *									*
 *	Status			in dev					*
 *
 */
#include <OpenVRUI/vsg/mathUtils.h>

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#endif

#include <util/common.h>

#include <sysdep/opengl.h>
#include <config/CoviseConfig.h>
#include "vvTouchTable.h"
#include "vvViewer.h"
#include "vvConfig.h"
#include "vvPluginSupport.h"
#include "vvPluginList.h"
#include "vvTui.h"

using namespace vive;
using namespace covise;
vvTouchTable *vvTouchTable::tt = NULL;
vvTouchTable::vvTouchTable()
{
    tt = this;
    ttInterface = new vvTouchTableInterface();
}

vvTouchTable *vvTouchTable::instance()
{
    if (tt == NULL)
        tt = new vvTouchTable();
    return tt;
}

void vvTouchTable::config()
{
}

vvTouchTable::~vvTouchTable()
{
}

void vvTouchTable::update()
{
}
