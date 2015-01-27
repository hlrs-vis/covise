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
#include <OpenVRUI/osg/mathUtils.h>

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#endif

#include <util/common.h>

#include <sysdep/opengl.h>
#include <config/CoviseConfig.h>
#include "coVRTouchTable.h"
#include "VRViewer.h"
#include "coVRConfig.h"
#include "coVRPluginSupport.h"
#include "coVRPluginList.h"
#include "coVRTui.h"
#include <osg/Geode>
#include <osg/StateSet>

using namespace opencover;
using namespace covise;
coVRTouchTable *coVRTouchTable::tt = NULL;
coVRTouchTable::coVRTouchTable()
{
    tt = this;
    ttInterface = new coVRTouchTableInterface();
}

coVRTouchTable *coVRTouchTable::instance()
{
    if (tt == NULL)
        tt = new coVRTouchTable();
    return tt;
}

void coVRTouchTable::config()
{
}

coVRTouchTable::~coVRTouchTable()
{
}

void coVRTouchTable::update()
{
}
