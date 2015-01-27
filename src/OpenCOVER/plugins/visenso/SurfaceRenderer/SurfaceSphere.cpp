/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
* SurfaceSphere.cpp
*
*  Created on: Apr 28, 2011
*      Author: ac_te
*/

#include "SurfaceSphere.h"
#include "ParamSurface.h"
#include "SurfaceRenderer.h"
#include <PluginUtil/GenericGuiObject.h>
#include <cover/VRSceneGraph.h>

#include <string>
#include <iostream>

using namespace std;
using namespace osg;

SurfaceSphere::SurfaceSphere()
    : GenericGuiObject("Sphere")
{
    p_active = addGuiParamBool("Visible", false);
    p_showSurfaceSphere = addGuiParamBool("Visible", false);

    menuItemSurfaceSphere = new coCheckboxMenuItem("Sphere", true, NULL);
    menuItemSurfaceSphere->setMenuListener(this);

    // SurfaceRenderer::plugin->getObjectsMenu()->insert(menuItemSurface, 0);
    updateMenuItem();
}

SurfaceSphere::~SurfaceSphere()
{
}

void SurfaceSphere::preFrame()
{
}

void SurfaceSphere::menuEvent(coMenuItem *menuItem)
{

    if (menuItem == menuItemSurfaceSphere)
    {
        if (p_showSurfaceSphere->getValue())
        {
            p_showSurfaceSphere->setValue(false);
        }
        else
        {
            p_showSurfaceSphere->setValue(true);
        }
        updateMenuItem();
    }
}

void SurfaceSphere::updateMenuItem()
{

    menuItemSurfaceSphere->setState(p_showSurfaceSphere->getValue());

    if (p_showSurfaceSphere->getValue())
    {
        menuItemSurfaceSphere->setName("Zeige Bild als Oberflaeche");
    }
    else
    {
        menuItemSurfaceSphere->setName("Zeige Netz als Oberflaeche");
    }
}

void SurfaceSphere::guiParamChanged(GuiParam *guiParam)
{

    if (guiParam == p_showSurfaceSphere)
    {

        updateMenuItem();
    }
}
