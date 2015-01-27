/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * SurfaceSphere.h
 *
 *  Created on: Apr 28, 2011
 *      Author: Jessica Wolz
 */

#ifndef SURFACESPHERE_H_
#define SURFACESPHERE_H_

#include "SurfaceRenderer.h"
#include "ParamSurface.h"

#include <PluginUtil/GenericGuiObject.h>
#include <cover/OpenCOVER.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRPlugin.h>

#include <string>
#include <iostream>

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/osg/OSGVruiMatrix.h>

#include <osg/MatrixTransform>
#include <config/CoviseConfig.h>

using namespace std;
using namespace osg;

class SurfaceSphere : public GenericGuiObject, public coMenuListener
{

public:
    SurfaceSphere();
    virtual ~SurfaceSphere();
    void update();
    void preFrame();

protected:
    void guiParamChanged(GuiParam *guiParam);
    void menuEvent(coMenuItem *menuItem);

private:
    GuiParamBool *p_active;
    GuiParamBool *p_showSurfaceSphere;

    coCheckboxMenuItem *menuItemSurfaceSphere;

    void updateMenuItem();
};
#endif /* SURFACESPHERE_H_ */
