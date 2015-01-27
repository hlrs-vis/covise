/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * MobiusStrip.h
 *
 *  Created on: May 12, 2011
 *      Author: ac_te
 */

#ifndef MOBIUSSTRIP_H_
#define MOBIUSSTRIP_H_

#include <string>
#include <stdexcept>
#include <iostream>

#include <PluginUtil/GenericGuiObject.h>
#include <osg/MatrixTransform>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRPlugin.h>
#include <config/CoviseConfig.h>

class SurfaceRenderer;
using namespace covise;
using namespace opencover;

class MobiusStrip : GenericGuiObject
{

public:
    MobiusStrip();
    ~MobiusStrip();

    void setVisible();

protected:
    void guiParamChanged(GuiParam *guiParam);

private:
    GuiParamBool *gui_visible;
    GuiParamInt *gui_mode;

    int p_mode;
    int p_visible;

    SurfaceRenderer *m_rpRenderer;
};
#endif /* MOBIUSSTRIP_H_ */
