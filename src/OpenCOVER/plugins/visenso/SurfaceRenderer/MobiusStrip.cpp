/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * MobiusStrip.cpp
 *
 *  Created on: Apr 28, 2011
 *      Author: Jessica Wolz
 */

#include "SurfaceRenderer.h"
#include "MobiusStrip.h"
#include <cover/VRSceneGraph.h>
#include "ParamSurface.h"

using namespace osg;
using namespace std;

MobiusStrip::MobiusStrip()
    : GenericGuiObject("MobiusStrip")
{

    gui_visible = addGuiParamBool("SichtBar", false);
    gui_mode = addGuiParamInt("Oberfläche", 0);
}

MobiusStrip::~MobiusStrip()
{
}

void MobiusStrip::guiParamChanged(GuiParam *guiParam)
{

    if (guiParam == gui_visible)
    {
        p_visible = gui_visible->getValue();
        //setVisible();
    }
}

/*void MobiusStrip::setVisible(){
    std::cout << "p_visible:" << p_visible << std::endl;
    if (p_visible == true){
	m_rpRenderer ->setVisible(1);
    }
}*/
