/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
// Description:   CaveVOX. Volume explorer for the Cave.
//
// Author:        Jurgen Schulze (jschulze@ucsd.edu)
//
// Creation Date: 2005-12-14
//
// **************************************************************************

#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coPopupHandle.h>
#include <OpenVRUI/coFlatPanelGeometry.h>
#include <OpenVRUI/coFlatButtonGeometry.h>
#include "coRectButtonGeometry.h"
#include <OpenVRUI/coValuePoti.h>
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coFrame.h>
#include <string.h>
//#include <vtk/vtkStructuredGridReader.h>
#include "CaveVOX.h"

CaveVOX *plugin = NULL;

int coVRInit(coVRPlugin *m)
{
    if (plugin == NULL)
        plugin = new CaveVOX(m);
    return 0;
}

// REQUIRED!
void coVRDelete(coVRPlugin *)
{
    delete plugin;
}

// coVRPreFrame()
// called before each frame
void coVRPreFrame()
{
    plugin->preFrame();
}

/// Constructor
CaveVOX::CaveVOX(coVRPlugin *)
{
    //create panel
    panel = new coPanel(new coFlatPanelGeometry(coUIElement::BLACK));
    handle = new coPopupHandle("CaveVOX");

    // Create main menu button
    caveVOXMenuItem = new coButtonMenuItem("CaveVOX");
    caveVOXMenuItem->setMenuListener(this);
    // add button to main menu (need to adjust)
    cover->getMenu()->add(caveVOXMenuItem);
}

/// Destructor
CaveVOX::~CaveVOX()
{
    delete panel;
    delete handle;
    delete caveVOXMenuItem;
}

void CaveVOX::menuEvent(coMenuItem *menuItem)
{
    // listen for initPDB frame to open close
    if (menuItem == caveVOXMenuItem)
    {
        if (handle->isVisible())
        {
            handle->setVisible(false);
        }
        else
        {
            handle->setVisible(true);
        }
    }
    handle->update();
}

// need to define because abstract
void CaveVOX::potiValueChanged(float, float, coValuePoti *, int)
{
}

// load new structure listener
void CaveVOX::buttonEvent(coButton *cobutton)
{
}

/// Called before each frame
void CaveVOX::preFrame()
{
    handle->update();
}
