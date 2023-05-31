/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
// Description:   ModelViewer
//
// Author:        Jurgen Schulze (jschulze@ucsd.edu)
//
// Creation Date: 2006-09-17
//
// **************************************************************************

#include <iostream>
#include <ostream>

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
#include <OpenVRUI/coRectButtonGeometry.h>
#include <OpenVRUI/coValuePoti.h>
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coFrame.h>
#include <config/CoviseConfig.h>
#include <string.h>

// OSG:
#include <osg/Node>
#include <osgDB/ReadFile>

// Local:
#include "ModelViewer.h"

using std::cerr;
using std::endl;
using covise::coCoviseConfig;

/// Constructor
ModelViewer::ModelViewer()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

bool ModelViewer::init()
{
    //create panel
    panel = new coPanel(new coFlatPanelGeometry(coUIElement::BLACK));
    handle = new coPopupHandle("ModelViewer");

    // Create main menu button
    modelViewerMenuItem = new coButtonMenuItem("ModelViewer");
    modelViewerMenuItem->setMenuListener(this);
    // add button to main menu (need to adjust)
    cover->getMenu()->add(modelViewerMenuItem);

    std::string modelFile = coCoviseConfig::getEntry("COVER.Plugin.ModelViewer.ModelFile");
    cerr << "Trying to read file: " << modelFile << " ..." << endl;
    if (!modelFile.empty())
    {
        osg::Node *modelNode = osgDB::readNodeFile(modelFile);
        if (modelNode == NULL)
            cerr << "Error reading file" << endl;
        else
        {
            cover->getObjectsRoot()->addChild(modelNode);
            cerr << "File read." << endl;
        }
    }
    else
    {
        cerr << "Error: COVER.Plugin.ModelViewer.ModelFile needs to point to a 3D model file" << endl;
    }

    return true;
}

/// Destructor
ModelViewer::~ModelViewer()
{
    delete panel;
    delete handle;
    delete modelViewerMenuItem;
}

void ModelViewer::menuEvent(coMenuItem *menuItem)
{
    // listen for initPDB frame to open close
    if (menuItem == modelViewerMenuItem)
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
void ModelViewer::potiValueChanged(float, float, coValuePoti *, int)
{
}

// load new structure listener
void ModelViewer::buttonEvent(coButton *)
{
}

/// Called before each frame
void ModelViewer::preFrame()
{
    handle->update();
}

COVERPLUGIN(ModelViewer)
