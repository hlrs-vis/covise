/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  ViewerPf.h
//  Class for display of VRML models using Performer.
//

#ifndef COSENSIVESENSOR_H
#define COSENSIVESENSOR_H

#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <OpenVRUI/coInteractionManager.h>
#include <osg/Matrix>
#include <util/coExport.h>
#include <util/common.h>
#include <vrml97/vrml/config.h>
#include <vrml97/vrml/Viewer.h>
#include <vrml97/vrml/VrmlNode.h>

#include <cover/coVRPluginSupport.h>
#include <cover/ui/Owner.h>
#include <util/DLinkList.h>
#include <PluginUtil/coSensor.h>


namespace opencover {
namespace ui {
class Action;
}
}

using namespace vrml;
using namespace opencover;

namespace osg
{
class Node;
};

class osgViewerObject;
class coCubeMap;

namespace vrui
{
class coTrackerButtonInteraction;
}

class VRML97COVEREXPORT coSensiveSensor : public coPickSensor, public ui::Owner
{
public:
    static bool modified;

    enum SensorType
    {
        NONE = 0,
        PICK
    };
    coSensiveSensor(osg::Node *n, osgViewerObject *vObj, void *object, vrml::VrmlScene *s, osg::MatrixTransform *VRoot);
    virtual ~coSensiveSensor();
    virtual void update();
    virtual void disactivate();
    virtual int getType()
    {
        if (d_scene)
            return PICK;
        else
            return NONE;
    };
    void remove()
    {
        d_scene = NULL;
    };
    void *getVrmlObject()
    {
        return vrmlObject;
    };
    ui::Action *getButton() const
    {
        return button;
    };

protected:
    osg::Vec3 firstHitPoint;
    osg::Matrix firstInvPointerMat;
    int pointerGrabbed;
    void *vrmlObject;
    VrmlScene *d_scene;
    osgViewerObject *viewerObj;
    int wasReleased;
    float distance;
    bool childActive;
    coSensiveSensor *parentSensor;
    void resetChildActive();
    void setChildActive();
    //PointerTooltip *tt;
    vrui::coTrackerButtonInteraction *VrmlInteraction;
    ui::Action *button = nullptr;
};
#endif
