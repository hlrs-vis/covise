/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _RemoteVehicle_NODE_PLUGIN_H
#define _RemoteVehicle_NODE_PLUGIN_H

#include <util/common.h>
#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>
#include <util/byteswap.h>

#include <util/coTypes.h>

#include <vrml97/vrml/Player.h>
#include <vrml97/vrml/config.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/coEventQueue.h>
#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/System.h>
#include <vrml97/vrml/Viewer.h>
#include <vrml97/vrml/VrmlScene.h>
#include <vrml97/vrml/VrmlNamespace.h>
#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFBool.h>
#include <vrml97/vrml/VrmlMFFloat.h>
#include <vrml97/vrml/VrmlSFInt.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFRotation.h>
#include <vrml97/vrml/VrmlMFInt.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>

using namespace vrml;
using namespace opencover;

class PLUGINEXPORT VrmlNodeRemoteVehicle : public VrmlNodeChild
{
public:
    static VrmlNodeRemoteVehicle *instance();

    static void initFields(VrmlNodeRemoteVehicle *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeRemoteVehicle(VrmlScene *scene = 0);
    VrmlNodeRemoteVehicle(const VrmlNodeRemoteVehicle &n);

    virtual VrmlNodeRemoteVehicle *toRemoteVehicle() const;

    void eventIn(double timeStamp, const char *eventName,
                 const VrmlField *fieldValue);

    void setVRMLVehicle(const osg::Matrix &trans);
    virtual void render(Viewer *);

private:
    double timeStamp;

    static VrmlNodeRemoteVehicle *singleton;
    VrmlSFRotation d_carRotation;
    VrmlSFVec3f d_carTranslation;
    VrmlSFRotation d_carBodyRotation;
    VrmlSFVec3f d_carBodyTranslation;
};

#endif
