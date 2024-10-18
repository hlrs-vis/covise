/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SolarSystemNode_H
#define SolarSystemNode_H

#include <config/CoviseConfig.h>
#include <util/byteswap.h>

#include <util/coTypes.h>
#include <cover/coVRTui.h>

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

class PLUGINEXPORT VrmlNodeSolarSystem : public VrmlNodeChild
{
public:
    // Define the fields of SteeringWheel nodes
    static void initFields(VrmlNodeSolarSystem *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeSolarSystem(VrmlScene *scene = 0);
    VrmlNodeSolarSystem(const VrmlNodeSolarSystem &n);

    virtual VrmlNodeSolarSystem *toSolarSystemWheel() const;

    void eventIn(double timeStamp, const char *eventName,
                 const VrmlField *fieldValue);

    void setVenusPosition(double x, double y, double z);
    void setMarsPosition(double x, double y, double z);
    void setEarthPosition(double x, double y, double z);
    void setSaturnPosition(double x, double y, double z);
    void setJupiterPosition(double x, double y, double z);
    void setComet_CG_Position(double x, double y, double z);
    void setRosettaPosition(double x, double y, double z);
    void setPlanetScale(double s);
    static VrmlNodeSolarSystem *instance()
    {
        if(!inst)
        {
            inst = new VrmlNodeSolarSystem();
            initFields(inst, nullptr);
        }
        return inst;
    };

private:
    double timeStamp;

    // Fields
    VrmlSFRotation d_venusRotation;
    VrmlSFVec3f d_venusTranslation;
    VrmlSFRotation d_marsRotation;
    VrmlSFVec3f d_marsTranslation;
    VrmlSFRotation d_earthRotation;
    VrmlSFVec3f d_earthTranslation;
    VrmlSFRotation d_saturnRotation;
    VrmlSFVec3f d_saturnTranslation;
    VrmlSFRotation d_jupiterRotation;
    VrmlSFVec3f d_jupiterTranslation;
    VrmlSFVec3f d_comet_CG_Translation;
    VrmlSFVec3f d_rosettaTranslation;

    VrmlSFVec3f d_planetScale;
    static VrmlNodeSolarSystem *inst;
};

template<>
inline VrmlNode *VrmlNodeTemplate::creator<VrmlNodeSolarSystem>(vrml::VrmlScene *scene){
    return VrmlNodeSolarSystem::instance();
}

#endif
