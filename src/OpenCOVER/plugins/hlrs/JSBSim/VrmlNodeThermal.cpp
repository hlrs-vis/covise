#include "VrmlNodeThermal.h"

#include "JSBSim.h"

#include <vrml97/vrml/Viewer.h>

using namespace vrml;

void VrmlNodeThermal::initFields(VrmlNodeThermal *node, VrmlNodeType *t)
{
    initFieldsHelper(node, t,
        exposedField("direction", node->d_direction),
        exposedField("location", node->d_location),
        exposedField("maxBack", node->d_maxBack),
        exposedField("maxFront", node->d_maxFront),
        exposedField("minBack", node->d_minBack),
        exposedField("minFront", node->d_minFront),
        exposedField("height", node->d_height),
        exposedField("velocity", node->d_velocity),
        exposedField("turbulence", node->d_turbulence));
}

const char *VrmlNodeThermal::typeName()
{
    return "Thermal";
}

VrmlNodeThermal::VrmlNodeThermal(VrmlScene *scene)
    : VrmlNodeChild(scene, typeName())
    , d_direction(0, 0, 1)
    , d_location(0, 0, 0)
    , d_maxBack(10)
    , d_maxFront(10)
    , d_minBack(1)
    , d_minFront(1)
    , d_height(100)
    , d_velocity(0, 0, 4)
    , d_turbulence(0.0)
{
}

VrmlNodeThermal::VrmlNodeThermal(const VrmlNodeThermal &n)
    : VrmlNodeChild(n)
{
    d_direction = n.d_direction;
    d_location = n.d_location;
    d_maxBack = n.d_maxBack;
    d_maxFront = n.d_maxFront;
    d_minBack = n.d_minBack;
    d_minFront = n.d_minFront;
    d_height = n.d_height;
    d_velocity = n.d_velocity;
    d_turbulence = n.d_turbulence;
}

VrmlNodeThermal::~VrmlNodeThermal()
{
}

void VrmlNodeThermal::eventIn(double timeStamp,
    const char *eventName,
    const VrmlField *fieldValue)
{
    // if (strcmp(eventName, "carNumber"))
    //  {
    // }
    //  Check exposedFields
    // else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }
}

void VrmlNodeThermal::render(Viewer *viewer)
{
    // Is viewer inside the cylinder?
    float x, y, z;
    viewer->getPosition(&x, &y, &z);
    if (y > d_location.y() && y < d_location.y() + d_height.get())
    {
        VrmlSFVec3f toViewer(x, y, z);
        toViewer.subtract(&d_location); // now we have the vector to the viewer
        VrmlSFVec3f dir = d_direction;
        *(toViewer.get() + 1) = 0; // y == height = 0
        *(dir.get() + 1) = 0;
        float dist = (float)toViewer.length();
        toViewer.normalize();
        dir.normalize();
        // angle between the sound direction and the viewer
        float angle = (float)acos(toViewer.dot(&d_direction));
        // fprintf(stderr,"angle: %f",angle/M_PI*180.0);
        float cang = (float)cos(angle / 2.0);
        float rmin, rmax;
        double intensity;
        rmin = fabs(d_minBack.get() * d_minFront.get() / (cang * cang * (d_minBack.get() - d_minFront.get()) + d_minFront.get()));
        rmax = fabs(d_maxBack.get() * d_maxFront.get() / (cang * cang * (d_maxBack.get() - d_maxFront.get()) + d_maxFront.get()));
        // fprintf(stderr,"rmin: %f rmax: %f",rmin,rmax);
        if (dist <= rmin)
            intensity = 1.0;
        else if (dist > rmax)
            intensity = 0.0;
        else
        {
            intensity = (rmax - dist) / (rmax - rmin);
        }
        osg::Vec3 v(d_velocity.x(), -d_velocity.z(), d_velocity.y()); // velocities are in VRML orientation (y-up)
        v *= intensity;
        JSBSimPlugin::instance()->addThermal(v, d_turbulence.get() * intensity);
    }
    setModified();
}
