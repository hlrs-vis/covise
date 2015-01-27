/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MASS_POINT_DYNAMICS_PLUGIN_H
#define _MASS_POINT_DYNAMICS_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: MassPointDynamics Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: Florian Seybold		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>

//#include "MassSpringState.h"
#include "MassSpringDamperSystem.h"
#include "RungeKutta.h"

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/PositionAttitudeTransform>

class MassPointDynamicsPlugin : public coVRPlugin
{
public:
    MassPointDynamicsPlugin();
    ~MassPointDynamicsPlugin();

    bool init();

    // this will be called in PreFrame
    void preFrame();

    // this will be called if an object with feedback arrives
    void newInteractor(RenderObject *container, coInteractor *i);

    // this will be called if a COVISE object arrives
    void addObject(RenderObject *container,
                   RenderObject *obj, RenderObject *normObj,
                   RenderObject *colorObj, RenderObject *texObj,
                   osg::Group *root,
                   int numCol, int colorBinding, int colorPacking,
                   float *r, float *g, float *b, int *packedCol,
                   int numNormals, int normalBinding,
                   float *xn, float *yn, float *zn, float transparency);

    // this will be called if a COVISE object has to be removed
    void removeObject(const char *objName, bool replace);

private:
    //MassSpringDamperSystem system;
    //RungeKutta4<MassSpringDamperSystem, MassSpringDamperSystem> systemIntegrator;
    //MassPoint* mp1, *mp2, *mp3, *mp4, *mp5;

    MassPointSystem system;
    std::vector<MassPoint> *massPointVector;
    std::vector<Joint *> *jointVector;
    RungeKutta<MassPoint, Joint> integrator;

    double time;

    osg::Sphere *sphere;
    osg::ShapeDrawable *sphereDrawable;
    osg::Geode *massPointGeode;
    std::vector<osg::PositionAttitudeTransform *> transformVector;

    osg::Vec3Array *planeVertices;
    osg::DrawArrays *planeBase;
    osg::Geometry *planeGeometry;
    osg::Geode *planeGeode;
};
#endif
