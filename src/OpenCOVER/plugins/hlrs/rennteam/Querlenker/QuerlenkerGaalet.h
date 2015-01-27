/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _QuerlenkerGaalet_h
#define _QuerlenkerGaalet_h
/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Querlenker Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: Manuel Dorsch                                                   **
 **                                                                          **
 ** History:  		               	                                **
 ** Nov-01  v1	    		       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <cover/coVRTui.h>
#include <cover/coTabletUI.h>

#include "../../../../VehicleUtil/gaalet/include/gaalet.h"

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/Group>
#include <osg/PositionAttitudeTransform>

//#include "RoadSystem/RoadSystem.h"

using namespace covise;
using namespace opencover;

typedef gaalet::algebra<gaalet::signature<4, 1> > cm;

class Querlenker : public coVRPlugin, public coTUIListener
{
public:
    Querlenker();
    virtual ~Querlenker();

    bool init();

    // this will be called in PreFrame
    void preFrame();

    // this will be called if an object with feedback arrives
    void newInteractor(RenderObject *container, coInteractor *i)
    {
    }

    // this will be called if a COVISE object arrives
    void addObject(RenderObject *container,
                   RenderObject *obj, RenderObject *normObj,
                   RenderObject *colorObj, RenderObject *texObj,
                   osg::Group *root,
                   int numCol, int colorBinding, int colorPacking,
                   float *r, float *g, float *b, int *packedCol,
                   int numNormals, int normalBinding,
                   float *xn, float *yn, float *zn, float transparency)
    {
    }

    // this will be called if a COVISE object has to be removed
    void removeObject(const char *objName, bool replace)
    {
    }

    void key(int type, int keySym, int /*mod*/)
    {
    }

private:
    cm::mv<0x08>::type ep;
    cm::mv<0x10>::type em;

    cm::mv<0x08, 0x10>::type en;
    cm::mv<0x08, 0x10>::type ei;

    cm::mv<0x18>::type E;

    cm::mv<0x01>::type e1;
    cm::mv<0x02>::type e2;
    cm::mv<0x04>::type e3;
    cm::mv<0x1f>::type I;
    cm::mv<0x07>::type i;

    cm::mv<0x00>::type one;

    osg::PositionAttitudeTransform *sphereSETransform;
    osg::PositionAttitudeTransform *sphereNETransform;
    osg::PositionAttitudeTransform *sphereNWTransform;
    osg::PositionAttitudeTransform *sphereSWTransform;
    osg::PositionAttitudeTransform *sphereFSLTransform;
    osg::PositionAttitudeTransform *sphereFSBLTransform;
    osg::PositionAttitudeTransform *sphereFBLTransform;
    osg::PositionAttitudeTransform *sphereFPBLTransform;
    osg::PositionAttitudeTransform *sphereFTOLTransform;
    osg::PositionAttitudeTransform *sphereFTLTransform;
    osg::PositionAttitudeTransform *sphereWFLTransform;
    osg::PositionAttitudeTransform *sphereWFL1Transform;
    osg::PositionAttitudeTransform *sphereWFL2Transform;
    osg::PositionAttitudeTransform *sphereWFL3Transform;
    osg::Cylinder *cylinderfl;
    osg::ShapeDrawable *cylinderflDrawable;
    osg::Cylinder *cylinderfu;
    osg::ShapeDrawable *cylinderfuDrawable;
    osg::Cylinder *cylinderfp;
    osg::ShapeDrawable *cylinderfpDrawable;
    osg::Cylinder *cylinderfpb;
    osg::ShapeDrawable *cylinderfpbDrawable;
    osg::Cylinder *cylinderfsb;
    osg::ShapeDrawable *cylinderfsbDrawable;
    osg::Cylinder *cylinderfpsb;
    osg::ShapeDrawable *cylinderfpsbDrawable;
    osg::Cylinder *cylinderdpwfl;
    osg::ShapeDrawable *cylinderdpwflDrawable;
    osg::Cylinder *cylinderdpfsbl;
    osg::ShapeDrawable *cylinderdpfsblDrawable;
    osg::Cylinder *cylinderdpflol;
    osg::ShapeDrawable *cylinderdpflolDrawable;

    double alpha;
    double beta;
    double gamma;

    typedef cm::mv<0x00, 0x03, 0x05, 0x06, 0x09, 0x0a, 0x0c, 0x11, 0x12, 0x14>::type D_type;

    typedef cm::mv<0x03, 0x05, 0x06, 0x09, 0x0a, 0x0c, 0x11, 0x12, 0x14, 0x18>::type Pp_type;

    coTUITab *querlenkerTab;
    coTUIFloatSlider *alphaSlider;
    coTUIFloatSlider *betaSlider;
    coTUIFloatSlider *gammaSlider;
};
#endif
