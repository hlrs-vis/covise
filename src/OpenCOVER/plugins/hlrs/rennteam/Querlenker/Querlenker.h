/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Querlenker_h
#define _Querlenker_h
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

//#include "../../../../VehicleUtil/gealg/GeometricAlgebra.h"
//#include "/mnt/raid/home/hpcmdors/src/gealg/head/GeometricAlgebra.h"
#include <gaalet.h>

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/Group>
#include <osg/PositionAttitudeTransform>

//#include "RoadSystem/RoadSystem.h"

using namespace covise;
using namespace opencover;

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
    osg::PositionAttitudeTransform *sphereFLLTransform;
    osg::PositionAttitudeTransform *sphereFLL2Transform;
    osg::PositionAttitudeTransform *sphereFULTransform;
    osg::PositionAttitudeTransform *sphereFUL2Transform;
    osg::PositionAttitudeTransform *sphereFLOLTransform;
    osg::PositionAttitudeTransform *sphereFUOLTransform;
    osg::PositionAttitudeTransform *sphereFSLTransform;
    osg::PositionAttitudeTransform *sphereFSBLTransform;
    osg::PositionAttitudeTransform *sphereFBLTransform;
    osg::PositionAttitudeTransform *sphereFPBLTransform;
    osg::PositionAttitudeTransform *sphereFTOLTransform;
    osg::PositionAttitudeTransform *sphereFTLTransform;
    osg::PositionAttitudeTransform *sphereWFL1Transform;
    osg::PositionAttitudeTransform *sphereWFL2Transform;
    osg::PositionAttitudeTransform *sphereWFL3Transform;
    osg::PositionAttitudeTransform *sphereWFL4Transform;
    osg::PositionAttitudeTransform *sphereMATransform;
    osg::Cylinder *cylinderfl;
    osg::ShapeDrawable *cylinderflDrawable;
    osg::Cylinder *cylinderfl2;
    osg::ShapeDrawable *cylinderfl2Drawable;
    osg::Cylinder *cylinderfu;
    osg::ShapeDrawable *cylinderfuDrawable;
    osg::Cylinder *cylinderfu2;
    osg::ShapeDrawable *cylinderfu2Drawable;
    osg::Cylinder *cylinderfp;
    osg::ShapeDrawable *cylinderfpDrawable;
    osg::Cylinder *cylinderfs;
    osg::ShapeDrawable *cylinderfsDrawable;
    osg::Cylinder *cylinderfpb;
    osg::ShapeDrawable *cylinderfpbDrawable;
    osg::Cylinder *cylinderfsb;
    osg::ShapeDrawable *cylinderfsbDrawable;
    osg::Cylinder *cylinderfpsb;
    osg::ShapeDrawable *cylinderfpsbDrawable;
    osg::Cylinder *cylinderftl;
    osg::ShapeDrawable *cylinderftlDrawable;
    osg::Cylinder *cylinderffv;
    /*osg::ShapeDrawable* cylinderffvDrawable;
	osg::Cylinder* cylinderffvp;
	osg::ShapeDrawable* cylinderffvpDrawable;
	osg::Cylinder* cylinderffvo;
	osg::ShapeDrawable* cylinderffvoDrawable; */

    double alpha;
    double gamma;

    typedef gaalet::algebra<gaalet::signature<4, 1> > cm;

    typedef cm::mv<1, 2, 4>::type Vector;
    typedef cm::mv<1, 2, 4, 8, 0x10>::type Point;
    typedef Point Sphere;
    typedef cm::mv<1, 2, 4, 8, 0x10>::type Plane;

    typedef cm::mv<0, 3, 5, 6>::type Rotor;

    typedef cm::mv<0x03, 0x05, 0x06, 0x09, 0x0a, 0x0c, 0x11, 0x12, 0x14>::type S_type;
    typedef cm::mv<0x00, 0x03, 0x05, 0x06, 0x09, 0x0a, 0x0c, 0x0f, 0x11, 0x12, 0x14, 0x17>::type D_type;
    cm::mv<0x01>::type e1;
    cm::mv<0x02>::type e2;
    cm::mv<0x04>::type e3;
    cm::mv<0x08>::type ep;
    cm::mv<0x10>::type em;

    cm::mv<0x00>::type one;

    cm::mv<0x08, 0x10>::type e0;
    cm::mv<0x08, 0x10>::type einf;

    cm::mv<0x18>::type E;

    cm::mv<0x1f>::type Ic;
    cm::mv<0x07>::type Ie;

    /*typedef typeof(gealg::mv<6, 0x171412110f0c, 0x10>::type()+gealg::mv<6, 0x0a0906050300, 0x10>::type()) D_expr_t;
   typedef D_expr_t::result_type D_type;

   typedef typeof(gealg::mv<5, 0x181412110c, 0x10>::type()+gealg::mv<5, 0x0a09060503, 0x10>::type()) Pp_expr_t;
   typedef Pp_expr_t::result_type Pp_type;

   typedef typeof(gealg::mv<5, 0x1c1a191615, 0x10>::type()+gealg::mv<5, 0x130e0d0b07, 0x10>::type()) L_expr_t;
   typedef L_expr_t::result_type L_type;

   typedef std::pair<D_type, L_type> DL_type;*/

    void Radaufhaengung_wfr(double u_wfr, double steerAngle, D_type &D_wfr, Vector &nrc_wfr) const;
    //DL_type Radaufhaengung(double alpha, double gamma);

    coTUITab *querlenkerTab;
    coTUIFloatSlider *alphaSlider;
    coTUIFloatSlider *gammaSlider;
};
#endif
