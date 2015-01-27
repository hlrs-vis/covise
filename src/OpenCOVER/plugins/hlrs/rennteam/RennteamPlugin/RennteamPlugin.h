/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _RennteamPlugin_h
#define _RennteamPlugin_h
/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Rennteam Plugin (does nothing)                              **
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
#include <cover/coVRTui.h>

//#include "/home/hpcyborg/src/gealg/head/CarDynamicsPA2004.h"
#include "gealg/CarDynamicsF07114.h"

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/Group>
#include <osg/PositionAttitudeTransform>

#include "RoadSystem/RoadSystem.h"

using namespace covise;
using namespace opencover;

class RennteamPlugin : public coVRPlugin
{
public:
    RennteamPlugin();
    ~RennteamPlugin();

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

    void key(int type, int keySym, int /*mod*/);

private:
    gealg::mv<1, 0x04>::type getGroundDistance(const gealg::mv<3, 0x040201>::type &);
    gealg::mv<6, 0x060504030201>::type getContactPoint(const gealg::mv<3, 0x040201>::type &p_w, Road *&road, double &u);

    osg::Cylinder *wheelShape;
    osg::ShapeDrawable *wheelDrawable;
    osg::Geode *wheelGeode;
    osg::PositionAttitudeTransform *wheelTransform;
    osg::PositionAttitudeTransform *wheelTransformFL;
    osg::PositionAttitudeTransform *wheelTransformFR;
    osg::PositionAttitudeTransform *wheelTransformRL;
    osg::PositionAttitudeTransform *wheelTransformRR;

    osg::Geode *planeGeode;
    osg::PositionAttitudeTransform *planeTransform;

    osg::Box *bodyShape;
    osg::ShapeDrawable *bodyDrawable;
    osg::Geode *bodyGeode;
    osg::PositionAttitudeTransform *bodyTransform;

    osg::Quat wheelQuatFL;
    osg::Quat wheelQuatFR;
    osg::Quat wheelQuatRL;
    osg::Quat wheelQuatRR;

    cardyn::StateVectorType y;
    cardyn::ExpressionVectorType dy;
    gealg::RungeKutta<cardyn::ExpressionVectorType, cardyn::StateVectorType, 19> integrator;

    double steerAngle;
    double u_wfl;
    double u_wfr;
    int gear;

    RoadSystem *roadSystem;
    osg::Group *roadGroup;
    Road *currentRoad[4];
    double currentLongPos[4];

    coTUITab *vdTab;
    coTUILabel *k_Pp_Label;
    coTUILabel *d_Pp_Label;
    coTUILabel *k_Pq_Label;
    coTUILabel *d_Pq_Label;
    coTUISlider *k_Pp_Slider;
    coTUISlider *d_Pp_Slider;
    coTUISlider *k_Pq_Slider;
    coTUISlider *d_Pq_Slider;

    gealg::mv<1, 0x08, 0x10>::type ep;
    gealg::mv<1, 0x10, 0x10>::type em;

    gealg::mv<2, 0x1008, 0x10>::type en;
    gealg::mv<2, 0x1008, 0x10>::type ei;

    gealg::mv<1, 0x18, 0x10>::type E;

    gealg::mv<1, 0x01, 0x10>::type e1;
    gealg::mv<1, 0x02, 0x10>::type e2;
    gealg::mv<1, 0x04, 0x10>::type e3;
    gealg::mv<1, 0x1f, 0x10>::type I;
    gealg::mv<1, 0x07, 0x10>::type i;

    gealg::mv<1, 0x00, 0x10>::type one;

    typedef typeof(gealg::mv<6, 0x171412110f0c, 0x10>::type() + gealg::mv<6, 0x0a0906050300, 0x10>::type()) D_expr_t;
    typedef D_expr_t::result_type D_type;

    typedef gealg::mv<5, 0x1008040201, 0x10>::type P_expr_t;
    typedef P_expr_t::result_type P_type;

    typedef std::pair<D_type, P_type> DP_type;

    gealg::mv<3, 0x040201, 0x10>::type v_wf;
    gealg::mv<4, 0x06050300, 0x10>::type R_ks;

    DP_type Radaufhaengung_wfl(double u_wfl, double steerAngle);
    DP_type Radaufhaengung_wfr(double u_wfr, double steerAngle);
};
#endif
