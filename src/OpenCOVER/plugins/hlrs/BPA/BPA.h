/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _BPA_NODE_PLUGIN_H
#define _BPA_NODE_PLUGIN_H

#include <util/common.h>
#include <unordered_map>

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>

#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coTabletUI.h>
using namespace covise;
using namespace opencover;

#include <config/CoviseConfig.h>
#include <util/coTypes.h>
#include <osg/Material>
#include <osg/Vec3>
#include <osg/Array>
#include <osg/ShapeDrawable>

class BPA;

class Trajectory
{
public:
    Trajectory(BPA *bpa);
    ~Trajectory();
    osg::Vec3 startPos;
    osg::Vec3 startVelocity;
    osg::Vec4 color;
    osg::Material *mtl;
    osg::StateSet *lineState;
    float length;
    bool correctVelocity;
    double gamma;
    double D;
    double W;
    double alpha;
    double Rho;
    double viscosity;
    double surfacetension;
    double velo;
    double Vol;
    float kappa;
    bool recalcVelocities;
    float veloDiff;

    osg::Geometry *geom;
    osg::Vec3Array *vert;
    osg::Vec3Array *velos;
    osg::Vec4Array *colors;

    osg::DrawArrays *primitives;

    osg::ref_ptr<osg::Geode> geode;
    void createGeometry();
    void recalc();
    BPA *bpa;
    void computeVelocity();
    void setColor(float r, float g, float b, float a);
    float getMinimalDistance(Trajectory *t, osg::Vec3 &p1);
    float getMinimalDistanceSlow(Trajectory *t, osg::Vec3 &p1);

    double getX(double v);
    double getDX(double v);
    float distance(osg::Vec3 &p,osg::Vec3 &p0,osg::Vec3 &p1);
    float distance(Trajectory *t,int gi,int git);

    void setStartVelocity(BPA *bpa,float vel);
    bool recalcVeloDiff=true;
    void computeVeloDiff(osg::Vec3 &origin); // compute the velocity difference between origin and the start/end of the trajectory on the wall
};

class BPA : public coTUIListener
{
public:
    BPA(std::string filename, osg::Group *g);
    ~BPA();
    coTUIFloatSlider *velocity;
    coTUILabel *velocityLabel;
    coTUIFloatSlider *originVelocity;
    coTUILabel *originVelocityLabel;
    coTUIFloatSlider *length;
    coTUILabel *lengthLabel;
    coTUIColorButton *lineColor;
    coTUILabel *lineColorLabel;
    coTUILabel *originLabel;
    coTUILabel *filenameLabel;
    coTUILabel *rhoLabel;
    coTUIEditFloatField *rhoEdit;
    coTUILabel *viscosityLabel;
    coTUIEditFloatField *viscosityEdit;
    coTUILabel *stLabel;
    coTUIEditFloatField *stEdit;
    coTUIButton *minErrorButton;
    double standardDeviation;
    osg::Vec3 Origin;

    void recalc();

    void loadDxf(std::string filename);
    void loadTxt(std::string filename);
    void loadnfix(std::string filename);

    std::list<Trajectory *> trajectories;
    std::list<Trajectory *> left;
    std::list<Trajectory *> right;
    void tabletEvent(coTUIElement *tUIItem);
    void tabletPressEvent(coTUIElement *tUIItem);

    void setOriginVelocity(float ov);

    osg::MatrixTransform *sphereTrans;

    osg::Geode *geode;
    osg::Sphere *sphere;
    osg::ShapeDrawable *sphereDrawable;
    void calcIntersection();

    static void intersectLines(osg::Vec3 p0,osg::Vec3 p1,osg::Vec3 d0,osg::Vec3 d1,osg::Vec3 &c0, osg::Vec3 &c1);
    inline float distancePointLine(osg::Vec3 p0,osg::Vec3 d0, osg::Vec3 p){ return(d0 ^ (p0-p)).length()/d0.length();};

    osg::ref_ptr<osg::Group> trajectoriesGroup;
    float floorHeight;
};


class BPAPlugin : public coVRPlugin, public coTUIListener
{
public:
    BPAPlugin();
    ~BPAPlugin();
    bool init();
    static BPAPlugin *plugin;

    coTUITab *BPATab;
    coTUIToggleButton *airResistance;
    coTUIToggleButton *OriginComputationType;
    coTUIToggleButton *ignoreUpward;
    coTUIToggleButton *allToAll;
    coTUILabel *angleLabel;
    coTUIEditFloatField *angleEdit;
    coTUIButton *writeButton;
    coTUIEditFloatField *originVeloEdit;

    static int SloadBPA(const char *filename, osg::Group *parent, const char *ck = "");
    int loadBPA(const char *filename, osg::Group *parent);
    static int SunloadBPA(const char *filename, const char *ck = "");
    int unloadBPA(const char *filename);

    // this will be called in PreFrame
    void preFrame();
    osg::ref_ptr<osg::Group> BPAGroup;

    std::map<std::string, BPA *> bpa_map;
    std::list<BPA *> bpa_list;

private:
    void tabletEvent(coTUIElement *tUIItem);
    void tabletPressEvent(coTUIElement *tUIItem);
};
#endif
