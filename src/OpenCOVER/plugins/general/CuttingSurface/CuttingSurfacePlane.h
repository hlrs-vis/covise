/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUTTINGSURFACE_PLANE_H
#define _CUTTINGSURFACE_PLANE_H

#include "CuttingSurfaceInteraction.h"

namespace vrui
{
class coTrackerButtonInteraction;
class coCombinedButtonInteraction;
}

namespace opencover
{
class coVR3DTransRotInteractor;
class coInteractor;
class coPlane;
}

using namespace opencover;

class CuttingSurfacePlane
{
private:
    coInteractor *inter_;
    bool newModule_;
    bool wait_;
    bool showPickInteractor_;
    bool showDirectInteractor_;

    osg::Vec3 point_, normal_;

    coVR3DTransRotInteractor *planePickInteractor_;
    vrui::coTrackerButtonInteraction *planeDirectInteractor_;
    float interSize_;

    // extract the parameter values from coInteractor
    void getParameters();

    osg::ref_ptr<osg::Geode> geode_; ///< Geometry node
    osg::ref_ptr<osg::Geometry> outlineGeometry_; ///< Geometry object
    osg::ref_ptr<osg::Geometry> polyGeometry_; ///< Geometry object
    osg::ref_ptr<osg::Vec3Array> outlineCoords_;
    osg::ref_ptr<osg::Vec3Array> polyCoords_;
    osg::ref_ptr<osg::Vec3Array> polyNormal_;
    osg::Group *parent_;
    bool hasCase_;

    coPlane *testPlane_;
    bool intersectFlag_, oldIntersectFlag_;
    // geometry, intersection lines with bounding box
    void createGeometry();
    void deleteGeometry();
    void updateGeometry();
    void showGeometry(bool);
    int testIntersection();

public:
    // constructor
    CuttingSurfacePlane(coInteractor *inter, CuttingSurfacePlugin *p);

    // destructor
    ~CuttingSurfacePlane();

    // update after module execute
    void update(coInteractor *inter);

    // set new flag
    void setNew();

    // direct interaction
    void preFrame(int restrictToAxis = CuttingSurfaceInteraction::RESTRICT_NONE);
    void restrict(int restrictToAxis);

    //show and make spheres intersectable
    void showPickInteractor();
    void showDirectInteractor();

    // hide
    void hideDirectInteractor();
    void hidePickInteractor();

    void setInteractorPoint(osg::Vec3 p);
    void setInteractorNormal(osg::Vec3 n);
    const osg::Vec3 getInteractorPoint()
    {
        return point_;
    };
    const osg::Vec3 getInteractorNormal()
    {
        return normal_;
    };

    bool sendClipPlane();

    void setCaseTransform(osg::MatrixTransform *t);
    CuttingSurfacePlugin *plugin;
};
#endif
