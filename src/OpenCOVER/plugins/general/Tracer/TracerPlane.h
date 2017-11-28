/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TRACER_PLANE_H
#define _TRACER_PLANE_H

#include <cover/coVRPluginSupport.h>
#include <osg/Geometry>

namespace opencover
{
class coVR3DTransRotInteractor;
class coVR2DTransInteractor;
class coPlane;
}

namespace vrui
{
class coTrackerButtonInteraction;
}

class TracerPlugin;

using namespace opencover;

/**
   s1, s2 as quad diagonal
   default quad normal=zaxis=dir3
   default dir=xaxis=dir1

     y
    dir2
     ^
     |
     |
    pos4----pos2
     |       |
     |       |
    pos1----pos3 ---> dir1  x

*/
class TracerPlane
{
protected:
    coInteractor *_inter;
    coVR3DTransRotInteractor *_s0;
    coVR2DTransInteractor *_s1, *_s2, *_s3, *_s4;
    vrui::coTrackerButtonInteraction *_directInteractor;
    coPlane *_plane;

    float _interSize;

    osg::Vec3 _pos0, _pos1, _pos2, _pos3, _pos4, _dir1, _dir2, _dir3, _diag;
    float _c, _b, _a;
    osg::ref_ptr<osg::Vec3Array> coordLine_;
    osg::ref_ptr<osg::Vec3Array> coordPoly_;
    osg::ref_ptr<osg::Vec3Array> polyNormal_;
    osg::ref_ptr<osg::Geode> geometryNode; ///< Geometry node
    osg::ref_ptr<osg::Geometry> geometryLine_; ///< Geometry object
    osg::ref_ptr<osg::Geometry> geometryPoly_; ///< Geometry object
    osg::Group *parent;
    virtual void computeQuad12();
    virtual void computeQuad1();
    virtual void computeQuad2();
    virtual void computeQuad34();
    virtual void computeQuad3();
    virtual void computeQuad4();

    void createGeometry();
    void updateGeometry();
    void deleteGeometry();

    bool _newModule;
    bool showPickInteractor_;
    bool showDirectInteractor_;
    bool _execOnChange;

    bool keepSquare_;
    bool wait_;
    std::string initialObjectName_; //we have to save it for the grmsg, because _inter is not

    bool _cyberclassroom; /// hide s1 s2 s3 s4 for cyberclassroom
    TracerPlugin *plugin;

public:
    // constructor
    TracerPlane(coInteractor *inter, TracerPlugin *p);

    // destructor
    virtual ~TracerPlane();

    // update after module execute
    virtual void update(coInteractor *inter);

    // set new flag
    virtual void setNew();

    // direct interaction
    virtual void preFrame();

    //show and make spheres intersectable
    virtual void showDirectInteractor();
    virtual void showPickInteractor();
    virtual void sendShowPickInteractorMsg();
    virtual void showGeometry();

    // hide
    virtual void hideDirectInteractor();
    virtual void hidePickInteractor();
    virtual void sendHidePickInteractorMsg();
    virtual void hideGeometry();

    virtual osg::Vec3 getParameterStartpoint1();
    virtual osg::Vec3 getParameterStartpoint2();
    virtual osg::Vec3 getParameterDirection();

    virtual void setParameterStartpoint1(osg::Vec3 sp1);
    virtual void setParameterStartpoint2(osg::Vec3 sp2);
    virtual void setParameterDirection(osg::Vec3 d);

    void setCaseTransform(osg::MatrixTransform *t);

    osg::Vec3 getStartpoint()
    {
        return _pos1;
    };
    osg::Vec3 getEndpoint()
    {
        return _pos2;
    };
    osg::Vec3 getDirection1()
    {
        return _dir1;
    };
    osg::Vec3 getDirection2()
    {
        return _dir2;
    };
    osg::Vec3 getDirection3()
    {
        return _dir3;
    };
    osg::Vec3 getPos3()
    {
        return _pos3;
    };
    osg::Vec3 getPos4()
    {
        return _pos4;
    };
    osg::Vec3 getPos0()
    {
        return _pos0;
    };

    bool wasStopped();
    bool wasStarted();
    bool isRunning();

    void keepSquare(bool k)
    {
        keepSquare_ = k;
    }

    void setStartpoint1(osg::Vec3 aVector);
    void setStartpoint2(osg::Vec3 aVector);
    void setDirection(osg::Vec3 aVector);

private:
    void updatePlane();
    TracerPlane();
};

// local variables:
// mode: c++
// c-basic-offset: 3
// end:

#endif
