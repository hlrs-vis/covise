/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TRACER_LINE_H
#define _TRACER_LINE_H

namespace opencover
{
class coVR3DTransInteractor;
class coVR3DTransRotInteractor;
}

namespace vrui
{
class coTrackerButtonInteraction;
}

using namespace opencover;

#include <osg/Vec3>
#include <osg/Geode>
#include <osg/StateSet>

// s1, s2 as line points
//
// pos1----pos2
//
class TracerLine
{
private:
    coInteractor *_inter;
    coVR3DTransInteractor *_s1;
    coVR3DTransInteractor *_s2;
    coVR3DTransRotInteractor *_s0; // Angriffspunkt in Mitte
    vrui::coTrackerButtonInteraction *_directInteractor;

    float _interSize;

    osg::Vec3 _pos1, _pos2;

    osg::ref_ptr<osg::Vec3Array> coord;

    osg::ref_ptr<osg::StateSet> state;
    osg::ref_ptr<osg::Geode> geometryNode; ///< Geometry node
    osg::ref_ptr<osg::Geometry> geometry; ///< Geometry object
    osg::Group *parent;

    void createGeometry();
    void updateGeometry();
    void deleteGeometry();

    bool _newModule;
    bool showPickInteractor_;
    bool showDirectInteractor_;
    bool _execOnChange; //< Execute on change: read covise.config in C'Tor
    osg::Matrix computeM0();

    string initialObjectName_; //we have to save it for the grmsg, because _inter is not always valid
    //    osg::MatrixTransform *interactorCaseDCS_;
    TracerPlugin *plugin;

public:
    // constructor
    TracerLine(coInteractor *inter, TracerPlugin *p);

    // destructor
    ~TracerLine();

    // update after module execute
    void update(coInteractor *inter);

    // set new flag
    void setNew();

    // direct interaction
    void preFrame();

    //show and make spheres intersectable
    void showPickInteractor();
    void showDirectInteractor();
    void sendShowPickInteractorMsg();
    void showGeometry();

    // hide
    void hideDirectInteractor();
    void hidePickInteractor();
    void sendHidePickInteractorMsg();
    void hideGeometry();

    void setStartpoint(osg::Vec3 sp1);
    osg::Vec3 getStartpoint()
    {
        return _pos1;
    };
    void setEndpoint(osg::Vec3 sp2);
    osg::Vec3 getEndpoint()
    {
        return _pos2;
    };

    void setCaseTransform(osg::MatrixTransform *t);

    bool wasStopped();
    bool wasStarted();
    bool isRunning();
};

#endif
