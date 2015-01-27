/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ISOSURFACE_POINT_H
#define _ISOSURFACE_POINT_H

namespace vrui
{
class coTrackerButtonInteraction;
}
namespace opencover
{
class coVR3DTransInteractor;
class coInteractor;
}

#include <osg/Vec3>
class IsoSurfacePlugin;

class IsoSurfacePoint
{
private:
    opencover::coInteractor *inter_;
    bool newModule_;
    bool wait_;
    bool showPickInteractor_;
    bool showDirectInteractor_;

    osg::Vec3 isoPoint_;

    opencover::coVR3DTransInteractor *pointPickInteractor_;
    vrui::coTrackerButtonInteraction *pointDirectInteractor_;
    float interSize_;

    // extract the parameter values from coInteractor
    void getParameters();
    IsoSurfacePlugin *plugin;

public:
    // constructor
    IsoSurfacePoint(opencover::coInteractor *inter, IsoSurfacePlugin *p);

    // destructor
    ~IsoSurfacePoint();

    // update after module execute
    void update(opencover::coInteractor *inter);

    // set new flag
    void setNew();

    // direct interaction
    void preFrame();

    //show and make spheres intersectable
    void showPickInteractor();
    void showDirectInteractor();

    // hide
    void hideDirectInteractor();
    void hidePickInteractor();
};
#endif
