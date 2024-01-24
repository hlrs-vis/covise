/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUTTINGSURFACE_BOX_H
#define _CUTTINGSURFACE_BOX_H

namespace vrui
{
class coTrackerButtonInteraction;
}
namespace opencover
{
class coVR3DTransInteractor;
class coInteractor;
} // namespace opencover

using namespace opencover;

#include <osg/Vec3>

class CuttingSurfaceBox
{
private:
    coInteractor *inter_;
    bool newModule_;
    bool wait_;
    bool showPickInteractor_;
    bool showDirectInteractor_;

    osg::Vec3 centerPoint_, radiusPoint_;

    coVR3DTransInteractor *cornerBPickInteractor_;
    coVR3DTransInteractor *cornerAPickInteractor_;
    vrui::coTrackerButtonInteraction *cornerDirectInteractor_;
    float interSize_;

    // extract the parameter values from coInteractor
    void getParameters();

    // todo geometry should be a a white line or a transparent sphere between center and radius which is updated during interaction
    void showGeometry();
    void updateGeometry();
    void hideGeometry();
    void tmpHidePickInteractor();

public:
    // constructor
    CuttingSurfaceBox(coInteractor *inter);

    // destructor
    ~CuttingSurfaceBox();

    // update after module execute
    void update(coInteractor *inter);

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
