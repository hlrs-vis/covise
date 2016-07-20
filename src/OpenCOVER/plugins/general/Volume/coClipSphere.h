/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef CO_CLIP_SPHERE_H
#define CO_CLIP_SPHERE_H

#include <boost/scoped_ptr.hpp>

#include <osg/Matrix>
#include <osg/Vec3>

#include <PluginUtil/coVR3DTransRotInteractor.h>

class coClipSphere
{
public:

    coClipSphere();

    void preFrame();

    void setValid(bool valid);
    bool valid() const;
    void setActive(bool active);
    bool active() const;
    void setInteractorActive(bool interactorActive);
    bool interactorActive() const;

    void setRadius(float radius);
    float radius() const;
    void setMatrix(osg::Matrix const& m);
    osg::Vec3 getPosition() const;

private:

    boost::scoped_ptr<opencover::coVR3DTransRotInteractor> pickInteractor_;

    bool valid_;
    bool active_;
    bool interactorActive_;
    float radius_;

};

#endif
