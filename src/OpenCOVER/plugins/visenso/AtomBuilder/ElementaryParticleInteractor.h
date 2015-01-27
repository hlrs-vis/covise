/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ELEMENTARY_PARTICLE_INTERACTOR_H
#define _ELEMENTARY_PARTICLE_INTERACTOR_H

#include <PluginUtil/coVR2DTransInteractor.h>
#include <osg/Vec3>

using namespace opencover;
using namespace covise;
class ElementaryParticleInteractor : public coVR2DTransInteractor
{
private:
    std::string geoFileName_;

    bool animate_, radiusAnimate_;
    float animTime_, length_, radiusAnimTime_, radiusLength_;
    osg::Vec3 animStartPos_, animEndPos_, animVector_;
    float radiusAnimStartAngle_, radiusAnimEndAngle_, radiusAnimDiffAngle_, radiusAnimRadius_;
    bool minus_;

protected:
    void createGeometry();
    osg::Vec3 normal_;
    osg::Vec3 initialPos_;

public:
    // position and normal in object coordinates
    // size in world coordinates (mm)
    ElementaryParticleInteractor(osg::Vec3 initialPos, osg::Vec3 normal, float size, std::string geofilename);

    virtual ~ElementaryParticleInteractor();
    virtual void resetPosition();
    virtual void preFrame();
    void startAnimation(osg::Vec3 to);
    void startRadiusAnimation(float r, float rstart, float aend);
    osg::Vec3 getInitialPosition()
    {
        return initialPos_;
    };
    void setInitialPosition(osg::Vec3 p)
    {
        initialPos_ = p;
    };
};

#endif
