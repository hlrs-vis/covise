/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _NULEON_INTERACTOR_H
#define _NULEON_INTERACTOR_H

#include "ElementaryParticleInteractor.h"
#include <osg/Vec3>

class NucleonInteractor : public ElementaryParticleInteractor
{
private:
    float nucleusRadius_;
    osg::Vec3 finalPos_;

protected:
public:
    // position and normal in object coordinates
    // size in world coordinates (mm)
    NucleonInteractor(osg::Vec3 startPos, osg::Vec3 normal, float size, std::string geofilename, osg::Vec3 finalPos, float nucleusRadius);

    virtual ~NucleonInteractor();
    ///virtual void stopInteraction();
    bool insideNucleus();

    osg::Vec3 getFinalPosition()
    {
        return finalPos_;
    };
};

#endif
