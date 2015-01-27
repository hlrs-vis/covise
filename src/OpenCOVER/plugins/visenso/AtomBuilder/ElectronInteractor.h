/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ELECTRON_INTERACTOR_H
#define _ELECTRON_INTERACTOR_H

#include "ElementaryParticleInteractor.h"
#include <osg/Vec3>

class ElectronInteractor : public ElementaryParticleInteractor
{
private:
    float nucleusRadius_, kShellRadius_, lShellRadius_, mShellRadius_;
    osg::Vec3 center_;

protected:
public:
    // position and normal in object coordinates
    // size in world coordinates (mm)
    ElectronInteractor(osg::Vec3 pos, osg::Vec3 normal, float size, std::string geofilename, float nucleusRadius, float kShellRadius, float lShellRadius, float mShellRadius);

    virtual ~ElectronInteractor();
    //virtual void stopInteraction();
    bool insideKShell();
    bool insideLShell();
    bool insideMShell();
    float getAngle();
};

#endif
