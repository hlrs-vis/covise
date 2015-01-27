/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "NucleonInteractor.h"

#include <cover/coVRPluginSupport.h>
using namespace opencover;
using namespace covise;

NucleonInteractor::NucleonInteractor(osg::Vec3 initialPos, osg::Vec3 normal, float size, std::string geoFileName, osg::Vec3 finalPos, float nucleusRadius)
    : ElementaryParticleInteractor(initialPos, normal, size, geoFileName)
{

    if (cover->debugLevel(3))
        fprintf(stderr, "NucleonInteractor::NucleonInteractor finalPos=[%f %f %f]\n", finalPos[0], finalPos[1], finalPos[2]);

    nucleusRadius_ = nucleusRadius;

    finalPos_ = finalPos;
    ///initialPos_=startPos;
}

NucleonInteractor::~NucleonInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "NucleonInteractor::~NucleonInteractor\n");
}

/*
void NucleonInteractor::stopInteraction()
{
   coVRIntersectionInteractor::stopInteraction();

	// snap to final position
	osg::Vec3 pos = getPosition();
   osg::Vec3 dist=pos-center_;
   float d=dist.length();

   //fprintf(stderr,"nucleon pos=[%f %f %f]\n", pos[0], pos[1], pos[2]);
   //fprintf(stderr,"center =[%f %f %f]\n", center_[0], center_[1], center_[2]);
   //fprintf(stderr,"d=%f, nucleusRadius_=%f\n\n", d, nucleusRadius_);
   if ( d <= nucleusRadius_ )
   {
      //fprintf(stderr,"moving to final pos [%f %f %f]\n", finalPos_[0], finalPos_[1], finalPos_[2]);
	   updateTransform(finalPos_, normal_);
   }
   else
   {
      startAnimation();
   }
}
*/

bool NucleonInteractor::insideNucleus()
{
    osg::Vec3 pos = getPosition();
    float d = pos.length();

    if (d <= nucleusRadius_)
        return true;
    else
        return false;
}
