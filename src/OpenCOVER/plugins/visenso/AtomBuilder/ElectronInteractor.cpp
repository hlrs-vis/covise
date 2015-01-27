/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ElectronInteractor.h"
#include <cover/coVRPluginSupport.h>
using namespace opencover;
using namespace covise;
ElectronInteractor::ElectronInteractor(osg::Vec3 pos, osg::Vec3 normal, float size, std::string geoFileName, float nucleusRadius, float kShellRadius, float lShellRadius, float mShellRadius)
    : ElementaryParticleInteractor(pos, normal, size, geoFileName)
{

    if (cover->debugLevel(3))
        fprintf(stderr, "ElectronInteractor::ElectronInteractor\n");

    nucleusRadius_ = nucleusRadius;
    kShellRadius_ = kShellRadius;
    lShellRadius_ = lShellRadius;
    mShellRadius_ = mShellRadius;
}

ElectronInteractor::~ElectronInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "ElectronInteractor::~ElectronInteractor\n");
}

/*

void ElectronInteractor::stopInteraction()
{
   coVRIntersectionInteractor::stopInteraction();

	// snap to middle radius
	osg::Vec3 pos = getPosition();

   float d=pos.length();
   osg::Vec3 dir=pos;
   dir.normalize();
   osg::Vec3 snapPos=dir;

   if ( (d >= nucleusRadius_) && (d <= kShellRadius_) )
   {
      //fprintf(stderr,"ElementaryParticleInteractor::stopInteraction inside kshell\n");
      
	   snapPos*=nucleusRadius_+0.5*(kShellRadius_ - nucleusRadius_);
     
	   //fprintf(stderr,"pos=%f %f %f snapPos=%f %f %f\n", pos[0], pos[1], pos[2], snapPos[0], snapPos[1], snapPos[2]);
	   updateTransform(snapPos, normal_);
		 
   }

   else if ( (d >= kShellRadius_) && (d <= lShellRadius_) )
   {
      //fprintf(stderr,"ElementaryParticleInteractor::stopInteraction inside kshell\n");
      snapPos*=kShellRadius_ + 0.5*(lShellRadius_ - kShellRadius_);
   
      //fprintf(stderr,"pos=%f %f %f snapPos=%f %f %f\n", pos[0], pos[1], pos[2], snapPos[0], snapPos[1], snapPos[2]);
      updateTransform(snapPos, normal_);
         
   }

   else if ( (d >= lShellRadius_) && (d <= mShellRadius_) )
   {
      //fprintf(stderr,"ElementaryParticleInteractor::stopInteraction inside kshell\n");
      snapPos*=lShellRadius_ + 0.5*(mShellRadius_ - lShellRadius_);

      //fprintf(stderr,"pos=%f %f %f snapPos=%f %f %f\n", pos[0], pos[1], pos[2], snapPos[0], snapPos[1], snapPos[2]);
      updateTransform(snapPos, normal_);
         
   }
   else
   {
      startAnimation(initialPos_);
   }
   
	
}
*/
bool ElectronInteractor::insideKShell()
{

    osg::Vec3 pos = getPosition();
    float d = pos.length();

    if ((d >= nucleusRadius_) && (d <= kShellRadius_))
        return true;
    else
        return false;
}

bool ElectronInteractor::insideLShell()
{

    osg::Vec3 pos = getPosition();
    float d = pos.length();

    if ((d >= kShellRadius_) && (d <= lShellRadius_))
        return true;
    else
        return false;
}

bool ElectronInteractor::insideMShell()
{

    osg::Vec3 pos = getPosition();
    float d = pos.length();

    if ((d >= lShellRadius_) && (d <= mShellRadius_))
        return true;
    else
        return false;
}

float ElectronInteractor::getAngle()
{
    // nicht allgemeingueltig sondern nur fuer normale 0,1,0
    osg::Vec3 v = getPosition();
    float a;
    v.normalize();
    float dot = v * (osg::Vec3(0, 0, 1));
    a = acos(dot);
    if (v[0] < 0.0)
        a = 2 * M_PI - a;
    return a;
}
