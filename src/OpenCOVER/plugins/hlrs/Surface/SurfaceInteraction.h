/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SurfaceInteraction_H
#define _SurfaceInteraction_H

//----------------------------------------------//
//												//
//												//
// Note:										//
// Obsolete, use MultitouchNavigation instead	//
//												//
//												//
//----------------------------------------------//

#include "SurfacePlugin.h"

class SurfaceContact;

class SurfaceInteraction
{
private:
    double angleBetween3DVectors(osg::Vec3 v1, osg::Vec3 v2);
    double _previousValue, _radius;
    osg::Vec3d _prev3DVec1, _prev3DVec2;
    osg::Vec3d _previousVector3D, _initial;
    int _counter;

public:
    /*enum RunningState
	{
		StateStarted = 0,
		StateRunning,
		StateStopped,
		StateNotRunning
	};*/

    SurfaceInteraction();
    virtual ~SurfaceInteraction();

    virtual void rotateXY(std::list<SurfaceContact> &contacts);
    virtual void moveXY(std::list<SurfaceContact> &contacts);
    virtual void scaleXYZ(std::list<SurfaceContact> &contacts);
    virtual void rotateZ(std::list<SurfaceContact> &contacts);
    virtual void moveZ(std::list<SurfaceContact> &contacts);
    void reset()
    {
        _counter = 0;
        _previousValue = _radius = 0.;
        _previousVector3D = _initial = osg::Vec3d(0., 0., 0.);
    };

    //bool wasStarted() const { return (runningState == StateStarted);    }
    //bool isRunning()  const { return (runningState == StateRunning);    }
    //bool wasStopped() const { return (runningState == StateStopped);    }
    //bool isIdle()     const { return (runningState == StateNotRunning); }

    //protected:
    //	RunningState runningState;
};
#endif
