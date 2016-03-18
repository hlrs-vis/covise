/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MultitouchNavigation_H
#define _MultitouchNavigation_H
/****************************************************************************\ 
**                                                            (C)2012 HLRS  **
**                                                                          **
** Description: Multitouch Navigation										**
**                                                                          **
**                                                                          **
** Author:																	**
**         Jens Dehlke														**
**                                                                          **
** History:  								                                **
** Sep-12  v1.0	    				       		                            **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include "MultitouchPlugin.h"
#include <cover/VRSceneGraph.h>
#include <cover/coVRConfig.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <osg/io_utils>

class TouchContact;

class MultitouchNavigation
{
private:
    double _previousValue, _initialValue, _radius;
    osg::Vec3d _prev3DVec1, _prev3DVec2;
    osg::Vec3d _previousVector3D, _initial;
    int _counter;

    double angleBetween3DVectors(osg::Vec3 v1, osg::Vec3 v2);

public:
    MultitouchNavigation();
    virtual ~MultitouchNavigation();

    virtual void rotateXY(TouchContact c);
    virtual void moveXY(TouchContact c);
    virtual void continuousMoveXY(TouchContact c);
    virtual void walkXY(TouchContact c);
    virtual void scaleXYZ(std::list<TouchContact> &contacts);
    virtual void continuousScaleXYZ(std::list<TouchContact> &contacts);
    virtual void rotateZ(std::list<TouchContact> &contacts);
    virtual void moveZ(TouchContact c);
    virtual void continuousMoveZ(TouchContact c);
    virtual void walkZ(TouchContact c);
    virtual void fly(TouchContact c);

    void reset()
    {
        _counter = 0;
        _previousValue = _initialValue = _radius = 0.;
        _previousVector3D = _initial = osg::Vec3d(0., 0., 0.);
    };
};
#endif
