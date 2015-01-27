/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *									*
 *          								*
 *                            (C) 1996					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			VRSpacePointer.h (Performer 2.0)	*
 *									*
 *	Description
 *									*
 *	Author			Frank Foehl				*
 *									*
 *	Date			15.1.1997				*
 *									*
 *	Status			in dev					*
 *									*
 ************************************************************************/

#ifndef __VR_SPACEPOINTER_H
#define __VR_SPACEPOINTER_H

#include <util/common.h>

#ifdef PHANTOM_TRACKER
#include "phantom.h"
#endif

#ifdef BG_LIB
//#include "bglib.h"
#endif

#define LEFT_UPPER_TOGGLE_MASK 0x0001
#define LEFT_MIDDLE_TOGGLE_MASK 0x0004
#define LEFT_LOWER_TOGGLE_MASK 0x0010

#define RIGHT_UPPER_TOGGLE_MASK 0x0002
#define RIGHT_MIDDLE_TOGGLE_MASK 0x0008
#define RIGHT_LOWER_TOGGLE_MASK 0x0020

#define LEFT_MOMENTARY_MASK 0x0040
#define RIGHT_MOMENTARY_MASK 0x0080

#define JOYSTICK_TRIGGER_MASK 0x0100

#define AIC1 0x01
#define AIC2 0x02
#define AIC3 0x04
#define AIC4 0x08
#define AIC5 0x10
#define AIC6 0x20
#define AIC7 0x40
#define AIC8 0x80

// base class
#include <osg/Matrix>

class INPUT_LEGACY_EXPORT VRSpacePointer
{

private:
    osg::Matrix matrix;

    float mx0, my0;
    float speed;
    float zero1, zero2, zero3;
    //pfCoord      coord0;
    int trackingType;
    int oldflags;
#ifdef PHANTOM_TRACKER
    int phid;
#endif
#ifdef BG_LIB
    Chans fb, bb;
#endif

public:
    VRSpacePointer();
    ~VRSpacePointer();

    void init(int tt);
    void update(osg::Matrix &mat, unsigned int *button);
};
#endif
