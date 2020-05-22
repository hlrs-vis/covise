#ifndef __ROBOTTYPES__
#define __ROBOTTYPES__

#include "CLink.h"
#include "CJoint.h"

typedef std::vector<CJoint>     JointHandler;
typedef std::vector<CLink>      LinkHandler;

enum
{
    //This parameter is about how many variable used for end effector
    //position and orientation : 3 for position, 3 for orientation
    NUMBEROFSETPARAMETERS = 6
};

#endif