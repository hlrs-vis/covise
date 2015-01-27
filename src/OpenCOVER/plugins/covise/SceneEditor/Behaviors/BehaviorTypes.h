/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef BEHAVIOR_TYPES_H
#define BEHAVIOR_TYPES_H

class BehaviorTypes
{
public:
    // Important: The order of the Behaviors defines the order the events will be received.
    //            Behaviors on top of the list will receive events first.
    // Current constraints:
    //   KINEMATICS_BEHAVIOR before TRANSFORM_BEHAVIOR (the object will only be transformed if a static part is picked)
    enum Type
    {
        NONE,
        MOUNT_BEHAVIOR,
        KINEMATICS_BEHAVIOR,
        TRANSFORM_BEHAVIOR,
        CAMERA_BEHAVIOR,
        SINUS_SCALING_BEHAVIOR,
        APPEARANCE_BEHAVIOR,
        HIGHLIGHT_BEHAVIOR,
        VARIANT_BEHAVIOR
    };
};

#endif
