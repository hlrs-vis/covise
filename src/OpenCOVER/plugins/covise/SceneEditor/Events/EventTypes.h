/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EVENT_TYPES_H
#define EVENT_TYPES_H

class EventTypes
{
public:
    enum Type
    {
        NONE,
        PRE_FRAME_EVENT,
        START_MOUSE_EVENT,
        STOP_MOUSE_EVENT,
        DO_MOUSE_EVENT,
        DOUBLE_CLICK_EVENT,
        SET_TRANSFORM_AXIS_EVENT,
        MOUSE_ENTER_EVENT,
        MOUSE_EXIT_EVENT,
        SELECT_EVENT,
        DESELECT_EVENT,
        MOUNT_EVENT,
        UNMOUNT_EVENT,
        APPLY_MOUNT_RESTRICTIONS_EVENT,
        GET_CAMERA_EVENT,
        REPAINT_EVENT,
        SET_SIZE_EVENT,
        POST_INTERACTION_EVENT,
        TRANSFORM_CHANGED_EVENT, // changes in position (all sceneobjects) or size (room/shape/window)
        SWITCH_VARIANT_EVENT,
        SET_APPEARANCE_COLOR_EVENT,
        SETTINGS_CHANGED_EVENT,
        MOVE_OBJECT_EVENT,
        INIT_KINEMATICS_STATE_EVENT
    };
};

#endif
