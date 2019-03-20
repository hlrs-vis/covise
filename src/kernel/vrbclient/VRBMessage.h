/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRB_MESSAGE_H
#define VRB_MESSAGE_H

namespace covise
{

enum vrb_gui_subtype
{
    LOAD_FILE = 0,
    NEW_FILE,
    DO_QUIT,
    LOAD_PLUGIN,
    NUM_GUI_SUBTYPES
};
}
namespace vrb
{
VRBEXPORT enum vrbMessageType
{
    LOCK = 0,
    UNLOCK,
    AVATAR,
    TIMESTEP,
    TIMESTEP_ANIMATE,
    TIMESTEP_SYNCRONIZE,
    SYNC_MODE,
    MASTER,
    SLAVE,
    MOVE_HAND,
    MOVE_HEAD,
    AR_VIDEO_FRAME,
    SYNC_KEYBOARD,
    ADD_SELECTION,
    CLEAR_SELECTION




};
}
#endif
