/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRUI_ACTION_USER_DATA_H
#define VRUI_ACTION_USER_DATA_H

#include <OpenVRUI/coAction.h>
#include <OpenVRUI/sginterface/vruiUserData.h>

namespace vrui
{

/// Userdata that can be attached to Nodes in the scenegraph
class OPENVRUIEXPORT vruiActionUserData : public virtual vruiUserData
{
public:
    vruiActionUserData(coAction *a) ///< Constructor
    {
        action = a;
    }
    virtual ~vruiActionUserData();

    coAction *action; ///< the associated action
};
}
#endif
