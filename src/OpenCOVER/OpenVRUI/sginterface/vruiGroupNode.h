/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRUI_GROUPNODE_H
#define VRUI_GROUPNODE_H

#include <OpenVRUI/sginterface/vruiNode.h>

namespace vrui
{

class OPENVRUIEXPORT vruiGroupNode : public virtual vruiNode
{

public:
    vruiGroupNode();
    virtual ~vruiGroupNode();;
};
}
#endif
