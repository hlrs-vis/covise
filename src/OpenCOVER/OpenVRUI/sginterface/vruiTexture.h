/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRUI_TEXTURE_H
#define VRUI_TEXTURE_H

#include <util/coTypes.h>

namespace vrui
{

/// Userdata that can be attached to Nodes in the scenegraph
class OPENVRUIEXPORT vruiTexture
{
public:
    vruiTexture()
    {
    } ///< Constructor
    virtual ~vruiTexture();
};
}
#endif
