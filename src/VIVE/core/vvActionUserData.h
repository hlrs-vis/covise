/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <OpenVRUI/sginterface/vruiActionUserData.h>

namespace vive
{

/// Userdata that can be attached to Nodes in the scenegraph
class VVCORE_EXPORT vvActionUserData : public vrui::vruiActionUserData
{
public:
    vvActionUserData(vrui::coAction *a); ///< Constructor
    virtual ~vvActionUserData();
};
}
