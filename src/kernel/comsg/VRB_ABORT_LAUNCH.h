/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COMSG_VRB_ABORT_LAUNCH_H
#define COMSG_VRB_ABORT_LAUNCH_H

#include <net/message_macros.h>
#include <util/coExport.h>

namespace covise{

DECL_MESSAGE_CLASS(VRB_PERMIT_LAUNCH, VRBCLIENTEXPORT, int, requestorID, int, launcherID, bool, permit);

}

#endif // !COMSG_VRB_ABORT_LAUNCH_H