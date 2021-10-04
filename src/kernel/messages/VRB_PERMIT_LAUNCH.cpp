/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VRB_PERMIT_LAUNCH.h"
#include <net/message.h>
#include <net/message_sender_interface.h>
#include <net/message_types.h>
#include <net/tokenbuffer.h>
#include <net/tokenbuffer_serializer.h>
#include <net/tokenbuffer_util.h>
namespace covise
{
  IMPL_MESSAGE_WITH_SUB_CLASSES(VRB_PERMIT_LAUNCH, VRB_PERMIT_LAUNCH_TYPE);
  IMPL_SUB_MESSAGE_CLASS(VRB_PERMIT_LAUNCH, VRB_PERMIT_LAUNCH_TYPE, Ask, int, senderID, int, launcherID, covise::Program, program);
  IMPL_SUB_MESSAGE_CLASS(VRB_PERMIT_LAUNCH, VRB_PERMIT_LAUNCH_TYPE, Answer, int, requestorID, int, launcherID, bool, permit, int, code);
  IMPL_SUB_MESSAGE_CLASS(VRB_PERMIT_LAUNCH, VRB_PERMIT_LAUNCH_TYPE, Abort, int, requestorID, int, launcherID, covise::Program, program);
}
