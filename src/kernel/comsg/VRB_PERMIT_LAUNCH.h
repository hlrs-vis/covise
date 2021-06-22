/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COMSG_VRB_PERMIT_LAUNCH_H
#define COMSG_VRB_PERMIT_LAUNCH_H

#include <net/message_macros.h>
#include <util/coExport.h>
#include <vrb/ProgramType.h>
namespace covise{

  enum class VRB_PERMIT_LAUNCH_TYPE
  {
    Ask,
    Answer
  };

  DECL_MESSAGE_WITH_SUB_CLASSES(VRB_PERMIT_LAUNCH, VRB_PERMIT_LAUNCH_TYPE, VRBCLIENTEXPORT);
  DECL_SUB_MESSAGE_CLASS(VRB_PERMIT_LAUNCH, VRB_PERMIT_LAUNCH_TYPE, Ask, VRBCLIENTEXPORT, int, senderID, int, launcherID, vrb::Program, program);
  DECL_SUB_MESSAGE_CLASS(VRB_PERMIT_LAUNCH, VRB_PERMIT_LAUNCH_TYPE, Answer, VRBCLIENTEXPORT, int, requestorID, int, launcherID, bool, permit, int, code);
}

#endif // !COMSG_VRB_PERMIT_LAUNCH_H