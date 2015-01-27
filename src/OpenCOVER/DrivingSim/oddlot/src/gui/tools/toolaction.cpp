/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   31.03.2010
**
**************************************************************************/

#include "toolaction.hpp"

#include "src/util/odd.hpp"

//################//
// CONSTRUCTOR    //
//################//

ToolAction::ToolAction(ODD::EditorId editorId, ODD::ToolId toolId)
    : editorId_(editorId)
    , toolID_(toolId)
{
}
