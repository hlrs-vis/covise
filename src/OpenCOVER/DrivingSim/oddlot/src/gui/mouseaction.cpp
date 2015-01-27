/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10.05.2010
**
**************************************************************************/

#include "mouseaction.hpp"

//################//
// CONSTRUCTOR    //
//################//

MouseAction::MouseAction(MouseAction::MouseActionType mouseActionType, QGraphicsSceneMouseEvent *mouseEvent)
    : mouseActionType_(mouseActionType)
    , event_(mouseEvent)
    , intercepted_(false)
{
}
