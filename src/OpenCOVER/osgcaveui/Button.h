/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_BUTTON_H_
#define _CUI_BUTTON_H_

// Local:
#include "Widget.h"
#include "Card.h"

namespace cui
{
class Interaction;

/** This is the implementation of a push button, which triggers an action.
*/
class CUIEXPORT Button : public Card
{
public:
    Button(Interaction *);
};
}

#endif
