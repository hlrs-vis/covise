/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_RADIO_BUTTON_H_
#define _CUI_RADIO_BUTTON_H_

#include "CheckBox.h"

namespace cui
{
class CUIEXPORT RadioButton : public CheckBox
{
public:
    RadioButton(Interaction *);
    virtual ~RadioButton(){};

    virtual void buttonEvent(InputDevice *, int);
};
}

#endif
