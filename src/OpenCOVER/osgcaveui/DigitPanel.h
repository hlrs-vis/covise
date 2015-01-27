/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _DIGITPANEL_H_
#define _DIGITPANEL_H_

// CUI:
#include "Widget.h"
#include "Panel.h"
#include "DigitLabel.h"

namespace cui
{
class Interaction;

class CUIEXPORT PanelDigit
{
public:
    cui::DigitLabel *_digit;
    int _pos[2];

    PanelDigit(cui::DigitLabel *, int, int);
    ~PanelDigit();
};

class CUIEXPORT DigitPanel : public Panel
{
public:
    DigitPanel(Interaction *, Appearance, Movability);
    ~DigitPanel();
    void addDigit(cui::DigitLabel *, int, int);
    void removeDigit(DigitLabel *);
    bool setDigitPos(cui::DigitLabel *, int, int);

protected:
    std::list<PanelDigit *> _digits;
    int _minCol, _minRow, _maxCol, _maxRow;
    bool digitButtonEvent(DigitLabel *, int, int);
    bool digitCursorUpdate(DigitLabel *, InputDevice *);
    virtual void initGraphics();
    virtual void layout();
};
}
#endif
