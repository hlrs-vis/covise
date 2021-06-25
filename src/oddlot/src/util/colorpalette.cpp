/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   08.04.2010
**
**************************************************************************/

#include "colorpalette.hpp"

ColorPalette::ColorPalette()
{
    brightRed_ = QColor(190, 91, 91);
    darkRed_ = QColor(177, 40, 11);

    brightOrange_ = QColor(212, 181, 113);
    darkOrange_ = QColor(212, 147, 37);

    brightGreen_ = QColor(91, 190, 91);
    darkGreen_ = QColor(32, 119, 52);

    brightCyan_ = QColor(92, 185, 190);
    darkCyan_ = QColor(14, 128, 137);

    brightBlue_ = QColor(91, 135, 190);
    darkBlue_ = QColor(44, 71, 145);

    brightPurple_ = QColor(188, 155, 225);
    darkPurple_ = QColor(137, 37, 212);

    brightGrey_ = QColor(175, 193, 182);
    darkGrey_ = QColor(82, 111, 96);
}
