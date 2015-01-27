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

#ifndef COLORPALETTE_HPP
#define COLORPALETTE_HPP

#include <QColor>

class ColorPalette
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    ColorPalette();
    virtual ~ColorPalette()
    { /* does nothing */
    }

    // Colors //
    //
    const QColor &brightRed() const
    {
        return brightRed_;
    }
    const QColor &darkRed() const
    {
        return darkRed_;
    }

    const QColor &brightOrange() const
    {
        return brightOrange_;
    }
    const QColor &darkOrange() const
    {
        return darkOrange_;
    }

    const QColor &brightGreen() const
    {
        return brightGreen_;
    }
    const QColor &darkGreen() const
    {
        return darkGreen_;
    }

    const QColor &brightCyan() const
    {
        return brightCyan_;
    }
    const QColor &darkCyan() const
    {
        return darkCyan_;
    }

    const QColor &brightBlue() const
    {
        return brightBlue_;
    }
    const QColor &darkBlue() const
    {
        return darkBlue_;
    }

    const QColor &brightPurple() const
    {
        return brightPurple_;
    }
    const QColor &darkPurple() const
    {
        return darkPurple_;
    }

    const QColor &brightGrey() const
    {
        return brightGrey_;
    }
    const QColor &darkGrey() const
    {
        return darkGrey_;
    }

private:
    //	ColorPalette(){ /* not allowed */ }
    ColorPalette(const ColorPalette &)
    { /* not allowed */
    }

    //################//
    // PROPERTIES     //
    //################//

private:
    // Colors //
    //
    QColor brightRed_;
    QColor darkRed_;

    QColor brightOrange_;
    QColor darkOrange_;

    QColor brightGreen_;
    QColor darkGreen_;

    QColor brightCyan_;
    QColor darkCyan_;

    QColor brightBlue_;
    QColor darkBlue_;

    QColor brightPurple_;
    QColor darkPurple_;

    QColor brightGrey_;
    QColor darkGrey_;
};

#endif // COLORPALETTE_HPP
