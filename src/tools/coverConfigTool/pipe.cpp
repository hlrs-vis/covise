/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 ** pipe.cpp
 ** 2004-01-29, Matthias Feurer
 ****************************************************************************/

#include <qstring.h>
#include "pipe.h"
#include "window.h"

Pipe::Pipe()
{
    index = 0;
    hardPipe = 0;
    display = ":0.0";
    windowMap = WindowMap();
}

void Pipe::setIndex(int i)
{
    index = i;
}

void Pipe::setHardPipe(int i)
{
    hardPipe = i;
}

void Pipe::setDisplay(QString s)
{
    display = s;
}

void Pipe::setWindowMap(WindowMap wm)
{
    windowMap = wm;
}

void Pipe::addWindow(QString name, Window w)
{
    windowMap[name] = w;
}

int Pipe::getIndex()
{
    return index;
}

int Pipe::getHardPipe()
{
    return hardPipe;
}

QString Pipe::getDisplay()
{
    return display;
}

WindowMap *Pipe::getWindowMap()
{
    return &windowMap;
}

int Pipe::getNumWindows()
{
    return windowMap.count();
}
