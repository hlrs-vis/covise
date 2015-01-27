/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 ** pipe.h
 ** 2004-01-20, Matthias Feurer
 ****************************************************************************/

#ifndef PIPE_H
#define PIPE_H

#include <qstring.h>
#include <qmap.h>
#include <qptrlist.h>
#include "window.h"

typedef QMap<QString, Window> WindowMap;

class Pipe
{

public:
    Pipe();
    void setIndex(int i);
    void setHardPipe(int i);
    void setDisplay(QString s);
    void setWindowMap(WindowMap wm);
    void addWindow(QString name, Window w);

    int getIndex();
    int getHardPipe();
    QString getDisplay();
    WindowMap *getWindowMap();
    int getNumWindows();

private:
    int index;
    int hardPipe;
    QString display;
    WindowMap windowMap;
};
#endif // PIPE_H
