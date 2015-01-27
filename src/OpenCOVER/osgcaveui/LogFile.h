/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LOG_FILE_H_
#define _LOG_FILE_H_

// CUI:
#include "Dial.h"
#include "Marker.h"
#include "PickBox.h"

// C++:
#include <stdio.h>

class vvStopwatch;

namespace cui
{

class CUIEXPORT LogFile
{
protected:
    FILE *_fp;
    vvStopwatch *_sw;

public:
    enum MoveType
    {
        WAND = 0, // moved with wand movement or rotation
        TRACKBALL // moved with trackball (tumbled)
    };
    LogFile(const char *);
    ~LogFile();
    void addDialChangedLog(cui::Dial *);
    void addCardClickedLog(cui::Card *, int, int);
    void addButtonStateLog(int, int, int);
    void addPickBoxMovedLog(cui::PickBox *, MoveType);
    void addMarkerLog(int, cui::Marker *);
    void addLog(const char *);
    void logStart();
    void logStop();
    void logInputDevices(osg::Matrix &, osg::Matrix &);
};
}

#endif
