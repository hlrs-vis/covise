/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GNUPLOT_H
#define _GNUPLOT_H

#ifdef WIN32
#include <winsock2.h>
#include <windows.h>
#include <tchar.h>
#include <stdio.h>
#include <strsafe.h>

#define BUFSIZE 4096

#endif

#include <do/coDoData.h>
#include <api/coSimpleModule.h>
#include <util/coviseCompat.h>

#include <do/coDoPolygons.h>
#include <do/coDoPixelImage.h>
#include <do/coDoTexture.h>

using namespace covise;

class Gnuplot : public coSimpleModule
{
private:
    virtual int compute(const char *port);
    coOutputPort *p_geom, *p_texture;
    coInputPort *p_data;

    coBooleanParam *windowed;
    coStringParam *p_command, *p_blocks;

    bool gnuplot_running;
    int fdpc[2]; // pipe parent -> client
    int fdcp[2]; // pipe client -> parent

    int writeStdIn(const void *_Buf, unsigned int _MaxCharCount);
    int readStdOut(void *_Buf, unsigned int _MaxCharCount);

#ifdef WIN32
    HANDLE g_hChildStd_IN_Rd;
    HANDLE g_hChildStd_IN_Wr;
    HANDLE g_hChildStd_OUT_Rd;
    HANDLE g_hChildStd_OUT_Wr;
    void CreateChildProcess();
    void ErrorExit(const TCHAR *lpszFunction);
#endif

public:
    Gnuplot(int argc, char *argv[]);
};
#endif
