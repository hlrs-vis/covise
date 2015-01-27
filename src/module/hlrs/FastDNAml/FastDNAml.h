/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _FAST_DNA_ML_H
#define _FAST_DNA_ML_H

/**************************************************************************\ 
 **                                                                        **
 **                                                                        **
 ** Description: Online connection to FastDNAml Simulation                                 **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                                (C) 1995                                **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Author: Uwe Woessner                                                   **
 ** Date:   14.10.2003  V1.0                                               **
\**************************************************************************/
#include <api/coModule.h>
using namespace covise;
#include <util/coviseCompat.h>

#include "gui/gui.h"

extern char *readTreeFile(const char *fileName, int ignore_lengths, bool useColoring, bool addNames, bool lowRes, int hostID);
extern char *readTree(int ignore_lengths, bool useColoring, const char *buffer, bool addNames, bool lowRes, int hostID);
class FastDNAml : public coModule
{

private:
    // 'Global' gui vars
    gui_t *master_conn;

    //  member functions
    virtual int compute(const char *port);
    virtual void quit(void);

    //  Parameter names

    //  Shared memory data
    coDoText *descr;

    //  Local data
    int count;
    // iv tree sent to the viewer
    char *ivbuf;
    coOutputPort *p_ivOut;
    coFileBrowserParam *p_path;
    coStringParam *p_host;
    coIntScalarParam *p_port;
    coBooleanParam *p_getData;
    coBooleanParam *p_colors;
    coBooleanParam *p_length;
    coBooleanParam *p_names;
    coBooleanParam *p_lowres;

public:
    FastDNAml();

    virtual float idle();
    /// Overload this if you want to notice immediate mode parameter changes
    virtual void param(const char *paramName);
    virtual void postInst();

    ~FastDNAml();
};
#endif // _READ_IV_H
