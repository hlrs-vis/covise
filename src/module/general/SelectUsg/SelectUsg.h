/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SELECT_USG_H
#define _SELECT_USG_H

/**************************************************************************\ 
 ** (C)1997 RUS                                                            **
 ** (C)2001 VirCinity IT-Consulting GmbH, Stuttgart                        **
 **                                                                        **
 ** Description:  COVISE SelectUsg application module                      **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                       (C) VirCinity       2001                         **
 **                                                                        **
 **                                                                        **
 ** Authors:          Andreas Werner, Sasha Cioringa                       **
 **                                                                        **
 **                                                                        **
 ** Date:  30.05.01                                                        **
\**************************************************************************/

#include <api/coSimpleModule.h>
using namespace covise;

class SelectUsg : public coSimpleModule
{

private:
    virtual int compute(const char *port);

    virtual void copyAttributesToOutObj(coInputPort **, coOutputPort **, int);

    // parameters
    coChoiceParam *p_type;
    coStringParam *p_selection;

    // ports
    coInputPort *p_inPort1, *p_inPort2, *p_inPort3;
    coOutputPort *p_outPort1, *p_outPort2;

    //private data
    int run_no;

public:
    SelectUsg(int argc, char *argv[]);
};
#endif
