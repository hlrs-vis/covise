/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MINMAX_NEW_H
#define _MINMAX_NEW_H

/**************************************************************************\ 
 ** (C)1994 RUS                                                            **
 ** (C)2001 VirCinity IT-Consulting GmbH, Stuttgart                        **
 **                                                                        **
 ** Description:  COVISE Tube application module                           **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                       (C) VirCinity 2000, 2001                         **
 **                                                                        **
 **                                                                        **
 ** Author:  R.Lang, D.Rantzau, Sasha Cioringa                             **
 **                                                                        **
 **                                                                        **
 ** Date:  18.05.94  V1.0                                                  **
 ** Date:  08.11.00                                                        **
\**************************************************************************/

#include <api/coSimpleModule.h>
using namespace covise;
#define MAX_BUCKETS 200

class MinMax : public coSimpleModule
{

private:
    virtual int compute(const char *port);

    // parameters
    coIntSliderParam *p_buck;

    // ports
    coInputPort *p_inPort1;
    coOutputPort *p_outPort1, *p_outPort2, *p_outPort3;

    //private data
    float min, max, *yplot;
    long buckets;

protected:
    virtual void preHandleObjects(coInputPort **);
    virtual void postHandleObjects(coOutputPort **);

public:
    MinMax(int argc, char *argv[]);
    virtual ~MinMax();
};
#endif
