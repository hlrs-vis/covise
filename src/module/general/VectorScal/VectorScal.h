/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VECT_SCAL_NEW_H
#define _VECT_SCAL_NEW_H

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  COVISE VectScal application module                       **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) Vircinity 2000                         **
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
#include <util/coviseCompat.h>

class VectScal : public coSimpleModule
{
    COMODULE

private:
    virtual int compute(const char *port);

    // parameters

    coChoiceParam *p_option;

    // ports
    coInputPort *p_inPort;
    coOutputPort *p_outPort;

    // private data

public:
    VectScal(int argc, char *argv[]);
};
#endif
