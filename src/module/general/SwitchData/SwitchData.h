/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
**                                                           (C)2009 ZAIK **
**                                                                        **
** Description: Output a selectable input object.                         **
**                                                                        **
**      Author: Martin Aumueller (aumueller@uni-koeln.de)                 **
**                                                                        **
\**************************************************************************/

#ifndef SWITCHDATA_H
#define SWITCHDATA_H

#include <api/coModule.h>
using namespace covise;

class SwitchData : public coModule
{
private:
    // input
    coInputPort **m_dataIn;
    coInputPort *m_switchIn;

    // parameters
    coChoiceParam *m_switchParam;

    // output
    coOutputPort *m_dataOut;
    coOutputPort *m_switchOut;

    // methods
    virtual int compute(const char *port);

public:
    SwitchData(int argc, char *argv[]);
};

#endif
