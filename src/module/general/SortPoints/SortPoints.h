/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
**                                                           (C)2007 ZAIK **
**                                                                        **
** Description: Sort points according to associated data                  **
**                                                                        **
**      Author: Martin Aumueller (aumueller@uni-koeln.de)                 **
**                                                                        **
\**************************************************************************/

#ifndef SORTPOINTS_H
#define SORTPOINTS_H

#include <api/coSimpleModule.h>
using namespace covise;

class SortPoints : public coSimpleModule
{
private:
    // Ports:
    coInputPort *m_pointIn;
    coInputPort *m_dataIn;
    coOutputPort *m_pointLowerOut;
    coOutputPort *m_dataLowerOut;
    coOutputPort *m_pointMiddleOut;
    coOutputPort *m_dataMiddleOut;
    coOutputPort *m_pointUpperOut;
    coOutputPort *m_dataUpperOut;

    // Parameters:
    coFloatParam *m_lowerBound;
    coFloatParam *m_upperBound;

    // Methods:
    virtual int compute(const char *port);

public:
    SortPoints(int argc, char *argv[]);
};

#endif
