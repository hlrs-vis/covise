/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  MODULE IntToFloat
//
//  Make a coDoFloat object out of a coDoIntArr
//
//  Initial version: 13-12-07 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2001 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _INT_TO_FLOAT_
#define _INT_TO_FLOAT_

#include <api/coSimpleModule.h>
using namespace covise;

class IntToFloat : public coSimpleModule
{
public:
    coInputPort *p_int;
    coOutputPort *p_float;
    IntToFloat();
    ~IntToFloat()
    {
    }

protected:
private:
    virtual int compute();
};
#endif
