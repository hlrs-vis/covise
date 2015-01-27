/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _RESULT_INT_PARAM_H
#define _RESULT_INT_PARAM_H

#include "ResultParam.h"

class SCAEXPORT ResultIntParam : public ResultParam
{
public:
    ResultIntParam(const char *name, int i);
    int getValue() const
    {
        return val_;
    };
    void setValue(int val);
    const char *getClosest(float &diff, int num, const char *const *entries);

private:
    int val_;
};
#endif
