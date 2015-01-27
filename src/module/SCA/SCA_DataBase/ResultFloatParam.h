/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _RESULT_FLOAT_PARAM_H
#define _RESULT_FLOAT_PARAM_H

#include "ResultParam.h"

class SCAEXPORT ResultFloatParam : public ResultParam
{
public:
    ResultFloatParam(const char *name, float f, int precision);
    float getValue() const
    {
        return val_;
    };
    void setValue(float val);
    const char *getClosest(float &diff, int num, const char *const *entries);

private:
    float val_;
    int prec_;

    void fillValString(char *val);
};
#endif
