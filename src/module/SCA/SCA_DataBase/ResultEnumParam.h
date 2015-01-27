/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _RESULT_ENUM_PARAM_H
#define _RESULT_ENUM_PARAM_H

#include "ResultParam.h"

class SCAEXPORT ResultEnumParam : public ResultParam
{
public:
    ResultEnumParam(const char *name, int num, const char *const *enums, int curEnum);
    ~ResultEnumParam();

    const char *getValue() const
    {
        return enums_[id_];
    };
    void setValue(int num);
    void setValue(int num, const char *const *enums, int curEnum);

    const char *getClosest(float &diff, int num, const char *const *entries);

private:
    int num_;
    int id_;
    char **enums_;

    void setEnumLabels(int num, const char *const *enums);
    void cleanLabels();
};
#endif
