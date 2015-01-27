/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ResultEnumParam.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

ResultEnumParam::ResultEnumParam(const char *name, int num, const char *const *enums, int curEnum)
    : ResultParam(ResultParam::ENUM)
{
    num_ = 0;
    id_ = curEnum;
    enums_ = NULL;

    setValue(num, enums, curEnum);

    setLabel(name, enums_[id_]);
}

ResultEnumParam::~ResultEnumParam()
{
    cleanLabels();
}

void
ResultEnumParam::cleanLabels()
{
    int i;
    if (num_ != 0)
    {
        for (i = 0; i < num_; i++)
        {
            delete[] enums_[i];
        }
        delete[] enums_;
        num_ = 0;
    }
}

void
ResultEnumParam::setEnumLabels(int num, const char *const *enums)
{
    int i;
    cleanLabels();

    num_ = num;
    enums_ = new char *[num_];
    for (i = 0; i < num_; i++)
    {
        enums_[i] = new char[strlen(enums[i]) + 1];
        strcpy(enums_[i], enums[i]);
    }
}

void
ResultEnumParam::setValue(int curEnum)
{
    if (curEnum >= num_)
    {
        return;
    }
    id_ = curEnum;
    setLabel(enums_[id_]);
}

void
ResultEnumParam::setValue(int num, const char *const *enums, int curEnum)
{
    if (curEnum >= num)
    {
        return;
    }
    setEnumLabels(num, enums);
    setValue(curEnum);
}

const char *
ResultEnumParam::getClosest(float &diff,
                            int num,
                            const char *const *entries)
{
    int min_idx = 0;
    int i;
    const char *cur_entry;

    for (i = 0; i < num; i++)
    {
        cur_entry = strstr(entries[i], "=");
        if (cur_entry != NULL)
        {
            cur_entry++;

            if (strcmp(enums_[id_], cur_entry) == 0)
            {
                min_idx = i;
                break;
            }
        }
    }

    if (i == num)
    {
        diff = FLT_MAX;
        return NULL;
    }

    diff = 0.0;
    return entries[min_idx];
}
