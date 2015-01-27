/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ResultFloatParam.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>

ResultFloatParam::ResultFloatParam(const char *name, float f, int precission)
    : ResultParam(ResultParam::FLOAT)
{
    val_ = f;
    prec_ = precission;

    char val[1024];
    fillValString(val);

    setLabel(name, val);
}

void
ResultFloatParam::fillValString(char *val)
{
    char format[100], prec[10];
    strcpy(format, "%1.");
    sprintf(prec, "%d", prec_);
    strcat(format, prec);
    strcat(format, "e");

    sprintf(val, format, val_);
}

void
ResultFloatParam::setValue(float f)
{
    val_ = f;

    char val[1024];
    fillValString(val);

    setLabel(val);
}

const char *
ResultFloatParam::getClosest(float &diff,
                             int num,
                             const char *const *entries)
{
    float fmin = FLT_MAX, fcur = 0.0;
    int min_idx = 0;
    int i;
    const char *cur_entry;

    for (i = 0; i < num; i++)
    {
        cur_entry = strstr(entries[i], "=");
        if (cur_entry)
        {
            cur_entry++;

            sscanf(cur_entry, "%e", &fcur);
            if (fabs(val_ - fcur) < fmin)
            {
                fmin = fabs(val_ - fcur);
                min_idx = i;
            }
        }
    }

    diff = fmin;

    return entries[min_idx];
}
