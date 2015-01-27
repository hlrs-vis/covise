/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ResultIntParam.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

ResultIntParam::ResultIntParam(const char *name, int i)
    : ResultParam(ResultParam::INT)
{
    val_ = i;

    char val[1024];
    sprintf(val, "%d", i);
    setLabel(name, val);
}

void
ResultIntParam::setValue(int i)
{
    val_ = i;

    char val[1024];
    sprintf(val, "%d", i);
    setLabel(val);
}

const char *
ResultIntParam::getClosest(float &diff,
                           int num,
                           const char *const *entries)
{
    int min = INT_MAX, min_idx = 0;
    int i;
    const char *cur_entry;

    for (i = 0; i < num; i++)
    {
        cur_entry = strstr(entries[i], "=");
        if (cur_entry)
        {
            cur_entry++;
            //cerr <<  entries[i] << " " << cur_entry << endl;
            if (abs(val_ - atoi(cur_entry)) < min)
            {
                min = abs(val_ - atoi(cur_entry));
                min_idx = i;
            }
        }
        //cerr << "MIN: " << min << endl;
    }

    diff = (float)min;

    return entries[min_idx];
}
