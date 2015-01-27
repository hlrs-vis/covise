/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SAI_H
#define _SAI_H

#include <stdio.h>

class CadmouldData
{
public:
    enum
    {
        FAIL = -1,
        SUCCESS = 0
    };

    int no_elements;
    int no_points;
    int *elem[3];
    float *thickness;
    float *points[3];
    float *value;
    int *connect;
    float *fill_time;
    int load(const char *meshfile, const char *datafile);
    int init(int allocPoints, int allocElem);
    ~CadmouldData();
    CadmouldData();
};
#endif //_SAI_H
