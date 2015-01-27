/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "PointsParser.h"
#include <ctype.h>
#include <string.h>

PointsParser::PointsParser(const char *argPtr)
    : _OK(false)
{
    float x, y, z;
    char c1, c2;
    int numChars;

    while (*argPtr && isspace(*argPtr))
        argPtr++;

    while (*argPtr)
    {
        int res;
        res = sscanf(argPtr, "%c %f %f %f %c%n", &c1, &x, &y, &z, &c2, &numChars);
        if (res < 5)
            res = sscanf(argPtr, "%c%f,%f,%f%c%n", &c1, &x, &y, &z, &c2, &numChars);
        if (res < 5)
            res = sscanf(argPtr, "%c%f;%f;%f%c%n", &c1, &x, &y, &z, &c2, &numChars);
        if (res < 5)
            res = sscanf(argPtr, "%c%f/%f/%f%c%n", &c1, &x, &y, &z, &c2, &numChars);

        if (res == 5)
        {
            // This is a valid coordinate set
            _points.push_back(x);
            _points.push_back(y);
            _points.push_back(z);
        }
        else
        {
            return;
        }

        if (numChars == 0)
            numChars = 1;
        if (numChars <= (int)strlen(argPtr))
            argPtr += numChars;
        else
            break;

        while (*argPtr && isspace(*argPtr))
            argPtr++;
    }
    _OK = true;
}

PointsParser::~PointsParser()
{
}

bool
PointsParser::IsOK() const
{
    return _OK;
}

int
PointsParser::getNoPoints() const
{
    return (int)(_points.size() / 3);
}

void
PointsParser::getPoints(float **xc, float **yc, float **zc) const
{
    int no_points = getNoPoints();
    *xc = new float[no_points];
    *yc = new float[no_points];
    *zc = new float[no_points];
    int point = 0, coord = 0;
    for (; point < no_points; ++point)
    {
        (*xc)[point] = _points[coord++];
        (*yc)[point] = _points[coord++];
        (*zc)[point] = _points[coord++];
    }
}
