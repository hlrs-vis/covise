/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _POINTS_PARSER_H_
#define _POINTS_PARSER_H_

#include <util/coviseCompat.h>

class PointsParser
{
public:
    PointsParser(const char *points);
    virtual ~PointsParser();
    bool IsOK() const;
    int getNoPoints() const;
    void getPoints(float **xc, float **yc, float **zc) const;

protected:
private:
    vector<float> _points;
    bool _OK;
};
#endif
