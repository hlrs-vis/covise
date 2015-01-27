/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * LineStrip.h
 *
 *  Created on: Jan 2, 2012
 *      Author: jw_te
 */

#pragma once

#include <vector>
#ifndef WIN32
#include <boost/tr1/memory.hpp>
#endif
#include <memory>

namespace KardanikXML
{

class Point;

class LineStrip
{
public:
    typedef std::vector<std::tr1::shared_ptr<Point> > Points;

    LineStrip();

    void AddPoint(std::tr1::shared_ptr<Point> point);
    const Points &GetPoints() const;
    float GetRadius() const;
    void SetRadius(float radius);

private:
    float m_Radius;

    Points m_Points;
};
}
