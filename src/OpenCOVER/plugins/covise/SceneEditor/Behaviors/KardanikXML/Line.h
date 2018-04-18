/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * Line.h
 *
 *  Created on: Jan 2, 2012
 *      Author: jw_te
 */

#pragma once

#include <memory>

namespace KardanikXML
{

class Point;

class Line
{
private:
public:
    Line();
    Line(std::shared_ptr<Point> pointA, std::shared_ptr<Point> pointB);

    void SetPointA(std::shared_ptr<Point> pointA);
    std::shared_ptr<Point> GetPointA() const;

    void SetPointB(std::shared_ptr<Point> pointB);
    std::shared_ptr<Point> GetPointB() const;

    float GetRadius() const;
    void SetRadius(float radius);

private:
    float m_Radius;

    std::shared_ptr<Point> m_PointA;
    std::shared_ptr<Point> m_PointB;
};
}
