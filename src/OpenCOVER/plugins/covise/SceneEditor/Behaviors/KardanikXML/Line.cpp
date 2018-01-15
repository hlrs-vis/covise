/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Line.h"

using namespace std;

namespace KardanikXML
{

Line::Line()
    : m_Radius(0.2f)
{
}

Line::Line(std::shared_ptr<Point> pointA, std::shared_ptr<Point> pointB)
    : m_Radius(0.2f)
    , m_PointA(pointA)
    , m_PointB(pointB)
{
}

void
Line::SetPointA(std::shared_ptr<Point> pointA)
{
    m_PointA = pointA;
}

std::shared_ptr<Point>
Line::GetPointA() const
{
    return m_PointA;
}

void
Line::SetPointB(std::shared_ptr<Point> pointB)
{
    m_PointB = pointB;
}

std::shared_ptr<Point>
Line::GetPointB() const
{
    return m_PointB;
}

float Line::GetRadius() const
{
    return m_Radius;
}

void Line::SetRadius(float radius)
{
    m_Radius = radius;
}
}
