/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "LineStrip.h"

using namespace std;
using namespace std::tr1;

namespace KardanikXML
{

LineStrip::LineStrip()
    : m_Radius(0.2f)
{
}

void LineStrip::AddPoint(std::tr1::shared_ptr<Point> point)
{
    m_Points.push_back(point);
}

float LineStrip::GetRadius() const
{
    return m_Radius;
}

void LineStrip::SetRadius(float radius)
{
    m_Radius = radius;
}

const LineStrip::Points &
LineStrip::GetPoints() const
{
    return m_Points;
}
}
