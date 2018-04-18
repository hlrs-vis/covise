/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "OperatingRange.h"

#include <stdexcept>
#include <string>
#include <sstream>

#include <boost/foreach.hpp>

#include "Point.h"
#include "Body.h"

using namespace std;

namespace KardanikXML
{

OperatingRange::OperatingRange()
    : m_Axis(Z_AXIS)
    , m_Radius(1.0f)
{
}

std::shared_ptr<Point> OperatingRange::GetCenterPoint() const
{
    return m_CenterPoint;
}

void OperatingRange::SetAxis(Axis axis)
{
    m_Axis = axis;
}

void OperatingRange::SetAxis(const std::string &axis)
{
    if (axis == "X_AXIS")
    {
        m_Axis = X_AXIS;
    }
    else if (axis == "Y_AXIS")
    {
        m_Axis = Y_AXIS;
    }
    else if (axis == "Z_AXIS")
    {
        m_Axis = Z_AXIS;
    }
    else
    {
        throw invalid_argument("Wrong axis specification.");
    }
}

OperatingRange::Axis OperatingRange::GetAxis() const
{
    return m_Axis;
}

void OperatingRange::SetCenterPoint(std::shared_ptr<Point> point)
{
    m_CenterPoint = point;
}

float OperatingRange::GetRadius() const
{
    return m_Radius;
}

void OperatingRange::SetRadius(float radius)
{
    m_Radius = radius;
}

void OperatingRange::SetColor(const osg::Vec4f &color)
{
    m_Color = color;
}

void OperatingRange::SetColor(const std::string &colorString)
{
    stringstream value(colorString);
    float val1, val2, val3, val4;
    value >> val1 >> val2 >> val3 >> val4;
    osg::Vec4f color(val1, val2, val3, val4);
    SetColor(color);
}

const osg::Vec4f &OperatingRange::GetColor() const
{
    return m_Color;
}
}
