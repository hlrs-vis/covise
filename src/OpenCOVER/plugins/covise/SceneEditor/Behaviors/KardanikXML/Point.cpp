/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Point.h"

namespace KardanikXML
{

Point::Point()
    : m_X(0.0f)
    , m_Y(0.0f)
    , m_Z(0.0f)
{
}

Point::Point(float x, float y, float z)
    : m_X(x)
    , m_Y(y)
    , m_Z(z)
{
}

void
Point::SetName(const std::string &name)
{
    m_Name = name;
}

const std::string &
Point::GetName() const
{
    return m_Name;
}

void
Point::SetX(float x)
{
    m_X = x;
}

float
Point::GetX() const
{
    return m_X;
}

void
Point::SetY(float y)
{
    m_Y = y;
}

float
Point::GetY() const
{
    return m_Y;
}

void
Point::SetZ(float z)
{
    m_Z = z;
}

float
Point::GetZ() const
{
    return m_Z;
}
}
