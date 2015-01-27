/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PointRelative.h"

namespace KardanikXML
{

PointRelative::PointRelative()
{
}

PointRelative::PointRelative(const std::string &pointRefID, float xOffset, float yOffset, float zOffset)
    : m_PointRefID(pointRefID)
    , m_XOffset(xOffset)
    , m_YOffset(yOffset)
    , m_ZOffset(zOffset)
{
}

void PointRelative::SetXOffset(float x)
{
    m_XOffset = x;
}

float PointRelative::GetXOffset() const
{
    return m_XOffset;
}

void PointRelative::SetYOffset(float y)
{
    m_YOffset = y;
}

float PointRelative::GetYOffset() const
{
    return m_YOffset;
}

void PointRelative::SetZOffset(float z)
{
    m_ZOffset = z;
}

float PointRelative::GetZOffset() const
{
    return m_ZOffset;
}

void PointRelative::ResolveAgainst(std::tr1::shared_ptr<Point> referencePoint)
{
    float x = m_XOffset;
    float y = m_YOffset;
    float z = m_ZOffset;

    if (referencePoint)
    {
        x = referencePoint->GetX() + m_XOffset;
        y = referencePoint->GetY() + m_YOffset;
        z = referencePoint->GetZ() + m_ZOffset;
    }
    SetX(x);
    SetY(y);
    SetZ(z);
}

const std::string &PointRelative::GetPointRefID() const
{
    return m_PointRefID;
}
}
