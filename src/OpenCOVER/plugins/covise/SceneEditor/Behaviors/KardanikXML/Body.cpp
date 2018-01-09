/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Body.h"

#include <stdexcept>
#include <string>

#include <boost/foreach.hpp>

#include "Line.h"
#include "LineStrip.h"

#include "Point.h"

using namespace std;

namespace KardanikXML
{

Body::Body()
    : m_MotionID(0U)
{
}

Body::Body(const string &name)
    : m_Name(name)
    , m_MotionID(0U)
{
}

const Body::LineStrips &
Body::GetLineStrips() const
{
    return m_LineStrips;
}

void Body::AddLineStrip(std::shared_ptr<LineStrip> lineStrip)
{
    m_LineStrips.push_back(lineStrip);
    BOOST_FOREACH (std::shared_ptr<Point> point, lineStrip->GetPoints())
    {
        AddPoint(point);
    }
}

const Body::Lines &
Body::GetLines() const
{
    return m_Lines;
}

void Body::AddLine(std::shared_ptr<Line> line)
{
    m_Lines.push_back(line);
    AddPoint(line->GetPointA());
    AddPoint(line->GetPointB());
}

const Body::Points &
Body::GetPoints() const
{
    return m_Points;
}

void Body::AddPoint(std::shared_ptr<Point> point)
{
    m_Points.push_back(point);
}

void Body::SetName(const string &name)
{
    m_Name = name;
}

const string &
Body::GetName() const
{
    return m_Name;
}

Body::MotionType Body::GetMotionType() const
{
    return m_MotionType;
}

void Body::SetRadius(float radius)
{
    m_Radius = radius;
}

float Body::GetRadius() const
{
    return m_Radius;
}

void Body::SetMotionType(MotionType motionType)
{
    m_MotionType = motionType;
}

std::shared_ptr<Point> Body::GetPointByName(const string &name) const
{
    BOOST_FOREACH (std::shared_ptr<Point> point, m_Points)
    {
        if (point && point->GetName() == name)
        {
            return point;
        }
    }
    return std::shared_ptr<Point>();
}

void Body::SetMotionType(const string &motionType)
{
    if (motionType == "MOTION_STATIC")
    {
        m_MotionType = MOTION_STATIC;
    }
    else if (motionType == "MOTION_DYNAMIC")
    {
        m_MotionType = MOTION_DYNAMIC;
    }
    else if (motionType == "MOTION_DYNAMIC_NOCOLLISION")
    {
        m_MotionType = MOTION_DYNAMIC_NOCOLLISION;
    }
    else
    {
        throw invalid_argument(string("Unknown Motion Type \"") + motionType + string("\""));
    }
}

std::shared_ptr<Anchor> Body::GetAnchor() const
{
    return m_Anchor;
}

void Body::SetAnchor(std::shared_ptr<Anchor> anchor)
{
    m_Anchor = anchor;
}

void Body::SetMotionID(unsigned int id)
{
    m_MotionID = id;
}

unsigned int Body::GetMotionID() const
{
    return m_MotionID;
}

void Body::AddConnectedJoint(std::shared_ptr<Joint> connectedJoint)
{
    m_ConnectedJoints.push_back(connectedJoint);
}

const std::vector<std::weak_ptr<Joint> > &Body::GetConnectedJoints() const
{
    return m_ConnectedJoints;
}

const Body::OperatingRanges &Body::GetOperatingRanges() const
{
    return m_OperatingRanges;
}

void Body::AddOperatingRange(std::shared_ptr<OperatingRange> range)
{
    m_OperatingRanges.push_back(range);
}

void Body::SetParentConstruction(std::weak_ptr<Construction> construction)
{
    m_ParentConstruction = construction;
}

std::weak_ptr<Construction> Body::GetParentConstruction() const
{
    return m_ParentConstruction;
}
}
