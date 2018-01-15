/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * ConstructionParser.cpp
 *
 *  Created on: Feb 14, 2012
 *      Author: jw_te
 */

#include "ConstructionParser.h"

#include <sstream>
#include <iostream>
#include <iomanip>
#include <QDomElement>

#include "Body.h"
#include "Line.h"
#include "Point.h"
#include "PointRelative.h"
#include "Joint.h"
#include "BodyJointDesc.h"
#include "LineStrip.h"
#include "Anchor.h"
#include "Construction.h"
#include "OperatingRange.h"

using namespace std;

namespace KardanikXML
{

ConstructionParser::ConstructionParser()
    : m_NumUnnamedBodies(0U)
    , m_NumUnnamedPoints(0U)
    , m_DefaultLineSize(0.1f)
{
}

std::shared_ptr<Body> ConstructionParser::parseBodyElement(const QDomElement &bodyElement)
{
    std::shared_ptr<Body> body(new Body());
    string bodyName = getOrCreateBodyName(bodyElement);
    body->SetName(bodyName);
    string motionType = bodyElement.attribute("motionType", "MOTION_DYNAMIC_NOCOLLISION").toStdString();
    body->SetMotionType(motionType);
    unsigned int motionID = bodyElement.attribute("motionID").toUInt();
    body->SetMotionID(motionID);
    QDomNode child = bodyElement.firstChild();
    while (!child.isNull())
    {
        if (child.toElement().tagName() == "line")
        {
            std::shared_ptr<KardanikXML::Line> line = parseLineElement(child.toElement());
            body->AddLine(line);
        }
        else if (child.toElement().tagName() == "linestrip")
        {
            std::shared_ptr<KardanikXML::LineStrip> lineStrip = parseLinestripElement(child.toElement());
            body->AddLineStrip(lineStrip);
        }
        else if (child.toElement().tagName() == "point")
        {
            std::shared_ptr<KardanikXML::Point> point = parsePointElement(child.toElement());
            body->AddPoint(point);
        }
        else if (child.toElement().tagName() == "point_rel")
        {
            std::shared_ptr<KardanikXML::PointRelative> point = parsePointElementRelative(child.toElement());
            body->AddPoint(point);
        }
        else if (child.toElement().tagName() == "anchor")
        {
            std::shared_ptr<KardanikXML::Anchor> anchor = parseAnchorElement(child.toElement(), body);
            body->SetAnchor(anchor);
        }
        else if (child.toElement().tagName() == "operating_range")
        {
            std::shared_ptr<KardanikXML::OperatingRange> range = parseOperatingRangeElement(child.toElement(), body);
            body->AddOperatingRange(range);
        }
        child = child.nextSibling();
    }
    m_BodyMap.insert(BodyMap::value_type(bodyName, body));
    return body;
}

string ConstructionParser::getOrCreateBodyName(const QDomElement &bodyElement)
{
    string bodyName = bodyElement.attribute("name", "").toStdString();

    if (bodyName.empty())
    {
        ostringstream outName;
        outName << "UnnamedBody_" << setfill('0') << setw(4) << m_NumUnnamedBodies << ends;
        m_NumUnnamedBodies++;
        return outName.str();
    }
    else
    {
        return bodyName;
    }
}

std::shared_ptr<Line> ConstructionParser::parseLineElement(const QDomElement &lineElement)
{
    std::shared_ptr<Line> line(new Line());
    float radius = m_DefaultLineSize;
    if (lineElement.hasAttribute("radius"))
    {
        radius = lineElement.attribute("radius").toFloat();
    }
    line->SetRadius(radius);
    QDomElement pointAElement = lineElement.firstChildElement();
    if (!pointAElement.isNull())
    {
        std::shared_ptr<KardanikXML::Point> pointA = parsePoint(pointAElement);
        line->SetPointA(pointA);
    }

    QDomElement pointBElement = pointAElement.nextSiblingElement();
    if (!pointBElement.isNull())
    {
        std::shared_ptr<KardanikXML::Point> pointB = parsePoint(pointBElement);
        line->SetPointB(pointB);
    }
    return line;
}

std::shared_ptr<Joint> ConstructionParser::parseJointElement(const QDomElement &jointElement)
{
    std::shared_ptr<Joint> joint(new Joint());

    string axisName = jointElement.attribute("axis", "Y_AXIS").toStdString();
    joint->SetAxis(axisName);

    if (jointElement.hasAttribute("lower_limit"))
    {
        joint->SetLowerLimit(jointElement.attribute("lower_limit").toFloat());
    }

    if (jointElement.hasAttribute("upper_limit"))
    {
        joint->SetUpperLimit(jointElement.attribute("upper_limit").toFloat());
    }

    if (jointElement.hasAttribute("initial_angle"))
    {
        joint->SetInitialAngle(jointElement.attribute("initial_angle").toFloat());
    }

    QDomElement bodyDescAElement = jointElement.firstChildElement("bodydesc");
    if (!bodyDescAElement.isNull())
    {
        std::shared_ptr<BodyJointDesc> bodyDescA = parseBodyDescElement(bodyDescAElement);
        joint->SetBodyJointDescA(bodyDescA);
    }

    QDomElement bodyDescBElement = bodyDescAElement.nextSiblingElement("bodydesc");
    if (!bodyDescBElement.isNull())
    {
        std::shared_ptr<BodyJointDesc> bodyDescB = parseBodyDescElement(bodyDescBElement);
        joint->SetBodyJointDescB(bodyDescB);
    }
    return joint;
}

std::shared_ptr<BodyJointDesc> ConstructionParser::parseBodyDescElement(const QDomElement &bodyDescElement)
{
    std::shared_ptr<BodyJointDesc> bodyDesc(new BodyJointDesc());
    QDomNode child = bodyDescElement.firstChild();
    while (!child.isNull())
    {
        if (child.toElement().tagName().toStdString() == "bodyref")
        {
            std::shared_ptr<Body> bodyref = parseBodyRefElement(child.toElement());
            bodyDesc->SetBody(bodyref);
        }
        else if (child.toElement().tagName().toStdString() == "pointref")
        {
            std::shared_ptr<Point> pointref = parsePointRefElement(child.toElement());
            bodyDesc->SetPointInBody(pointref);
        }
        child = child.nextSibling();
    }
    return bodyDesc;
}

bool ConstructionParser::idHasNamespace(const std::string &id) const
{
    return id.rfind(':') != std::string::npos;
}

void ConstructionParser::setBodyNamespaceLookupCallback(BodyNamespaceLookup namespaceLookup)
{
    m_BodyNamespaceLookup = namespaceLookup;
}

void ConstructionParser::setPointNamespaceLookupCallback(PointNamespaceLookup namespaceLookup)
{
    m_PointNamespaceLookup = namespaceLookup;
}

std::shared_ptr<Body> ConstructionParser::getBodyByNameLocal(const std::string &bodyID) const
{
    BodyMap::const_iterator foundBody = m_BodyMap.find(bodyID);
    if (foundBody == m_BodyMap.end())
    {
        std::cerr << "No body " << bodyID << " found in namespace " << m_Construction->GetNamespace() << " !" << std::endl;
        return std::shared_ptr<Body>();
    }
    else
    {
        return foundBody->second;
    }
}

std::shared_ptr<Point> ConstructionParser::getPointByNameLocal(const std::string &pointID) const
{
    PointMap::const_iterator foundPoint = m_PointMap.find(pointID);
    if (foundPoint == m_PointMap.end())
    {
        std::cerr << "No point " << pointID << " found in namespace " << m_Construction->GetNamespace() << " !" << std::endl;
        return std::shared_ptr<Point>();
    }
    else
    {
        return foundPoint->second;
    }
}

std::shared_ptr<Body> ConstructionParser::getBodyByNameGlobal(const std::string &bodyID) const
{
    if (idHasNamespace(bodyID))
    {
        std::tuple<std::string, std::string> seperatedID = splitID(bodyID);
        if (m_BodyNamespaceLookup)
        {
            return m_BodyNamespaceLookup(get<0>(seperatedID), get<1>(seperatedID));
        }
        else
        {
            return std::shared_ptr<Body>();
        }
    }
    else
    {
        return getBodyByNameLocal(bodyID);
    }
}

std::shared_ptr<Point> ConstructionParser::getPointByNameGlobal(const std::string &pointID) const
{
    if (idHasNamespace(pointID))
    {
        std::tuple<std::string, std::string> seperatedID = splitID(pointID);
        if (m_PointNamespaceLookup)
        {
            return m_PointNamespaceLookup(get<0>(seperatedID), get<1>(seperatedID));
        }
        else
        {
            return std::shared_ptr<Point>();
        }
    }
    else
    {
        return getPointByNameLocal(pointID);
    }
}

std::tuple<std::string, std::string> ConstructionParser::splitID(const std::string &id) const
{
    std::string::size_type colonPos = id.rfind(':');
    return std::make_tuple(id.substr(0, colonPos), id.substr(colonPos + 1, std::string::npos));
}

std::shared_ptr<Body> ConstructionParser::parseBodyRefElement(const QDomElement &bodyRefElement)
{
    string bodyName = bodyRefElement.attribute("id", "").toStdString();
    return getBodyByNameGlobal(bodyName);
}

std::shared_ptr<Point> ConstructionParser::parsePointRefElement(const QDomElement &pointRefElement)
{
    string pointName = pointRefElement.attribute("id", "").toStdString();
    return getPointByNameGlobal(pointName);
}

std::shared_ptr<LineStrip> ConstructionParser::parseLinestripElement(const QDomElement &linestripElement)
{
    std::shared_ptr<LineStrip> linestrip(new LineStrip());
    float radius = m_DefaultLineSize;
    if (linestripElement.hasAttribute("radius"))
    {
        radius = linestripElement.attribute("radius").toFloat();
    }

    linestrip->SetRadius(radius);
    QDomElement pointElement = linestripElement.firstChildElement();

    while (!pointElement.isNull())
    {
        std::shared_ptr<Point> point = parsePoint(pointElement);
        linestrip->AddPoint(point);
        pointElement = pointElement.nextSiblingElement();
    }
    return linestrip;
}

std::shared_ptr<Point> ConstructionParser::parsePoint(const QDomElement &pointElement)
{
    if (pointElement.isNull())
    {
        return std::shared_ptr<Point>();
    }
    if (pointElement.tagName() == "point")
    {
        return parsePointElement(pointElement);
    }
    else if (pointElement.tagName() == "point_rel")
    {
        return parsePointElementRelative(pointElement);
    }
    return std::shared_ptr<Point>();
}

std::shared_ptr<Point> ConstructionParser::parsePointElement(const QDomElement &pointElement)
{
    std::shared_ptr<Point> point(new Point());
    float x = pointElement.attribute("x", "0.0").toFloat();
    point->SetX(x);
    float y = pointElement.attribute("y", "0.0").toFloat();
    point->SetY(y);
    float z = pointElement.attribute("z", "0.0").toFloat();
    point->SetZ(z);

    string pointName = getOrCreatePointName(pointElement);
    point->SetName(pointName);

    m_PointMap.insert(PointMap::value_type(pointName, point));

    return point;
}

std::shared_ptr<PointRelative> ConstructionParser::parsePointElementRelative(const QDomElement &pointElement)
{
    std::shared_ptr<PointRelative> point(new PointRelative());
    float x = pointElement.attribute("xoffset", "0.0").toFloat();
    point->SetXOffset(x);
    float y = pointElement.attribute("yoffset", "0.0").toFloat();
    point->SetYOffset(y);
    float z = pointElement.attribute("zoffset", "0.0").toFloat();
    point->SetZOffset(z);

    if (pointElement.hasAttribute("point_id"))
    {
        std::string pointID = pointElement.attribute("point_id").toStdString();

        std::shared_ptr<Point> refPoint = getPointByNameGlobal(pointID);
        point->ResolveAgainst(refPoint);
    }

    string pointName = getOrCreatePointName(pointElement);
    point->SetName(pointName);

    m_PointMap.insert(PointMap::value_type(pointName, point));

    return point;
}

std::shared_ptr<Anchor> ConstructionParser::parseAnchorElement(const QDomElement &anchorElement, std::shared_ptr<Body> body)
{
    std::shared_ptr<Anchor> anchor(new Anchor());
    anchor->SetParentBody(body);
    string pointID = anchorElement.attribute("point_id", "").toStdString();
    string anchorNodeID = anchorElement.attribute("node_id", "").toStdString();

    anchor->SetAnchorNodeName(anchorNodeID);
    std::shared_ptr<Point> point = body->GetPointByName(pointID);

    anchor->SetAnchorPoint(point);

    m_AnchorNodeInfo.insert(AnchorNodeInfo::value_type(anchorNodeID, point));
    m_AnchorNodeNameToBodyMap.insert(AnchorNodeNameToBodyMap::value_type(anchorNodeID, body));

    return anchor;
}

std::shared_ptr<OperatingRange> ConstructionParser::parseOperatingRangeElement(const QDomElement &rangeElement, std::shared_ptr<Body> body)
{
    std::shared_ptr<OperatingRange> range(new OperatingRange());

    string pointID = rangeElement.attribute("point_id", "").toStdString();
    std::shared_ptr<Point> point = body->GetPointByName(pointID);
    range->SetCenterPoint(point);

    float radius = rangeElement.attribute("radius", "1.0").toFloat();
    range->SetRadius(radius);

    string axisName = rangeElement.attribute("axis", "Z_AXIS").toStdString();
    range->SetAxis(axisName);

    string colorString = rangeElement.attribute("color", "0.0 1.0 0.0 1.0").toStdString();
    range->SetColor(colorString);

    return range;
}

string ConstructionParser::getOrCreatePointName(const QDomElement &pointElement)
{
    string pointName = pointElement.attribute("name", "").toStdString();

    if (pointName.empty())
    {
        ostringstream outName;
        outName << "UnnamedPoint_" << setfill('0') << setw(4) << m_NumUnnamedPoints << ends;
        m_NumUnnamedPoints++;
        return outName.str();
    }
    else
    {
        return pointName;
    }
}

std::shared_ptr<Construction> ConstructionParser::parseConstructionElement(const QDomElement &constructionElement)
{
    if (!m_Construction)
    {
        m_Construction.reset(new Construction());
    }

    std::string theNamespace = constructionElement.attribute("namespace", "").toStdString();
    m_Construction->SetNamespace(theNamespace);

    QDomNode child = constructionElement.firstChild();
    while (!child.isNull())
    {
        if (child.toElement().tagName() == "body")
        {
            std::shared_ptr<Body> body = parseBodyElement(child.toElement());
            if (body)
            {
                m_Construction->AddBody(body);
            }
        }
        else if (child.toElement().tagName() == "joint")
        {
            std::shared_ptr<Joint> joint = parseJointElement(child.toElement());
            if (joint)
            {
                m_Construction->AddJoint(joint);
            }
        }
        child = child.nextSibling();
    }
    return m_Construction;
}

bool ConstructionParser::hasAnchorNodeWithName(const std::string &anchorNodeName) const
{
    AnchorNodeInfo::const_iterator foundAnchorNode = m_AnchorNodeInfo.find(anchorNodeName);
    if (foundAnchorNode != m_AnchorNodeInfo.end())
    {
        return true;
    }
    else
    {
        return false;
    }
}

std::shared_ptr<Body> ConstructionParser::getBodyForAnchor(const std::string &anchorNodeName) const
{
    AnchorNodeNameToBodyMap::const_iterator foundAnchorNode = m_AnchorNodeNameToBodyMap.find(anchorNodeName);
    if (foundAnchorNode != m_AnchorNodeNameToBodyMap.end())
    {
        return foundAnchorNode->second;
    }
    else
    {
        return std::shared_ptr<Body>();
    }
}

std::shared_ptr<Construction> ConstructionParser::getConstruction() const
{
    return m_Construction;
}
}
