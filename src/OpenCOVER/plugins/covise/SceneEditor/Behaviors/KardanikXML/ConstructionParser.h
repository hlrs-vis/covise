/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * ConstructionParser.h
 *
 *  Created on: Feb 14, 2012
 *      Author: jw_te
 */

#ifndef CONSTRUCTIONPARSER_H_
#define CONSTRUCTIONPARSER_H_

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <tuple>
#include <QtXml/qdom.h>

namespace KardanikXML
{

class Body;
class Joint;
class Line;
class LineStrip;
class Point;
class PointRelative;
class BodyJointDesc;
class Anchor;
class Construction;
class OperatingRange;

class ConstructionParser
{

public:
    ConstructionParser();

    std::shared_ptr<Construction> parseConstructionElement(const QDomElement &constructionElement);
    bool hasAnchorNodeWithName(const std::string &anchorNodeName) const;
    std::shared_ptr<Body> getBodyForAnchor(const std::string &anchorNodeName) const;
    std::shared_ptr<Construction> getConstruction() const;

    typedef std::function<std::shared_ptr<Body>(std::string theNamespace, std::string theID)> BodyNamespaceLookup;
    void setBodyNamespaceLookupCallback(BodyNamespaceLookup namespaceLookup);

    typedef std::function<std::shared_ptr<Point>(std::string theNamespace, std::string theID)> PointNamespaceLookup;
    void setPointNamespaceLookupCallback(PointNamespaceLookup namespaceLookup);

    std::shared_ptr<Body> getBodyByNameLocal(const std::string &bodyID) const;
    std::shared_ptr<Point> getPointByNameLocal(const std::string &pointID) const;

    std::shared_ptr<Body> getBodyByNameGlobal(const std::string &bodyID) const;
    std::shared_ptr<Point> getPointByNameGlobal(const std::string &pointID) const;

private:
    std::shared_ptr<Body> parseBodyElement(const QDomElement &bodyElement);
    std::string getOrCreateBodyName(const QDomElement &bodyElement);
    std::shared_ptr<Line> parseLineElement(const QDomElement &lineElement);
    std::shared_ptr<LineStrip> parseLinestripElement(const QDomElement &linestripElement);
    std::string getOrCreatePointName(const QDomElement &pointElement);
    std::shared_ptr<Point> parsePoint(const QDomElement &pointElement);
    std::shared_ptr<Point> parsePointElement(const QDomElement &pointElement);
    std::shared_ptr<PointRelative> parsePointElementRelative(const QDomElement &pointElement);
    std::shared_ptr<Joint> parseJointElement(const QDomElement &jointElement);
    std::shared_ptr<Body> parseBodyRefElement(const QDomElement &bodyRefElement);
    std::shared_ptr<Point> parsePointRefElement(const QDomElement &pointRefElement);
    std::shared_ptr<BodyJointDesc> parseBodyDescElement(const QDomElement &bodyDescElement);
    std::shared_ptr<Anchor> parseAnchorElement(const QDomElement &anchorElement, std::shared_ptr<Body> body);
    std::shared_ptr<OperatingRange> parseOperatingRangeElement(const QDomElement &rangeElement, std::shared_ptr<Body> body);

    bool idHasNamespace(const std::string &id) const;
    std::tuple<std::string, std::string> splitID(const std::string &id) const;

    typedef std::map<std::string, std::shared_ptr<Body> > BodyMap;
    BodyMap m_BodyMap;

    typedef std::map<std::string, std::shared_ptr<Point> > PointMap;
    PointMap m_PointMap;

    typedef std::map<std::string, std::shared_ptr<Point> > AnchorNodeInfo;
    AnchorNodeInfo m_AnchorNodeInfo;

    typedef std::map<std::string, std::shared_ptr<Body> > AnchorNodeNameToBodyMap;
    AnchorNodeNameToBodyMap m_AnchorNodeNameToBodyMap;

    std::shared_ptr<Construction> m_Construction;

    unsigned int m_NumUnnamedBodies;
    unsigned int m_NumUnnamedPoints;
    float m_DefaultLineSize;

    BodyNamespaceLookup m_BodyNamespaceLookup;

    PointNamespaceLookup m_PointNamespaceLookup;
};
}

#endif /* CONSTRUCTIONPARSER_H_ */
