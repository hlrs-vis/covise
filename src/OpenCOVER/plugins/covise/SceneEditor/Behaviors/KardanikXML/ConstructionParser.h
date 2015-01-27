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
#ifndef WIN32
#include <boost/tr1/memory.hpp>
#include <boost/tr1/functional.hpp>
#include <boost/tr1/tuple.hpp>
#endif
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

    std::tr1::shared_ptr<Construction> parseConstructionElement(const QDomElement &constructionElement);
    bool hasAnchorNodeWithName(const std::string &anchorNodeName) const;
    std::tr1::shared_ptr<Body> getBodyForAnchor(const std::string &anchorNodeName) const;
    std::tr1::shared_ptr<Construction> getConstruction() const;

    typedef std::tr1::function<std::tr1::shared_ptr<Body>(std::string theNamespace, std::string theID)> BodyNamespaceLookup;
    void setBodyNamespaceLookupCallback(BodyNamespaceLookup namespaceLookup);

    typedef std::tr1::function<std::tr1::shared_ptr<Point>(std::string theNamespace, std::string theID)> PointNamespaceLookup;
    void setPointNamespaceLookupCallback(PointNamespaceLookup namespaceLookup);

    std::tr1::shared_ptr<Body> getBodyByNameLocal(const std::string &bodyID) const;
    std::tr1::shared_ptr<Point> getPointByNameLocal(const std::string &pointID) const;

    std::tr1::shared_ptr<Body> getBodyByNameGlobal(const std::string &bodyID) const;
    std::tr1::shared_ptr<Point> getPointByNameGlobal(const std::string &pointID) const;

private:
    std::tr1::shared_ptr<Body> parseBodyElement(const QDomElement &bodyElement);
    std::string getOrCreateBodyName(const QDomElement &bodyElement);
    std::tr1::shared_ptr<Line> parseLineElement(const QDomElement &lineElement);
    std::tr1::shared_ptr<LineStrip> parseLinestripElement(const QDomElement &linestripElement);
    std::string getOrCreatePointName(const QDomElement &pointElement);
    std::tr1::shared_ptr<Point> parsePoint(const QDomElement &pointElement);
    std::tr1::shared_ptr<Point> parsePointElement(const QDomElement &pointElement);
    std::tr1::shared_ptr<PointRelative> parsePointElementRelative(const QDomElement &pointElement);
    std::tr1::shared_ptr<Joint> parseJointElement(const QDomElement &jointElement);
    std::tr1::shared_ptr<Body> parseBodyRefElement(const QDomElement &bodyRefElement);
    std::tr1::shared_ptr<Point> parsePointRefElement(const QDomElement &pointRefElement);
    std::tr1::shared_ptr<BodyJointDesc> parseBodyDescElement(const QDomElement &bodyDescElement);
    std::tr1::shared_ptr<Anchor> parseAnchorElement(const QDomElement &anchorElement, std::tr1::shared_ptr<Body> body);
    std::tr1::shared_ptr<OperatingRange> parseOperatingRangeElement(const QDomElement &rangeElement, std::tr1::shared_ptr<Body> body);

    bool idHasNamespace(const std::string &id) const;
    std::tr1::tuple<std::string, std::string> splitID(const std::string &id) const;

    typedef std::map<std::string, std::tr1::shared_ptr<Body> > BodyMap;
    BodyMap m_BodyMap;

    typedef std::map<std::string, std::tr1::shared_ptr<Point> > PointMap;
    PointMap m_PointMap;

    typedef std::map<std::string, std::tr1::shared_ptr<Point> > AnchorNodeInfo;
    AnchorNodeInfo m_AnchorNodeInfo;

    typedef std::map<std::string, std::tr1::shared_ptr<Body> > AnchorNodeNameToBodyMap;
    AnchorNodeNameToBodyMap m_AnchorNodeNameToBodyMap;

    std::tr1::shared_ptr<Construction> m_Construction;

    unsigned int m_NumUnnamedBodies;
    unsigned int m_NumUnnamedPoints;
    float m_DefaultLineSize;

    BodyNamespaceLookup m_BodyNamespaceLookup;

    PointNamespaceLookup m_PointNamespaceLookup;
};
}

#endif /* CONSTRUCTIONPARSER_H_ */
