/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef RoadSystemVisitor_h
#define RoadSystemVisitor_h

#include <iostream>

#include <xercesc/dom/DOM.hpp>
#include <util/coExport.h>

class Tarmac;
class Road;
class Junction;
class Fiddleyard;
class PlaneStraightLine;
class PlaneArc;
class PlaneClothoid;
class PlanePolynom;
class Polynom;
class SuperelevationPolynom;
class CrossfallPolynom;
class LaneSection;
class Lane;
class RoadMark;
class PathConnection;

class VEHICLEUTILEXPORT RoadSystemVisitor
{
public:
    virtual ~RoadSystemVisitor()
    {
    }

    virtual void visit(Tarmac *);

    virtual void visit(Road *) = 0;
    virtual void visit(PlaneStraightLine *) = 0;
    virtual void visit(PlaneArc *) = 0;
    virtual void visit(PlaneClothoid *) = 0;
    virtual void visit(PlanePolynom *) = 0;
    virtual void visit(Polynom *) = 0;
    virtual void visit(SuperelevationPolynom *) = 0;
    virtual void visit(CrossfallPolynom *) = 0;
    virtual void visit(LaneSection *) = 0;
    virtual void visit(Lane *) = 0;
    virtual void visit(RoadMark *) = 0;

    virtual void visit(Junction *) = 0;
    virtual void visit(PathConnection *) = 0;

    virtual void visit(Fiddleyard *) = 0;

protected:
};

class VEHICLEUTILEXPORT XodrWriteRoadSystemVisitor : public RoadSystemVisitor
{
public:
    enum
    {
        ELEVATION,
        LANEWIDTH,
        BATTERWIDTH
    } polyType;

    XodrWriteRoadSystemVisitor();
    ~XodrWriteRoadSystemVisitor();

    void writeToFile(std::string);

    void visit(Road *);
    void visit(PlaneStraightLine *);
    void visit(PlaneArc *);
    void visit(PlaneClothoid *);
    void visit(PlanePolynom *);
    void visit(Polynom *);
    void visit(SuperelevationPolynom *);
    void visit(CrossfallPolynom *);
    void visit(LaneSection *);
    void visit(Lane *);
    void visit(RoadMark *);

    void visit(Junction *);
    void visit(PathConnection *);

    void visit(Fiddleyard *);

protected:
    xercesc::DOMImplementation *impl;
    xercesc::DOMDocument *document;
    xercesc::DOMElement *rootElement;
    xercesc::DOMElement *roadElement;
    xercesc::DOMElement *junctionElement;
    xercesc::DOMElement *fiddleyardElement;
    xercesc::DOMElement *geometryElement;
    xercesc::DOMElement *elevationProfileElement;
    xercesc::DOMElement *lateralProfileElement;
    xercesc::DOMElement *lanesElement;
    xercesc::DOMElement *sideElement;
    xercesc::DOMElement *laneElement;
};

#endif
