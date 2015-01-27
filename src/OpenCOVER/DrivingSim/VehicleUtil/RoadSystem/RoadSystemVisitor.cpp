/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "RoadSystemVisitor.h"

#include <xercesc/dom/DOMImplementation.hpp>
//#include <xercesc/dom/DOMWriter.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/util/XercesVersion.hpp>
#include <sstream>

#include "Road.h"
#include "Junction.h"
#include "Fiddleyard.h"
#include "Types.h"
#include "LaneSection.h"

void RoadSystemVisitor::visit(Tarmac *tarmac)
{
    std::cout << "Vising Tarmac: " << tarmac->getId() << std::endl;
}

XodrWriteRoadSystemVisitor::XodrWriteRoadSystemVisitor()
{
    impl = xercesc::DOMImplementation::getImplementation();
    document = impl->createDocument(0, xercesc::XMLString::transcode("OpenDRIVE"), 0);

    rootElement = document->getDocumentElement();

    roadElement = NULL;
    junctionElement = NULL;
    fiddleyardElement = NULL;
}

XodrWriteRoadSystemVisitor::~XodrWriteRoadSystemVisitor()
{
    delete document;
}

void XodrWriteRoadSystemVisitor::writeToFile(std::string filename)
{
#if (XERCES_VERSION_MAJOR < 3)
    xercesc::DOMWriter *writer = impl->createDOMWriter();
#else
    xercesc::DOMLSSerializer *writer = ((xercesc::DOMImplementationLS *)impl)->createLSSerializer();
    // set the format-pretty-print feature
    if (writer->getDomConfig()->canSetParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
        writer->getDomConfig()->setParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);
#endif

    xercesc::XMLFormatTarget *xmlTarget = new xercesc::LocalFileFormatTarget(filename.c_str());

#if (XERCES_VERSION_MAJOR < 3)
    if (!writer->writeNode(xmlTarget, *document))
    {
        std::cerr << "TrafficSimulation::writeXodr: Could not open file for writing!" << std::endl;
    }
#else
    xercesc::DOMLSOutput *output = ((xercesc::DOMImplementationLS *)impl)->createLSOutput();
    output->setByteStream(xmlTarget);

    if (!writer->write(document, output))
    {
        std::cerr << "TrafficSimulation::writeXodr: Could not open file for writing!" << std::endl;
    }

    delete output;
#endif

    delete xmlTarget;
    delete writer;
}

void XodrWriteRoadSystemVisitor::visit(Road *road)
{
    roadElement = document->createElement(xercesc::XMLString::transcode("road"));
    rootElement->appendChild(roadElement);
    roadElement->setAttribute(xercesc::XMLString::transcode("name"), xercesc::XMLString::transcode(road->getName().c_str()));
    std::ostringstream lengthStream;
    lengthStream << std::scientific << road->getLength();
    roadElement->setAttribute(xercesc::XMLString::transcode("length"), xercesc::XMLString::transcode(lengthStream.str().c_str()));
    roadElement->setAttribute(xercesc::XMLString::transcode("id"), xercesc::XMLString::transcode(road->getId().c_str()));
    Junction *junction = road->getJunction();
    std::string junctionId = (junction) ? junction->getId() : "-1";
    roadElement->setAttribute(xercesc::XMLString::transcode("junction"), xercesc::XMLString::transcode(junctionId.c_str()));

    xercesc::DOMElement *linkElement = document->createElement(xercesc::XMLString::transcode("link"));
    roadElement->appendChild(linkElement);
    TarmacConnection *conn = road->getPredecessorConnection();
    if (conn)
    {
        xercesc::DOMElement *predecessorElement = document->createElement(xercesc::XMLString::transcode("predecessor"));
        linkElement->appendChild(predecessorElement);
        Tarmac *tarmac = conn->getConnectingTarmac();
        predecessorElement->setAttribute(xercesc::XMLString::transcode("elementType"), xercesc::XMLString::transcode(tarmac->getTypeSpecifier().c_str()));
        std::string idString;
        FiddleRoad *fiddleroad = dynamic_cast<FiddleRoad *>(tarmac);
        if (fiddleroad)
        {
            idString = fiddleroad->getFiddleyard()->getId();
        }
        else
        {
            idString = tarmac->getId();
        }
        predecessorElement->setAttribute(xercesc::XMLString::transcode("elementId"), xercesc::XMLString::transcode(idString.c_str()));
        int direction = conn->getConnectingTarmacDirection();
        if (direction > 0)
        {
            predecessorElement->setAttribute(xercesc::XMLString::transcode("contactPoint"), xercesc::XMLString::transcode("start"));
        }
        else if (direction < 0)
        {
            predecessorElement->setAttribute(xercesc::XMLString::transcode("contactPoint"), xercesc::XMLString::transcode("end"));
        }
    }
    conn = road->getSuccessorConnection();
    if (conn)
    {
        xercesc::DOMElement *successorElement = document->createElement(xercesc::XMLString::transcode("successor"));
        linkElement->appendChild(successorElement);
        Tarmac *tarmac = conn->getConnectingTarmac();
        successorElement->setAttribute(xercesc::XMLString::transcode("elementType"), xercesc::XMLString::transcode(tarmac->getTypeSpecifier().c_str()));
        std::string idString;
        FiddleRoad *fiddleroad = dynamic_cast<FiddleRoad *>(tarmac);
        if (fiddleroad)
        {
            idString = fiddleroad->getFiddleyard()->getId();
        }
        else
        {
            idString = tarmac->getId();
        }
        successorElement->setAttribute(xercesc::XMLString::transcode("elementId"), xercesc::XMLString::transcode(idString.c_str()));
        int direction = conn->getConnectingTarmacDirection();
        if (direction > 0)
        {
            successorElement->setAttribute(xercesc::XMLString::transcode("contactPoint"), xercesc::XMLString::transcode("start"));
        }
        else if (direction < 0)
        {
            successorElement->setAttribute(xercesc::XMLString::transcode("contactPoint"), xercesc::XMLString::transcode("end"));
        }
    }

    xercesc::DOMElement *planViewElement = document->createElement(xercesc::XMLString::transcode("planView"));
    roadElement->appendChild(planViewElement);
    std::map<double, PlaneCurve *> planViewMap = road->getPlaneCurveMap();
    for (std::map<double, PlaneCurve *>::iterator mapIt = planViewMap.begin(); mapIt != planViewMap.end(); ++mapIt)
    {
        geometryElement = document->createElement(xercesc::XMLString::transcode("geometry"));
        planViewElement->appendChild(geometryElement);

        double geometryStart = mapIt->first;
        std::ostringstream sStream;
        sStream << std::scientific << geometryStart;
        Vector3D point = mapIt->second->getPoint(geometryStart);
        std::ostringstream xStream;
        xStream << std::scientific << point.x();
        std::ostringstream yStream;
        yStream << std::scientific << point.y();
        std::ostringstream hdgStream;
        hdgStream << std::scientific << point.z();
        double length = ((++mapIt) != planViewMap.end()) ? (mapIt->first - geometryStart) : (road->getLength() - geometryStart);
        std::ostringstream lengthStream;
        lengthStream << std::scientific << length;

        geometryElement->setAttribute(xercesc::XMLString::transcode("s"), xercesc::XMLString::transcode(sStream.str().c_str()));
        geometryElement->setAttribute(xercesc::XMLString::transcode("x"), xercesc::XMLString::transcode(xStream.str().c_str()));
        geometryElement->setAttribute(xercesc::XMLString::transcode("y"), xercesc::XMLString::transcode(yStream.str().c_str()));
        geometryElement->setAttribute(xercesc::XMLString::transcode("hdg"), xercesc::XMLString::transcode(hdgStream.str().c_str()));
        geometryElement->setAttribute(xercesc::XMLString::transcode("length"), xercesc::XMLString::transcode(lengthStream.str().c_str()));

        (--mapIt)->second->accept(this);
    }

    elevationProfileElement = document->createElement(xercesc::XMLString::transcode("elevationProfile"));
    roadElement->appendChild(elevationProfileElement);
    std::map<double, Polynom *> elevationMap = road->getElevationMap();
    polyType = ELEVATION;
    for (std::map<double, Polynom *>::iterator mapIt = elevationMap.begin(); mapIt != elevationMap.end(); ++mapIt)
    {
        mapIt->second->accept(this);
    }

    lateralProfileElement = document->createElement(xercesc::XMLString::transcode("lateralProfile"));
    roadElement->appendChild(lateralProfileElement);
    std::map<double, LateralProfile *> lateralMap = road->getLateralMap();
    for (std::map<double, LateralProfile *>::iterator mapIt = lateralMap.begin(); mapIt != lateralMap.end(); ++mapIt)
    {
        mapIt->second->accept(this);
    }

    lanesElement = document->createElement(xercesc::XMLString::transcode("lanes"));
    roadElement->appendChild(lanesElement);
    std::map<double, LaneSection *> laneSectionMap = road->getLaneSectionMap();
    for (std::map<double, LaneSection *>::iterator mapIt = laneSectionMap.begin(); mapIt != laneSectionMap.end(); ++mapIt)
    {
        mapIt->second->accept(this);
    }
}

void XodrWriteRoadSystemVisitor::visit(Junction *junction)
{
    junctionElement = document->createElement(xercesc::XMLString::transcode("junction"));
    rootElement->appendChild(junctionElement);

    junctionElement->setAttribute(xercesc::XMLString::transcode("name"), xercesc::XMLString::transcode(junction->getName().c_str()));
    junctionElement->setAttribute(xercesc::XMLString::transcode("id"), xercesc::XMLString::transcode(junction->getId().c_str()));

    std::map<Road *, PathConnectionSet> setMap = junction->getPathConnectionSetMap();
    for (std::map<Road *, PathConnectionSet>::iterator mapIt = setMap.begin(); mapIt != setMap.end(); ++mapIt)
    {
        PathConnectionSet set = mapIt->second;
        for (PathConnectionSet::iterator setIt = set.begin(); setIt != set.end(); ++setIt)
        {
            (*setIt)->accept(this);
        }
    }
}

void XodrWriteRoadSystemVisitor::visit(Fiddleyard *fiddleyard)
{
    fiddleyardElement = document->createElement(xercesc::XMLString::transcode("fiddleyard"));
    rootElement->appendChild(fiddleyardElement);

    fiddleyardElement->setAttribute(xercesc::XMLString::transcode("name"), xercesc::XMLString::transcode(fiddleyard->getName().c_str()));
    fiddleyardElement->setAttribute(xercesc::XMLString::transcode("id"), xercesc::XMLString::transcode(fiddleyard->getId().c_str()));

    TarmacConnection *conn = fiddleyard->getTarmacConnection();
    if (conn)
    {
        xercesc::DOMElement *linkElement = document->createElement(xercesc::XMLString::transcode("link"));
        fiddleyardElement->appendChild(linkElement);
        Tarmac *tarmac = conn->getConnectingTarmac();
        linkElement->setAttribute(xercesc::XMLString::transcode("elementType"), xercesc::XMLString::transcode(tarmac->getTypeSpecifier().c_str()));
        linkElement->setAttribute(xercesc::XMLString::transcode("elementId"), xercesc::XMLString::transcode(tarmac->getId().c_str()));
        int direction = conn->getConnectingTarmacDirection();
        if (direction > 0)
        {
            linkElement->setAttribute(xercesc::XMLString::transcode("contactPoint"), xercesc::XMLString::transcode("start"));
        }
        else if (direction < 0)
        {
            linkElement->setAttribute(xercesc::XMLString::transcode("contactPoint"), xercesc::XMLString::transcode("end"));
        }
    }

    std::map<int, VehicleSource *> sourceMap = fiddleyard->getVehicleSourceMap();
    for (std::map<int, VehicleSource *>::iterator mapIt = sourceMap.begin(); mapIt != sourceMap.end(); ++mapIt)
    {
        xercesc::DOMElement *sourceElement = document->createElement(xercesc::XMLString::transcode("source"));
        fiddleyardElement->appendChild(sourceElement);

        sourceElement->setAttribute(xercesc::XMLString::transcode("id"), xercesc::XMLString::transcode(mapIt->second->getId().c_str()));
        std::ostringstream laneId;
        laneId << mapIt->second->getLane();
        sourceElement->setAttribute(xercesc::XMLString::transcode("lane"), xercesc::XMLString::transcode(laneId.str().c_str()));
        std::ostringstream startId;
        startId << std::scientific << mapIt->second->getStartTime();
        sourceElement->setAttribute(xercesc::XMLString::transcode("startTime"), xercesc::XMLString::transcode(startId.str().c_str()));
        std::ostringstream repeatId;
        repeatId << std::scientific << mapIt->second->getRepeatTime();
        sourceElement->setAttribute(xercesc::XMLString::transcode("repeatTime"), xercesc::XMLString::transcode(repeatId.str().c_str()));
    }

    std::map<int, VehicleSink *> sinkMap = fiddleyard->getVehicleSinkMap();
    for (std::map<int, VehicleSink *>::iterator mapIt = sinkMap.begin(); mapIt != sinkMap.end(); ++mapIt)
    {
        xercesc::DOMElement *sinkElement = document->createElement(xercesc::XMLString::transcode("sink"));
        fiddleyardElement->appendChild(sinkElement);

        sinkElement->setAttribute(xercesc::XMLString::transcode("id"), xercesc::XMLString::transcode(mapIt->second->getId().c_str()));
        std::ostringstream laneId;
        laneId << mapIt->second->getLane();
        sinkElement->setAttribute(xercesc::XMLString::transcode("lane"), xercesc::XMLString::transcode(laneId.str().c_str()));
    }
}

void XodrWriteRoadSystemVisitor::visit(PlaneStraightLine *)
{
    xercesc::DOMElement *lineElement = document->createElement(xercesc::XMLString::transcode("line"));
    geometryElement->appendChild(lineElement);
}

void XodrWriteRoadSystemVisitor::visit(PlaneArc *arc)
{
    xercesc::DOMElement *arcElement = document->createElement(xercesc::XMLString::transcode("arc"));
    geometryElement->appendChild(arcElement);

    std::ostringstream curvStream;
    curvStream << std::scientific << arc->getCurvature(arc->getStart());
    arcElement->setAttribute(xercesc::XMLString::transcode("curvature"), xercesc::XMLString::transcode(curvStream.str().c_str()));
}

void XodrWriteRoadSystemVisitor::visit(PlaneClothoid *cloth)
{
    xercesc::DOMElement *spiralElement = document->createElement(xercesc::XMLString::transcode("spiral"));
    geometryElement->appendChild(spiralElement);

    std::ostringstream curvStartStream;
    curvStartStream << std::scientific << cloth->getCurvature(cloth->getStart());
    std::ostringstream curvEndStream;
    curvEndStream << std::scientific << cloth->getCurvature(cloth->getLength() + cloth->getStart());
    spiralElement->setAttribute(xercesc::XMLString::transcode("curvStart"), xercesc::XMLString::transcode(curvStartStream.str().c_str()));
    spiralElement->setAttribute(xercesc::XMLString::transcode("curvEnd"), xercesc::XMLString::transcode(curvEndStream.str().c_str()));
}

void XodrWriteRoadSystemVisitor::visit(PlanePolynom *poly)
{
    xercesc::DOMElement *poly3Element = document->createElement(xercesc::XMLString::transcode("poly3"));
    geometryElement->appendChild(poly3Element);

    double a, b, c, d;
    poly->getCoefficients(a, b, c, d);

    std::ostringstream aStream;
    aStream << std::scientific << a;
    std::ostringstream bStream;
    bStream << std::scientific << b;
    std::ostringstream cStream;
    cStream << std::scientific << c;
    std::ostringstream dStream;
    dStream << std::scientific << d;

    poly3Element->setAttribute(xercesc::XMLString::transcode("a"), xercesc::XMLString::transcode(aStream.str().c_str()));
    poly3Element->setAttribute(xercesc::XMLString::transcode("b"), xercesc::XMLString::transcode(bStream.str().c_str()));
    poly3Element->setAttribute(xercesc::XMLString::transcode("c"), xercesc::XMLString::transcode(cStream.str().c_str()));
    poly3Element->setAttribute(xercesc::XMLString::transcode("d"), xercesc::XMLString::transcode(dStream.str().c_str()));
}

void XodrWriteRoadSystemVisitor::visit(Polynom *poly)
{
    xercesc::DOMElement *polyElement;
    if (polyType == ELEVATION)
    {
        polyElement = document->createElement(xercesc::XMLString::transcode("elevation"));
        elevationProfileElement->appendChild(polyElement);
    }
    else if (polyType == LANEWIDTH)
    {
        polyElement = document->createElement(xercesc::XMLString::transcode("width"));
        laneElement->appendChild(polyElement);
    }

    else
    {
        std::cerr << "XodrWriteRoadSystemVisitor::visit(Polynom*): Unkown polynom type -> canceling visit..." << std::endl;
        return;
    }

    double a, b, c, d;
    poly->getCoefficients(a, b, c, d);

    std::ostringstream sStream;
    sStream << std::scientific << poly->getStart();
    std::ostringstream aStream;
    aStream << std::scientific << a;
    std::ostringstream bStream;
    bStream << std::scientific << b;
    std::ostringstream cStream;
    cStream << std::scientific << c;
    std::ostringstream dStream;
    dStream << std::scientific << d;

    if (polyType == ELEVATION)
    {
        polyElement->setAttribute(xercesc::XMLString::transcode("s"), xercesc::XMLString::transcode(sStream.str().c_str()));
    }
    else if (polyType == LANEWIDTH)
    {
        polyElement->setAttribute(xercesc::XMLString::transcode("sOffset"), xercesc::XMLString::transcode(sStream.str().c_str()));
    }

    polyElement->setAttribute(xercesc::XMLString::transcode("a"), xercesc::XMLString::transcode(aStream.str().c_str()));
    polyElement->setAttribute(xercesc::XMLString::transcode("b"), xercesc::XMLString::transcode(bStream.str().c_str()));
    polyElement->setAttribute(xercesc::XMLString::transcode("c"), xercesc::XMLString::transcode(cStream.str().c_str()));
    polyElement->setAttribute(xercesc::XMLString::transcode("d"), xercesc::XMLString::transcode(dStream.str().c_str()));
}

void XodrWriteRoadSystemVisitor::visit(SuperelevationPolynom *elev)
{
    xercesc::DOMElement *superelevationElement = document->createElement(xercesc::XMLString::transcode("superelevation"));
    lateralProfileElement->appendChild(superelevationElement);

    double a, b, c, d;
    elev->getCoefficients(a, b, c, d);

    std::ostringstream sStream;
    sStream << std::scientific << elev->getStart();
    std::ostringstream aStream;
    aStream << std::scientific << a;
    std::ostringstream bStream;
    bStream << std::scientific << b;
    std::ostringstream cStream;
    cStream << std::scientific << c;
    std::ostringstream dStream;
    dStream << std::scientific << d;

    superelevationElement->setAttribute(xercesc::XMLString::transcode("s"), xercesc::XMLString::transcode(sStream.str().c_str()));
    superelevationElement->setAttribute(xercesc::XMLString::transcode("a"), xercesc::XMLString::transcode(aStream.str().c_str()));
    superelevationElement->setAttribute(xercesc::XMLString::transcode("b"), xercesc::XMLString::transcode(bStream.str().c_str()));
    superelevationElement->setAttribute(xercesc::XMLString::transcode("c"), xercesc::XMLString::transcode(cStream.str().c_str()));
    superelevationElement->setAttribute(xercesc::XMLString::transcode("d"), xercesc::XMLString::transcode(dStream.str().c_str()));
}

void XodrWriteRoadSystemVisitor::visit(CrossfallPolynom *fall)
{
    xercesc::DOMElement *crossfallElement = document->createElement(xercesc::XMLString::transcode("crossfall"));
    lateralProfileElement->appendChild(crossfallElement);

    double a, b, c, d;
    fall->getCoefficients(a, b, c, d);

    std::ostringstream sStream;
    sStream << std::scientific << fall->getStart();
    std::ostringstream aStream;
    aStream << std::scientific << a;
    std::ostringstream bStream;
    bStream << std::scientific << b;
    std::ostringstream cStream;
    cStream << std::scientific << c;
    std::ostringstream dStream;
    dStream << std::scientific << d;
    std::string side;
    if (fall->getLeftFallFactor() < 0 && fall->getRightFallFactor() > 0)
    {
        side = "both";
    }
    else
    {
        if (fall->getLeftFallFactor() < 0)
        {
            side = "left";
        }
        else if (fall->getRightFallFactor() > 0)
        {
            side = "right";
        }
        else
        {
            side = "both";
        }
    }

    crossfallElement->setAttribute(xercesc::XMLString::transcode("s"), xercesc::XMLString::transcode(sStream.str().c_str()));
    crossfallElement->setAttribute(xercesc::XMLString::transcode("a"), xercesc::XMLString::transcode(aStream.str().c_str()));
    crossfallElement->setAttribute(xercesc::XMLString::transcode("b"), xercesc::XMLString::transcode(bStream.str().c_str()));
    crossfallElement->setAttribute(xercesc::XMLString::transcode("c"), xercesc::XMLString::transcode(cStream.str().c_str()));
    crossfallElement->setAttribute(xercesc::XMLString::transcode("d"), xercesc::XMLString::transcode(dStream.str().c_str()));
    crossfallElement->setAttribute(xercesc::XMLString::transcode("side"), xercesc::XMLString::transcode(side.c_str()));
}

void XodrWriteRoadSystemVisitor::visit(LaneSection *section)
{
    xercesc::DOMElement *laneSectionElement = document->createElement(xercesc::XMLString::transcode("laneSection"));
    lanesElement->appendChild(laneSectionElement);

    std::ostringstream sStream;
    sStream << std::scientific << section->getStart();
    laneSectionElement->setAttribute(xercesc::XMLString::transcode("s"), xercesc::XMLString::transcode(sStream.str().c_str()));

    xercesc::DOMElement *leftElement = document->createElement(xercesc::XMLString::transcode("left"));
    xercesc::DOMElement *centerElement = document->createElement(xercesc::XMLString::transcode("center"));
    xercesc::DOMElement *rightElement = document->createElement(xercesc::XMLString::transcode("right"));
    sideElement = leftElement;
    laneSectionElement->appendChild(leftElement);
    laneSectionElement->appendChild(centerElement);
    laneSectionElement->appendChild(rightElement);
    std::map<int, Lane *> laneMap = section->getLaneMap();
    for (std::map<int, Lane *>::iterator laneIt = laneMap.end(); laneIt != laneMap.begin();)
    {
        --laneIt;
        if (laneIt->first > 0)
        {
            sideElement = leftElement;
        }
        else if (laneIt->first == 0)
        {
            sideElement = centerElement;
        }
        else if (laneIt->first < 0)
        {
            sideElement = rightElement;
        }
        laneIt->second->accept(this);
    }
}

void XodrWriteRoadSystemVisitor::visit(Lane *lane)
{
    laneElement = document->createElement(xercesc::XMLString::transcode("lane"));
    sideElement->appendChild(laneElement);
    std::ostringstream idStream;
    idStream << lane->getId();
    std::string levelString = (lane->isOnLevel()) ? "true" : "false";
    laneElement->setAttribute(xercesc::XMLString::transcode("id"), xercesc::XMLString::transcode(idStream.str().c_str()));
    laneElement->setAttribute(xercesc::XMLString::transcode("type"), xercesc::XMLString::transcode(lane->getLaneTypeString().c_str()));
    laneElement->setAttribute(xercesc::XMLString::transcode("level"), xercesc::XMLString::transcode(levelString.c_str()));

    xercesc::DOMElement *linkElement = document->createElement(xercesc::XMLString::transcode("link"));
    laneElement->appendChild(linkElement);

    int predId = lane->getPredecessor();
    if (predId != Lane::NOLANE)
    {
        xercesc::DOMElement *predecessorElement = document->createElement(xercesc::XMLString::transcode("predecessor"));
        linkElement->appendChild(predecessorElement);
        std::ostringstream predecessorStream;
        predecessorStream << predId;
        predecessorElement->setAttribute(xercesc::XMLString::transcode("id"), xercesc::XMLString::transcode(predecessorStream.str().c_str()));
    }

    int succId = lane->getSuccessor();
    if (succId != Lane::NOLANE)
    {
        xercesc::DOMElement *successorElement = document->createElement(xercesc::XMLString::transcode("successor"));
        linkElement->appendChild(successorElement);
        std::ostringstream successorStream;
        successorStream << succId;
        successorElement->setAttribute(xercesc::XMLString::transcode("id"), xercesc::XMLString::transcode(successorStream.str().c_str()));
    }

    std::map<double, Polynom *> widthMap = lane->getWidthMap();
    polyType = LANEWIDTH;
    for (std::map<double, Polynom *>::iterator mapIt = widthMap.begin(); mapIt != widthMap.end(); ++mapIt)
    {
        mapIt->second->accept(this);
    }

    std::map<double, RoadMark *> roadMarkMap = lane->getRoadMarkMap();
    for (std::map<double, RoadMark *>::iterator mapIt = roadMarkMap.begin(); mapIt != roadMarkMap.end(); ++mapIt)
    {
        mapIt->second->accept(this);
    }
}

void XodrWriteRoadSystemVisitor::visit(RoadMark *mark)
{
    xercesc::DOMElement *roadMarkElement = document->createElement(xercesc::XMLString::transcode("roadMark"));
    laneElement->appendChild(roadMarkElement);

    std::ostringstream startStream;
    startStream << std::scientific << mark->getStart();
    std::ostringstream widthStream;
    widthStream << std::scientific << mark->getWidth();
    roadMarkElement->setAttribute(xercesc::XMLString::transcode("sOffset"), xercesc::XMLString::transcode(startStream.str().c_str()));
    roadMarkElement->setAttribute(xercesc::XMLString::transcode("type"), xercesc::XMLString::transcode(mark->getTypeString().c_str()));
    roadMarkElement->setAttribute(xercesc::XMLString::transcode("weight"), xercesc::XMLString::transcode(mark->getWeightString().c_str()));
    roadMarkElement->setAttribute(xercesc::XMLString::transcode("color"), xercesc::XMLString::transcode(mark->getColorString().c_str()));
    roadMarkElement->setAttribute(xercesc::XMLString::transcode("width"), xercesc::XMLString::transcode(widthStream.str().c_str()));
    roadMarkElement->setAttribute(xercesc::XMLString::transcode("laneChange"), xercesc::XMLString::transcode(mark->getLaneChangeString().c_str()));
}

void XodrWriteRoadSystemVisitor::visit(PathConnection *conn)
{
    xercesc::DOMElement *connectionElement = document->createElement(xercesc::XMLString::transcode("connection"));
    junctionElement->appendChild(connectionElement);

    connectionElement->setAttribute(xercesc::XMLString::transcode("id"), xercesc::XMLString::transcode(conn->getId().c_str()));
    connectionElement->setAttribute(xercesc::XMLString::transcode("incomingRoad"), xercesc::XMLString::transcode(conn->getIncomingRoad()->getId().c_str()));
    connectionElement->setAttribute(xercesc::XMLString::transcode("connectingRoad"), xercesc::XMLString::transcode(conn->getConnectingPath()->getId().c_str()));
    std::string dirString = (conn->getConnectingPathDirection() < 0) ? "end" : "start";
    connectionElement->setAttribute(xercesc::XMLString::transcode("contactPoint"), xercesc::XMLString::transcode(dirString.c_str()));

    LaneConnectionMap laneConnMap = conn->getLaneConnectionMap();
    for (LaneConnectionMap::iterator mapIt = laneConnMap.begin(); mapIt != laneConnMap.end(); ++mapIt)
    {
        xercesc::DOMElement *laneLinkElement = document->createElement(xercesc::XMLString::transcode("laneLink"));
        connectionElement->appendChild(laneLinkElement);

        std::ostringstream from;
        from << mapIt->first;
        std::ostringstream to;
        to << mapIt->second;

        laneLinkElement->setAttribute(xercesc::XMLString::transcode("from"), xercesc::XMLString::transcode(from.str().c_str()));
        laneLinkElement->setAttribute(xercesc::XMLString::transcode("to"), xercesc::XMLString::transcode(to.str().c_str()));
    }
}
