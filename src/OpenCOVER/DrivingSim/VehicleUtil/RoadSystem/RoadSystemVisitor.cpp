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
	XMLCh *t1 = NULL;
    impl = xercesc::DOMImplementation::getImplementation();
    document = impl->createDocument(0, t1 = xercesc::XMLString::transcode("OpenDRIVE"), 0); xercesc::XMLString::release(&t1);

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
	XMLCh *t1 = NULL;
	XMLCh *t2 = NULL;
    roadElement = document->createElement(t1 = xercesc::XMLString::transcode("road")); xercesc::XMLString::release(&t1);
    rootElement->appendChild(roadElement);
    roadElement->setAttribute(t1 = xercesc::XMLString::transcode("name"), t2 = xercesc::XMLString::transcode(road->getName().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    std::ostringstream lengthStream;
    lengthStream << std::scientific << road->getLength();
    roadElement->setAttribute(t1 = xercesc::XMLString::transcode("length"), t2 = xercesc::XMLString::transcode(lengthStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    roadElement->setAttribute(t1 = xercesc::XMLString::transcode("id"), t2 = xercesc::XMLString::transcode(road->getId().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    Junction *junction = road->getJunction();
    std::string junctionId = (junction) ? junction->getId() : "-1";
    roadElement->setAttribute(t1 = xercesc::XMLString::transcode("junction"), t2 = xercesc::XMLString::transcode(junctionId.c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);

    xercesc::DOMElement *linkElement = document->createElement(t1 = xercesc::XMLString::transcode("link")); xercesc::XMLString::release(&t1);
    roadElement->appendChild(linkElement);
    TarmacConnection *conn = road->getPredecessorConnection();
    if (conn)
    {
        xercesc::DOMElement *predecessorElement = document->createElement(t1 = xercesc::XMLString::transcode("predecessor"));  xercesc::XMLString::release(&t1);
        linkElement->appendChild(predecessorElement);
        Tarmac *tarmac = conn->getConnectingTarmac();
        predecessorElement->setAttribute(t1 = xercesc::XMLString::transcode("elementType"), t2 = xercesc::XMLString::transcode(tarmac->getTypeSpecifier().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
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
        predecessorElement->setAttribute(t1 = xercesc::XMLString::transcode("elementId"), t2 = xercesc::XMLString::transcode(idString.c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
        int direction = conn->getConnectingTarmacDirection();
        if (direction > 0)
        {
            predecessorElement->setAttribute(t1 = xercesc::XMLString::transcode("contactPoint"), t2 = xercesc::XMLString::transcode("start"));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
        }
        else if (direction < 0)
        {
            predecessorElement->setAttribute(t1 = xercesc::XMLString::transcode("contactPoint"), t2 = xercesc::XMLString::transcode("end"));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
        }
    }
    conn = road->getSuccessorConnection();
    if (conn)
    {
        xercesc::DOMElement *successorElement = document->createElement(t1 = xercesc::XMLString::transcode("successor")); xercesc::XMLString::release(&t1);
        linkElement->appendChild(successorElement);
        Tarmac *tarmac = conn->getConnectingTarmac();
        successorElement->setAttribute(t1 = xercesc::XMLString::transcode("elementType"), t2 = xercesc::XMLString::transcode(tarmac->getTypeSpecifier().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
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
        successorElement->setAttribute(t1 = xercesc::XMLString::transcode("elementId"), t2 = xercesc::XMLString::transcode(idString.c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
        int direction = conn->getConnectingTarmacDirection();
        if (direction > 0)
        {
            successorElement->setAttribute(t1 = xercesc::XMLString::transcode("contactPoint"), t2 = xercesc::XMLString::transcode("start"));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
        }
        else if (direction < 0)
        {
            successorElement->setAttribute(t1 = xercesc::XMLString::transcode("contactPoint"), t2 = xercesc::XMLString::transcode("end"));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
        }
    }

    xercesc::DOMElement *planViewElement = document->createElement(t1 = xercesc::XMLString::transcode("planView")); xercesc::XMLString::release(&t1);
    roadElement->appendChild(planViewElement);
    std::map<double, PlaneCurve *> planViewMap = road->getPlaneCurveMap();
    for (std::map<double, PlaneCurve *>::iterator mapIt = planViewMap.begin(); mapIt != planViewMap.end(); ++mapIt)
    {
        geometryElement = document->createElement(t1 = xercesc::XMLString::transcode("geometry")); xercesc::XMLString::release(&t1);
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

        geometryElement->setAttribute(t1 = xercesc::XMLString::transcode("s"), t2 = xercesc::XMLString::transcode(sStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
        geometryElement->setAttribute(t1 = xercesc::XMLString::transcode("x"), t2 = xercesc::XMLString::transcode(xStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
        geometryElement->setAttribute(t1 = xercesc::XMLString::transcode("y"), t2 = xercesc::XMLString::transcode(yStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
        geometryElement->setAttribute(t1 = xercesc::XMLString::transcode("hdg"), t2 = xercesc::XMLString::transcode(hdgStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
        geometryElement->setAttribute(t1 = xercesc::XMLString::transcode("length"), t2 = xercesc::XMLString::transcode(lengthStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);

        (--mapIt)->second->accept(this);
    }

    elevationProfileElement = document->createElement(t1 = xercesc::XMLString::transcode("elevationProfile")); xercesc::XMLString::release(&t1);
    roadElement->appendChild(elevationProfileElement);
    std::map<double, Polynom *> elevationMap = road->getElevationMap();
    polyType = ELEVATION;
    for (std::map<double, Polynom *>::iterator mapIt = elevationMap.begin(); mapIt != elevationMap.end(); ++mapIt)
    {
        mapIt->second->accept(this);
    }

    lateralProfileElement = document->createElement(t1 = xercesc::XMLString::transcode("lateralProfile")); xercesc::XMLString::release(&t1);
    roadElement->appendChild(lateralProfileElement);
    std::map<double, LateralProfile *> lateralMap = road->getLateralMap();
    for (std::map<double, LateralProfile *>::iterator mapIt = lateralMap.begin(); mapIt != lateralMap.end(); ++mapIt)
    {
        mapIt->second->accept(this);
    }

    lanesElement = document->createElement(t1 = xercesc::XMLString::transcode("lanes")); xercesc::XMLString::release(&t1);
    roadElement->appendChild(lanesElement);
    std::map<double, LaneSection *> laneSectionMap = road->getLaneSectionMap();
    for (std::map<double, LaneSection *>::iterator mapIt = laneSectionMap.begin(); mapIt != laneSectionMap.end(); ++mapIt)
    {
        mapIt->second->accept(this);
    }
}

void XodrWriteRoadSystemVisitor::visit(Junction *junction)
{
	XMLCh *t1 = NULL;
	XMLCh *t2 = NULL;
    junctionElement = document->createElement(t1 = xercesc::XMLString::transcode("junction")); xercesc::XMLString::release(&t1);
    rootElement->appendChild(junctionElement);

    junctionElement->setAttribute(t1 = xercesc::XMLString::transcode("name"), t2 = xercesc::XMLString::transcode(junction->getName().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    junctionElement->setAttribute(t1 = xercesc::XMLString::transcode("id"), t2 = xercesc::XMLString::transcode(junction->getId().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);

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
	XMLCh *t1 = NULL;
	XMLCh *t2 = NULL;
    fiddleyardElement = document->createElement(t1 = xercesc::XMLString::transcode("fiddleyard")); xercesc::XMLString::release(&t1);
    rootElement->appendChild(fiddleyardElement);

    fiddleyardElement->setAttribute(t1 = xercesc::XMLString::transcode("name"), t2 = xercesc::XMLString::transcode(fiddleyard->getName().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    fiddleyardElement->setAttribute(t1 = xercesc::XMLString::transcode("id"), t2 = xercesc::XMLString::transcode(fiddleyard->getId().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);

    TarmacConnection *conn = fiddleyard->getTarmacConnection();
    if (conn)
    {
        xercesc::DOMElement *linkElement = document->createElement(t1 = xercesc::XMLString::transcode("link")); xercesc::XMLString::release(&t1);
        fiddleyardElement->appendChild(linkElement);
        Tarmac *tarmac = conn->getConnectingTarmac();
        linkElement->setAttribute(t1 = xercesc::XMLString::transcode("elementType"), t2 = xercesc::XMLString::transcode(tarmac->getTypeSpecifier().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
        linkElement->setAttribute(t1 = xercesc::XMLString::transcode("elementId"), t2 = xercesc::XMLString::transcode(tarmac->getId().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
        int direction = conn->getConnectingTarmacDirection();
        if (direction > 0)
        {
            linkElement->setAttribute(t1 = xercesc::XMLString::transcode("contactPoint"), t2 = xercesc::XMLString::transcode("start"));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
        }
        else if (direction < 0)
        {
            linkElement->setAttribute(t1 = xercesc::XMLString::transcode("contactPoint"), t2 = xercesc::XMLString::transcode("end"));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
        }
    }

    std::map<int, VehicleSource *> sourceMap = fiddleyard->getVehicleSourceMap();
    for (std::map<int, VehicleSource *>::iterator mapIt = sourceMap.begin(); mapIt != sourceMap.end(); ++mapIt)
    {
        xercesc::DOMElement *sourceElement = document->createElement(t1 = xercesc::XMLString::transcode("source")); xercesc::XMLString::release(&t1);
        fiddleyardElement->appendChild(sourceElement);

        sourceElement->setAttribute(t1 = xercesc::XMLString::transcode("id"), t2 = xercesc::XMLString::transcode(mapIt->second->getId().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
        std::ostringstream laneId;
        laneId << mapIt->second->getLane();
        sourceElement->setAttribute(t1 = xercesc::XMLString::transcode("lane"), t2 = xercesc::XMLString::transcode(laneId.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
        std::ostringstream startId;
        startId << std::scientific << mapIt->second->getStartTime();
        sourceElement->setAttribute(t1 = xercesc::XMLString::transcode("startTime"), t2 = xercesc::XMLString::transcode(startId.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
        std::ostringstream repeatId;
        repeatId << std::scientific << mapIt->second->getRepeatTime();
        sourceElement->setAttribute(t1 = xercesc::XMLString::transcode("repeatTime"), t2 = xercesc::XMLString::transcode(repeatId.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    }

    std::map<int, VehicleSink *> sinkMap = fiddleyard->getVehicleSinkMap();
    for (std::map<int, VehicleSink *>::iterator mapIt = sinkMap.begin(); mapIt != sinkMap.end(); ++mapIt)
    {
        xercesc::DOMElement *sinkElement = document->createElement(t1 = xercesc::XMLString::transcode("sink")); xercesc::XMLString::release(&t1);
        fiddleyardElement->appendChild(sinkElement);

        sinkElement->setAttribute(t1 = xercesc::XMLString::transcode("id"), t2 = xercesc::XMLString::transcode(mapIt->second->getId().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
        std::ostringstream laneId;
        laneId << mapIt->second->getLane();
        sinkElement->setAttribute(t1 = xercesc::XMLString::transcode("lane"), t2 = xercesc::XMLString::transcode(laneId.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    }
}

void XodrWriteRoadSystemVisitor::visit(PlaneStraightLine *)
{
	XMLCh *t1 = NULL;
    xercesc::DOMElement *lineElement = document->createElement(t1 = xercesc::XMLString::transcode("line")); xercesc::XMLString::release(&t1);
    geometryElement->appendChild(lineElement);
}

void XodrWriteRoadSystemVisitor::visit(PlaneArc *arc)
{
	XMLCh *t1 = NULL;
	XMLCh *t2 = NULL;
    xercesc::DOMElement *arcElement = document->createElement(t1 = xercesc::XMLString::transcode("arc")); xercesc::XMLString::release(&t1);
    geometryElement->appendChild(arcElement);

    std::ostringstream curvStream;
    curvStream << std::scientific << arc->getCurvature(arc->getStart());
    arcElement->setAttribute(t1 = xercesc::XMLString::transcode("curvature"), t2 = xercesc::XMLString::transcode(curvStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
}

void XodrWriteRoadSystemVisitor::visit(PlaneClothoid *cloth)
{
	XMLCh *t1 = NULL;
	XMLCh *t2 = NULL;
    xercesc::DOMElement *spiralElement = document->createElement(t1 = xercesc::XMLString::transcode("spiral")); xercesc::XMLString::release(&t1);
    geometryElement->appendChild(spiralElement);

    std::ostringstream curvStartStream;
    curvStartStream << std::scientific << cloth->getCurvature(cloth->getStart());
    std::ostringstream curvEndStream;
    curvEndStream << std::scientific << cloth->getCurvature(cloth->getLength() + cloth->getStart());
    spiralElement->setAttribute(t1 = xercesc::XMLString::transcode("curvStart"), t2 = xercesc::XMLString::transcode(curvStartStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    spiralElement->setAttribute(t1 = xercesc::XMLString::transcode("curvEnd"), t2 = xercesc::XMLString::transcode(curvEndStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
}

void XodrWriteRoadSystemVisitor::visit(PlanePolynom *poly)
{
	XMLCh *t1 = NULL;
	XMLCh *t2 = NULL;
    xercesc::DOMElement *poly3Element = document->createElement(t1 = xercesc::XMLString::transcode("poly3")); xercesc::XMLString::release(&t1);
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

    poly3Element->setAttribute(t1 = xercesc::XMLString::transcode("a"), t2 = xercesc::XMLString::transcode(aStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    poly3Element->setAttribute(t1 = xercesc::XMLString::transcode("b"), t2 = xercesc::XMLString::transcode(bStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    poly3Element->setAttribute(t1 = xercesc::XMLString::transcode("c"), t2 = xercesc::XMLString::transcode(cStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    poly3Element->setAttribute(t1 = xercesc::XMLString::transcode("d"), t2 = xercesc::XMLString::transcode(dStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
}

void XodrWriteRoadSystemVisitor::visit(PlaneParamPolynom *poly)
{
	XMLCh *t1 = NULL;
	XMLCh *t2 = NULL;
	xercesc::DOMElement *poly3Element = document->createElement(t1 = xercesc::XMLString::transcode("paramPoly3")); xercesc::XMLString::release(&t1);
	geometryElement->appendChild(poly3Element);

	double aU, bU, cU, dU;
	double aV, bV, cV, dV;
	poly->getCoefficients(aU, bU, cU, dU, aV, bV, cV, dV);

	std::ostringstream aUStream;
	aUStream << std::scientific << aU;
	std::ostringstream bUStream;
	bUStream << std::scientific << bU;
	std::ostringstream cUStream;
	cUStream << std::scientific << cU;
	std::ostringstream dUStream;
	dUStream << std::scientific << dU;
	std::ostringstream aVStream;
	aVStream << std::scientific << aV;
	std::ostringstream bVStream;
	bVStream << std::scientific << bV;
	std::ostringstream cVStream;
	cVStream << std::scientific << cV;
	std::ostringstream dVStream;
	dVStream << std::scientific << dV;

	poly3Element->setAttribute(t1 = xercesc::XMLString::transcode("aU"), t2 = xercesc::XMLString::transcode(aUStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
	poly3Element->setAttribute(t1 = xercesc::XMLString::transcode("bU"), t2 = xercesc::XMLString::transcode(bUStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
	poly3Element->setAttribute(t1 = xercesc::XMLString::transcode("cU"), t2 = xercesc::XMLString::transcode(cUStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
	poly3Element->setAttribute(t1 = xercesc::XMLString::transcode("dU"), t2 = xercesc::XMLString::transcode(dUStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
	poly3Element->setAttribute(t1 = xercesc::XMLString::transcode("aV"), t2 = xercesc::XMLString::transcode(aVStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
	poly3Element->setAttribute(t1 = xercesc::XMLString::transcode("bV"), t2 = xercesc::XMLString::transcode(bVStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
	poly3Element->setAttribute(t1 = xercesc::XMLString::transcode("cV"), t2 = xercesc::XMLString::transcode(cVStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
	poly3Element->setAttribute(t1 = xercesc::XMLString::transcode("dV"), t2 = xercesc::XMLString::transcode(dVStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
	if(poly->isNormalized())
	{
		poly3Element->setAttribute(t1 = xercesc::XMLString::transcode("pRange"), t2 = xercesc::XMLString::transcode("normalized"));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
	}
	else
	{
		poly3Element->setAttribute(t1 = xercesc::XMLString::transcode("pRange"), t2 = xercesc::XMLString::transcode("arcLength"));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
	}
}
void XodrWriteRoadSystemVisitor::visit(Polynom *poly)
{
	XMLCh *t1 = NULL;
	XMLCh *t2 = NULL;
    xercesc::DOMElement *polyElement;
    if (polyType == ELEVATION)
    {
        polyElement = document->createElement(t1 = xercesc::XMLString::transcode("elevation")); xercesc::XMLString::release(&t1);
        elevationProfileElement->appendChild(polyElement);
    }
    else if (polyType == LANEWIDTH)
    {
        polyElement = document->createElement(t1 = xercesc::XMLString::transcode("width")); xercesc::XMLString::release(&t1);
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
        polyElement->setAttribute(t1 = xercesc::XMLString::transcode("s"), t2 = xercesc::XMLString::transcode(sStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    }
    else if (polyType == LANEWIDTH)
    {
        polyElement->setAttribute(t1 = xercesc::XMLString::transcode("sOffset"), t2 = xercesc::XMLString::transcode(sStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    }

    polyElement->setAttribute(t1 = xercesc::XMLString::transcode("a"), t2 = xercesc::XMLString::transcode(aStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    polyElement->setAttribute(t1 = xercesc::XMLString::transcode("b"), t2 = xercesc::XMLString::transcode(bStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    polyElement->setAttribute(t1 = xercesc::XMLString::transcode("c"), t2 = xercesc::XMLString::transcode(cStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    polyElement->setAttribute(t1 = xercesc::XMLString::transcode("d"), t2 = xercesc::XMLString::transcode(dStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
}

void XodrWriteRoadSystemVisitor::visit(SuperelevationPolynom *elev)
{
	XMLCh *t1 = NULL;
	XMLCh *t2 = NULL;
    xercesc::DOMElement *superelevationElement = document->createElement(t1 = xercesc::XMLString::transcode("superelevation")); xercesc::XMLString::release(&t1);
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

    superelevationElement->setAttribute(t1 = xercesc::XMLString::transcode("s"), xercesc::XMLString::transcode(sStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    superelevationElement->setAttribute(t1 = xercesc::XMLString::transcode("a"), xercesc::XMLString::transcode(aStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    superelevationElement->setAttribute(t1 = xercesc::XMLString::transcode("b"), xercesc::XMLString::transcode(bStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    superelevationElement->setAttribute(t1 = xercesc::XMLString::transcode("c"), xercesc::XMLString::transcode(cStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    superelevationElement->setAttribute(t1 = xercesc::XMLString::transcode("d"), xercesc::XMLString::transcode(dStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
}

void XodrWriteRoadSystemVisitor::visit(CrossfallPolynom *fall)
{
	XMLCh *t1 = NULL;
	XMLCh *t2 = NULL;
	if (fall->getLeftFallFactor() == fall->getRightFallFactor() )
	{
		xercesc::DOMElement *shapelElement = document->createElement(t1 = xercesc::XMLString::transcode("shape")); xercesc::XMLString::release(&t1);
		lateralProfileElement->appendChild(shapelElement);
		double a, b, c, d;
		fall->getCoefficients(a, b, c, d);
		std::ostringstream sstream;
		sstream << std::scientific << fall->getStart();
		shapelElement->setAttribute(t1 = xercesc::XMLString::transcode("s"), t2 = xercesc::XMLString::transcode(sstream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
		sstream << std::scientific << a;
		shapelElement->setAttribute(t1 = xercesc::XMLString::transcode("a"), t2 = xercesc::XMLString::transcode(sstream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
		sstream << std::scientific << b;
		shapelElement->setAttribute(t1 = xercesc::XMLString::transcode("b"), t2 = xercesc::XMLString::transcode(sstream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
		sstream << std::scientific << c;
		shapelElement->setAttribute(t1 = xercesc::XMLString::transcode("c"), t2 = xercesc::XMLString::transcode(sstream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
		sstream << std::scientific << d;
		shapelElement->setAttribute(t1 = xercesc::XMLString::transcode("d"), t2 = xercesc::XMLString::transcode(sstream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
		sstream << std::scientific << fall->getRightFallFactor();
		shapelElement->setAttribute(t1 = xercesc::XMLString::transcode("t"), t2 = xercesc::XMLString::transcode(sstream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
	}
	else
	{
		xercesc::DOMElement *crossfallElement = document->createElement(t1 = xercesc::XMLString::transcode("crossfall")); xercesc::XMLString::release(&t1);
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

		crossfallElement->setAttribute(t1 = xercesc::XMLString::transcode("s"), t2 = xercesc::XMLString::transcode(sStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
		crossfallElement->setAttribute(t1 = xercesc::XMLString::transcode("a"), t2 = xercesc::XMLString::transcode(aStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
		crossfallElement->setAttribute(t1 = xercesc::XMLString::transcode("b"), t2 = xercesc::XMLString::transcode(bStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
		crossfallElement->setAttribute(t1 = xercesc::XMLString::transcode("c"), t2 = xercesc::XMLString::transcode(cStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
		crossfallElement->setAttribute(t1 = xercesc::XMLString::transcode("d"), t2 = xercesc::XMLString::transcode(dStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
		crossfallElement->setAttribute(t1 = xercesc::XMLString::transcode("side"), t2 = xercesc::XMLString::transcode(side.c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
	}
}


void XodrWriteRoadSystemVisitor::visit(ShapePolynom *sp)
{
	XMLCh *t1 = NULL;
	XMLCh *t2 = NULL;
		xercesc::DOMElement *shapelElement = document->createElement(t1 = xercesc::XMLString::transcode("shape")); xercesc::XMLString::release(&t1);
		lateralProfileElement->appendChild(shapelElement);
		double a, b, c, d;
		sp->getCoefficients(a, b, c, d);
		std::ostringstream sstream;
		sstream << std::scientific << sp->getS();
		shapelElement->setAttribute(t1 = xercesc::XMLString::transcode("s"), t2 = xercesc::XMLString::transcode(sstream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
		sstream << std::scientific << a;
		shapelElement->setAttribute(t1 = xercesc::XMLString::transcode("a"), t2 = xercesc::XMLString::transcode(sstream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
		sstream << std::scientific << b;
		shapelElement->setAttribute(t1 = xercesc::XMLString::transcode("b"), t2 = xercesc::XMLString::transcode(sstream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
		sstream << std::scientific << c;
		shapelElement->setAttribute(t1 = xercesc::XMLString::transcode("c"), t2 = xercesc::XMLString::transcode(sstream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
		sstream << std::scientific << d;
		shapelElement->setAttribute(t1 = xercesc::XMLString::transcode("d"), t2 = xercesc::XMLString::transcode(sstream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
		sstream << std::scientific << sp->getTStart();
		shapelElement->setAttribute(t1 = xercesc::XMLString::transcode("t"), t2 = xercesc::XMLString::transcode(sstream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
	
}

void XodrWriteRoadSystemVisitor::visit(roadShapePolynoms *sps)
{
	for (auto it = sps->shapes.begin(); it != sps->shapes.end();)
	{
		visit(it->second);
	}
}

void XodrWriteRoadSystemVisitor::visit(LaneSection *section)
{
	XMLCh *t1 = NULL;
	XMLCh *t2 = NULL;
    xercesc::DOMElement *laneSectionElement = document->createElement(t1 = xercesc::XMLString::transcode("laneSection")); xercesc::XMLString::release(&t1);
    lanesElement->appendChild(laneSectionElement);

    std::ostringstream sStream;
    sStream << std::scientific << section->getStart();
    laneSectionElement->setAttribute(t1 = xercesc::XMLString::transcode("s"), t2 = xercesc::XMLString::transcode(sStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);

    xercesc::DOMElement *leftElement = document->createElement(t1 = xercesc::XMLString::transcode("left")); xercesc::XMLString::release(&t1);
    xercesc::DOMElement *centerElement = document->createElement(t1 = xercesc::XMLString::transcode("center")); xercesc::XMLString::release(&t1);
    xercesc::DOMElement *rightElement = document->createElement(t1 = xercesc::XMLString::transcode("right")); xercesc::XMLString::release(&t1);
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
	XMLCh *t1 = NULL;
	XMLCh *t2 = NULL;
    laneElement = document->createElement(t1 = xercesc::XMLString::transcode("lane")); xercesc::XMLString::release(&t1);
    sideElement->appendChild(laneElement);
    std::ostringstream idStream;
    idStream << lane->getId();
    std::string levelString = (lane->isOnLevel()) ? "true" : "false";
    laneElement->setAttribute(t1 = xercesc::XMLString::transcode("id"), t2 = xercesc::XMLString::transcode(idStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    laneElement->setAttribute(t1 = xercesc::XMLString::transcode("type"), t2 = xercesc::XMLString::transcode(lane->getLaneTypeString().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    laneElement->setAttribute(t1 = xercesc::XMLString::transcode("level"), t2 = xercesc::XMLString::transcode(levelString.c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);

    xercesc::DOMElement *linkElement = document->createElement(t1 = xercesc::XMLString::transcode("link")); xercesc::XMLString::release(&t1);
    laneElement->appendChild(linkElement);

    int predId = lane->getPredecessor();
    if (predId != Lane::NOLANE)
    {
        xercesc::DOMElement *predecessorElement = document->createElement(t1 = xercesc::XMLString::transcode("predecessor")); xercesc::XMLString::release(&t1);
        linkElement->appendChild(predecessorElement);
        std::ostringstream predecessorStream;
        predecessorStream << predId;
        predecessorElement->setAttribute(t1 = xercesc::XMLString::transcode("id"), t2 = xercesc::XMLString::transcode(predecessorStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    }

    int succId = lane->getSuccessor();
    if (succId != Lane::NOLANE)
    {
        xercesc::DOMElement *successorElement = document->createElement(t1 = xercesc::XMLString::transcode("successor")); xercesc::XMLString::release(&t1);
        linkElement->appendChild(successorElement);
        std::ostringstream successorStream;
        successorStream << succId;
        successorElement->setAttribute(t1 = xercesc::XMLString::transcode("id"), t2 = xercesc::XMLString::transcode(successorStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
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
	XMLCh *t1 = NULL;
	XMLCh *t2 = NULL;
    xercesc::DOMElement *roadMarkElement = document->createElement(t1 = xercesc::XMLString::transcode("roadMark")); xercesc::XMLString::release(&t1);
    laneElement->appendChild(roadMarkElement);

    std::ostringstream startStream;
    startStream << std::scientific << mark->getStart();
    std::ostringstream widthStream;
    widthStream << std::scientific << mark->getWidth();
    roadMarkElement->setAttribute(t1 = xercesc::XMLString::transcode("sOffset"), t2 = xercesc::XMLString::transcode(startStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    roadMarkElement->setAttribute(t1 = xercesc::XMLString::transcode("type"), t2 = xercesc::XMLString::transcode(mark->getTypeString().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    roadMarkElement->setAttribute(t1 = xercesc::XMLString::transcode("weight"), t2 = xercesc::XMLString::transcode(mark->getWeightString().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    roadMarkElement->setAttribute(t1 = xercesc::XMLString::transcode("color"), t2 = xercesc::XMLString::transcode(mark->getColorString().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    roadMarkElement->setAttribute(t1 = xercesc::XMLString::transcode("width"), t2 = xercesc::XMLString::transcode(widthStream.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    roadMarkElement->setAttribute(t1 = xercesc::XMLString::transcode("laneChange"), t2 = xercesc::XMLString::transcode(mark->getLaneChangeString().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
}

void XodrWriteRoadSystemVisitor::visit(PathConnection *conn)
{
	XMLCh *t1 = NULL;
	XMLCh *t2 = NULL;
    xercesc::DOMElement *connectionElement = document->createElement(t1 = xercesc::XMLString::transcode("connection")); xercesc::XMLString::release(&t1);
    junctionElement->appendChild(connectionElement);

    connectionElement->setAttribute(t1 = xercesc::XMLString::transcode("id"), t2 = xercesc::XMLString::transcode(conn->getId().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    connectionElement->setAttribute(t1 = xercesc::XMLString::transcode("incomingRoad"), t2 = xercesc::XMLString::transcode(conn->getIncomingRoad()->getId().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    connectionElement->setAttribute(t1 = xercesc::XMLString::transcode("connectingRoad"), t2 = xercesc::XMLString::transcode(conn->getConnectingPath()->getId().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    std::string dirString = (conn->getConnectingPathDirection() < 0) ? "end" : "start";
    connectionElement->setAttribute(t1 = xercesc::XMLString::transcode("contactPoint"), t2 = xercesc::XMLString::transcode(dirString.c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);

    LaneConnectionMap laneConnMap = conn->getLaneConnectionMap();
    for (LaneConnectionMap::iterator mapIt = laneConnMap.begin(); mapIt != laneConnMap.end(); ++mapIt)
    {
        xercesc::DOMElement *laneLinkElement = document->createElement(t1 = xercesc::XMLString::transcode("laneLink")); xercesc::XMLString::release(&t1);
        connectionElement->appendChild(laneLinkElement);

        std::ostringstream from;
        from << mapIt->first;
        std::ostringstream to;
        to << mapIt->second;

        laneLinkElement->setAttribute(t1 = xercesc::XMLString::transcode("from"), t2 = xercesc::XMLString::transcode(from.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
        laneLinkElement->setAttribute(t1 = xercesc::XMLString::transcode("to"), t2 = xercesc::XMLString::transcode(to.str().c_str()));  xercesc::XMLString::release(&t2); xercesc::XMLString::release(&t1);
    }
}
