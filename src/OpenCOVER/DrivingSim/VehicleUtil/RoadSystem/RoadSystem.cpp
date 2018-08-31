/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <xercesc/dom/DOM.hpp>
//#include <xercesc/dom/DOMWriter.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/BaseRefVectorOf.hpp>

#include "RoadSystem.h"

#include "OpenCRGSurface.h"

#include <string>
#include <sstream>
#include <fstream>
#include <limits>
#include <deque>
#include <iomanip>

#include <proj_api.h>

int RoadSystem::_tiles_x = 0;
int RoadSystem::_tiles_y = 0;
float RoadSystem::dSpace_v = -1;
RoadSystem *RoadSystem::__instance = NULL;

RoadSystem *RoadSystem::Instance()
{
    if (__instance == NULL)
    {
        __instance = new RoadSystem();
    }
    return __instance;
}

void RoadSystem::Destroy()
{
    delete __instance;
    __instance = NULL;
}

RoadSystem::RoadSystem()
{
}

const RoadSystemHeader &RoadSystem::getHeader()
{
    return header;
}

void RoadSystem::addRoad(Road *road)
{
    roadVector.push_back(road);
    roadIdMap[road->getId()] = road;
}

void RoadSystem::addController(Controller *controller)
{
    controllerVector.push_back(controller);
    controllerIdMap[controller->getId()] = controller;
}

void RoadSystem::addJunction(Junction *junction)
{
    junctionVector.push_back(junction);
    junctionIdMap[junction->getId()] = junction;
}

void RoadSystem::addFiddleyard(Fiddleyard *fiddleyard)
{
    fiddleyardVector.push_back(fiddleyard);
    fiddleyardIdMap[fiddleyard->getId()] = fiddleyard;
}

void RoadSystem::addPedFiddleyard(PedFiddleyard *fiddleyard)
{
    pedFiddleyardVector.push_back(fiddleyard);
}

void RoadSystem::clearPedFiddleyards()
{
    pedFiddleyardVector.clear();
}

std::string RoadSystem::getRoadId(Road *road)
{
    std::map<std::string, Road *>::iterator roadIt = roadIdMap.begin();
    for (; roadIt != roadIdMap.end(); roadIt++)
    {
        if (roadIt->second == road)
            return roadIt->first;
    }
    return std::string("");
}

void RoadSystem::addRoadSignal(RoadSignal *signal)
{
    signalVector.push_back(signal);
    signalIdMap[signal->getId()] = signal;
}

void RoadSystem::addRoadSensor(RoadSensor *sensor)
{
    sensorVector.push_back(sensor);
    sensorIdMap[sensor->getId()] = sensor;
}

Road *RoadSystem::getRoad(int i)
{
    return roadVector[i];
}
Road *RoadSystem::getRoad(std::string id)
{
    std::map<std::string, Road *>::iterator roadMapIt = roadIdMap.find(id);
    if (roadMapIt != roadIdMap.end())
    {
        return roadMapIt->second;
    }
    else
    {
        return NULL;
    }
}
int RoadSystem::getNumRoads()
{
    return roadVector.size();
}

Controller *RoadSystem::getController(int i)
{
    return controllerVector[i];
}

int RoadSystem::getNumControllers()
{
    return controllerVector.size();
}

Junction *RoadSystem::getJunction(int i)
{
    return junctionVector[i];
}

int RoadSystem::getNumJunctions()
{
    return junctionVector.size();
}

Fiddleyard *RoadSystem::getFiddleyard(int i)
{
    return fiddleyardVector[i];
}

int RoadSystem::getNumFiddleyards()
{
    return fiddleyardVector.size();
}

PedFiddleyard *RoadSystem::getPedFiddleyard(int i)
{
    return pedFiddleyardVector[i];
}

int RoadSystem::getNumPedFiddleyards()
{
    return pedFiddleyardVector.size();
}

RoadSignal *RoadSystem::getRoadSignal(int i)
{
    return signalVector[i];
}

int RoadSystem::getNumRoadSignals()
{
    return signalVector.size();
}

RoadSensor *RoadSystem::getRoadSensor(int i)
{
    return sensorVector[i];
}

int RoadSystem::getNumRoadSensors()
{
    return sensorVector.size();
}

void RoadSystem::parseOpenDrive(std::string filename)
{
    try
    {
        xercesc::XMLPlatformUtils::Initialize();
    }
    catch (const xercesc::XMLException &toCatch)
    {
        char *message = xercesc::XMLString::transcode(toCatch.getMessage());
        std::cout << "Error during initialization! :\n" << message << std::endl;
        xercesc::XMLString::release(&message);
        return;
    }

    xercesc::XercesDOMParser *parser = new xercesc::XercesDOMParser();
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);

    try
    {
        parser->parse(filename.c_str());
    }
    catch (...)
    {
        std::cerr << "Couldn't parse OpenDRIVE XML-file " << filename << "!" << std::endl;
    }

    xercesc::DOMDocument *xmlDoc = parser->getDocument();
    if (xmlDoc)
    {
        parseOpenDrive(xmlDoc->getDocumentElement());
    }
    xercesc::XMLPlatformUtils::Terminate();
}

void RoadSystem::parseOpenDrive(xercesc::DOMElement *rootElement)
{
    if (rootElement)
    {
		XMLCh *t1=NULL, *t2=NULL, *t3=NULL, *t4=NULL, *t5=NULL, *t6=NULL, *t7=NULL, *t8=NULL, *t9=NULL, *t10=NULL, *t11=NULL, *t12=NULL, *t13=NULL, *t14 = NULL, *t15 = NULL, *t16 = NULL, *t17 = NULL, *t18=NULL;
		char *cs;
        Road *road;
        std::map<Road *, xercesc::DOMElement *> roadDOMMap;
        xercesc::DOMNodeList *documentChildrenList = rootElement->getChildNodes();
        xercesc::DOMElement *documentChildElement;
        for (unsigned int childIndex = 0; childIndex < documentChildrenList->getLength(); ++childIndex)
        {
            documentChildElement = dynamic_cast<xercesc::DOMElement *>(documentChildrenList->item(childIndex));

            if (documentChildElement && xercesc::XMLString::compareIString(documentChildElement->getTagName(), t1 = xercesc::XMLString::transcode("road")) == 0)
            {
                std::string roadIdString = cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t2 = xercesc::XMLString::transcode("id"))); xercesc::XMLString::release(&t2); xercesc::XMLString::release(&cs);
                if (roadIdString.empty())
                {
                    std::string roadIdString = cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t2 = xercesc::XMLString::transcode("ID"))); xercesc::XMLString::release(&t2); xercesc::XMLString::release(&cs);
                }
                std::string roadNameString = cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t2 = xercesc::XMLString::transcode("name"))); xercesc::XMLString::release(&t2); xercesc::XMLString::release(&cs);
                double roadLength = atof(cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t2= xercesc::XMLString::transcode("length")))); xercesc::XMLString::release(&t2); xercesc::XMLString::release(&cs);
                road = new Road(roadIdString, roadNameString, roadLength);
                addRoad(road);
                roadDOMMap[road] = documentChildElement;

                xercesc::DOMNodeList *roadChildrenList = documentChildElement->getChildNodes();
                xercesc::DOMElement *roadChildElement;
                for (unsigned int childIndex = 0; childIndex < roadChildrenList->getLength(); ++childIndex)
                {
                    roadChildElement = dynamic_cast<xercesc::DOMElement *>(roadChildrenList->item(childIndex));
                    if (!roadChildElement)
                    {
                        //std::cerr << "A not-an-element in road tag";
                        //std::cerr << ", type of node: " << xercesc::XMLString::transcode(roadChildElement->getNodeName()) << std::endl;
                    }
                    else if (xercesc::XMLString::compareIString(roadChildElement->getTagName(), t2 = xercesc::XMLString::transcode("type")) == 0)
                    {
                        double typeStart;
                        std::string typeName;

                        typeStart = atof(cs = xercesc::XMLString::transcode(roadChildElement->getAttribute(t12 = xercesc::XMLString::transcode("s")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        typeName = cs = xercesc::XMLString::transcode(roadChildElement->getAttribute(t12 = xercesc::XMLString::transcode("type"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        road->addRoadType(typeStart, typeName);
                        // Neu Andreas 27-11-2012: No speedlimit in .xodr defined, then get standard values
                        road->addSpeedLimit(typeStart, typeName);
                    }
                    // Neu Andreas 27-11-2012: Speedlimit defined in .xodr
                    else if (xercesc::XMLString::compareIString(roadChildElement->getTagName(), t3 = xercesc::XMLString::transcode("speedlimit")) == 0)
                    {
                        double speedLimitStart;
                        double speedLimitNum;
                        speedLimitStart = atof(cs = xercesc::XMLString::transcode(roadChildElement->getAttribute(t12 = xercesc::XMLString::transcode("s")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        speedLimitNum = atof(cs = xercesc::XMLString::transcode(roadChildElement->getAttribute(t12 = xercesc::XMLString::transcode("max")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        road->addSpeedLimit(speedLimitStart, speedLimitNum);
                    }
                    else if (xercesc::XMLString::compareIString(roadChildElement->getTagName(), t4 = xercesc::XMLString::transcode("planView")) == 0)
                    {
                        double geometryStart;
                        double geometryX;
                        double geometryY;
                        double geometryHdg;
                        double geometryLength;
                        xercesc::DOMNodeList *planViewChildrenList = roadChildElement->getChildNodes();
                        xercesc::DOMElement *planViewChildElement;
                        for (unsigned int childIndex = 0; childIndex < planViewChildrenList->getLength(); ++childIndex)
                        {
                            planViewChildElement = dynamic_cast<xercesc::DOMElement *>(planViewChildrenList->item(childIndex));
                            if (planViewChildElement && xercesc::XMLString::compareIString(planViewChildElement->getTagName(), xercesc::XMLString::transcode("geometry")) == 0)
                            {
                                geometryStart = atof(cs = xercesc::XMLString::transcode(planViewChildElement->getAttribute(t12 = xercesc::XMLString::transcode("s")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                geometryX = atof(cs = xercesc::XMLString::transcode(planViewChildElement->getAttribute(t12 = xercesc::XMLString::transcode("x")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                geometryY = atof(cs = xercesc::XMLString::transcode(planViewChildElement->getAttribute(t12 = xercesc::XMLString::transcode("y")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                geometryHdg = atof(cs = xercesc::XMLString::transcode(planViewChildElement->getAttribute(t12 = xercesc::XMLString::transcode("hdg")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                geometryLength = atof(cs = xercesc::XMLString::transcode(planViewChildElement->getAttribute(t12 = xercesc::XMLString::transcode("length")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                xercesc::DOMNodeList *curveList = planViewChildElement->getChildNodes();
                                xercesc::DOMElement *curveElement;
                                for (unsigned int curveIndex = 0; curveIndex < curveList->getLength(); ++curveIndex)
                                {
                                    curveElement = dynamic_cast<xercesc::DOMElement *>(curveList->item(curveIndex));
                                    if (!curveElement)
                                    {
                                        //std::cerr << "A not-an-element in plan view geometry tag with start s: " << geometryStart;
                                        //std::cerr << ", type of node: " << xercesc::XMLString::transcode(curveElement->getNodeName()) << std::endl;
                                        //std::cerr << "Content of node: " << xercesc::XMLString::transcode(curveElement->getTextContent()) << std::endl;
                                    }
                                    else if (xercesc::XMLString::compareIString(curveElement->getTagName(), t5 = xercesc::XMLString::transcode("line")) == 0)
                                    {
                                        //std::cerr << "Added line geometry at s: " << geometryStart << std::endl;
                                        road->addPlanViewGeometryLine(geometryStart, geometryLength, geometryX, geometryY, geometryHdg);
                                    }
                                    else if (xercesc::XMLString::compareIString(curveElement->getTagName(), t6 = xercesc::XMLString::transcode("spiral")) == 0)
                                    {
                                        double curveCurvStart = atof(cs = xercesc::XMLString::transcode(curveElement->getAttribute(t12 = xercesc::XMLString::transcode("curvStart")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                        double curveCurvEnd = atof(cs = xercesc::XMLString::transcode(curveElement->getAttribute(t12 = xercesc::XMLString::transcode("curvEnd")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                        road->addPlanViewGeometrySpiral(geometryStart, geometryLength, geometryX, geometryY, geometryHdg, curveCurvStart, curveCurvEnd);
                                    }
                                    else if (xercesc::XMLString::compareIString(curveElement->getTagName(), t7 = xercesc::XMLString::transcode("arc")) == 0)
                                    {
                                        double curveCurvature = atof(cs = xercesc::XMLString::transcode(curveElement->getAttribute(t12 = xercesc::XMLString::transcode("curvature")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                        road->addPlanViewGeometryArc(geometryStart, geometryLength, geometryX, geometryY, geometryHdg, curveCurvature);
                                    }
                                    else if (xercesc::XMLString::compareIString(curveElement->getTagName(), t8 = xercesc::XMLString::transcode("poly3")) == 0)
                                    {
                                        double curveA = atof(cs = xercesc::XMLString::transcode(curveElement->getAttribute(t12 = xercesc::XMLString::transcode("a")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                        double curveB = atof(cs = xercesc::XMLString::transcode(curveElement->getAttribute(t12 = xercesc::XMLString::transcode("b")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                        double curveC = atof(cs = xercesc::XMLString::transcode(curveElement->getAttribute(t12 = xercesc::XMLString::transcode("c")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                        double curveD = atof(cs = xercesc::XMLString::transcode(curveElement->getAttribute(t12 = xercesc::XMLString::transcode("d")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                        road->addPlanViewGeometryPolynom(geometryStart, geometryLength, geometryX, geometryY, geometryHdg, curveA, curveB, curveC, curveD);
                                    }
									else if (xercesc::XMLString::compareIString(curveElement->getTagName(), t9 = xercesc::XMLString::transcode("paramPoly3")) == 0)
									{
										double curveAU = atof(cs = xercesc::XMLString::transcode(curveElement->getAttribute(t12 = xercesc::XMLString::transcode("aU")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
										double curveBU = atof(cs = xercesc::XMLString::transcode(curveElement->getAttribute(t12 = xercesc::XMLString::transcode("bU")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
										double curveCU = atof(cs = xercesc::XMLString::transcode(curveElement->getAttribute(t12 = xercesc::XMLString::transcode("cU")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
										double curveDU = atof(cs = xercesc::XMLString::transcode(curveElement->getAttribute(t12 = xercesc::XMLString::transcode("dU")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
										double curveAV = atof(cs = xercesc::XMLString::transcode(curveElement->getAttribute(t12 = xercesc::XMLString::transcode("aV")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
										double curveBV = atof(cs = xercesc::XMLString::transcode(curveElement->getAttribute(t12 = xercesc::XMLString::transcode("bV")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
										double curveCV = atof(cs = xercesc::XMLString::transcode(curveElement->getAttribute(t12 = xercesc::XMLString::transcode("cV")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
										double curveDV = atof(cs = xercesc::XMLString::transcode(curveElement->getAttribute(t12 = xercesc::XMLString::transcode("dV")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
										bool normalized = true;
										if(curveElement->getAttribute(xercesc::XMLString::transcode("pRange")))
										{
										    std::string pRange = cs = xercesc::XMLString::transcode(curveElement->getAttribute(t12 = xercesc::XMLString::transcode("pRange"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
											if (pRange == "arcLength")
												normalized = false;
										}
										road->addPlanViewGeometryPolynom(geometryStart, geometryLength, geometryX, geometryY, geometryHdg, curveAU, curveBU, curveCU, curveDU, curveAV, curveBV, curveCV, curveDV,normalized);
									}
									xercesc::XMLString::release(&t5);
									xercesc::XMLString::release(&t6);
									xercesc::XMLString::release(&t7);
									xercesc::XMLString::release(&t8);
									xercesc::XMLString::release(&t9);
                                }
                            }
                        }
                    }
                    else if (xercesc::XMLString::compareIString(roadChildElement->getTagName(), t5 = xercesc::XMLString::transcode("elevationProfile")) == 0)
                    {
                        xercesc::DOMNodeList *elevationProfileChildrenList = roadChildElement->getChildNodes();
                        xercesc::DOMElement *elevationProfileChildElement;
                        for (unsigned int childIndex = 0; childIndex < elevationProfileChildrenList->getLength(); ++childIndex)
                        {
                            elevationProfileChildElement = dynamic_cast<xercesc::DOMElement *>(elevationProfileChildrenList->item(childIndex));
                            if (elevationProfileChildElement && xercesc::XMLString::compareIString(elevationProfileChildElement->getTagName(), t6 = xercesc::XMLString::transcode("elevation")) == 0)
                            {
                                double elevationStart = atof(cs = xercesc::XMLString::transcode(elevationProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("s")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double elevationA = atof(cs = xercesc::XMLString::transcode(elevationProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("a")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double elevationB = atof(cs = xercesc::XMLString::transcode(elevationProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("b")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double elevationC = atof(cs = xercesc::XMLString::transcode(elevationProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("c")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double elevationD = atof(cs = xercesc::XMLString::transcode(elevationProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("d")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                road->addElevationPolynom(elevationStart, elevationA, elevationB, elevationC, elevationD);
                            }
							xercesc::XMLString::release(&t6);
                        }
                    }
                    else if (xercesc::XMLString::compareIString(roadChildElement->getTagName(), t6 = xercesc::XMLString::transcode("lateralProfile")) == 0)
                    {
                        xercesc::DOMNodeList *lateralProfileChildrenList = roadChildElement->getChildNodes();
                        xercesc::DOMElement *lateralProfileChildElement;
                        for (unsigned int childIndex = 0; childIndex < lateralProfileChildrenList->getLength(); ++childIndex)
                        {
                            lateralProfileChildElement = dynamic_cast<xercesc::DOMElement *>(lateralProfileChildrenList->item(childIndex));
                            if (lateralProfileChildElement && xercesc::XMLString::compareIString(lateralProfileChildElement->getTagName(), t7 = xercesc::XMLString::transcode("superelevation")) == 0)
                            {
                                double superelevationStart = atof(cs = xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("s")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double superelevationA = atof(cs = xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("a")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double superelevationB = atof(cs = xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("b")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double superelevationC = atof(cs = xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("c")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double superelevationD = atof(cs = xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("d")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                road->addSuperelevationPolynom(superelevationStart, superelevationA, superelevationB, superelevationC, superelevationD);
                            }
                            else if (lateralProfileChildElement && xercesc::XMLString::compareIString(lateralProfileChildElement->getTagName(), t8 = xercesc::XMLString::transcode("crossfall")) == 0)
                            {
                                double crossfallStart = atof(cs = xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("s")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double crossfallA = atof(cs = xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("a")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double crossfallB = atof(cs = xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("b")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double crossfallC = atof(cs = xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("c")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double crossfallD = atof(cs = xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("d")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                std::string crossfallSide(cs = xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("side")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                road->addCrossfallPolynom(crossfallStart, crossfallA, crossfallB, crossfallC, crossfallD, crossfallSide);
                            }
							else if (lateralProfileChildElement && xercesc::XMLString::compareIString(lateralProfileChildElement->getTagName(), t9 = xercesc::XMLString::transcode("shape")) == 0)
							{
								double crossfallStart = atof(cs = xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("s")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
								double crossfallA = atof(cs = xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("a")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
								double crossfallB = atof(cs = xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("b")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
								double crossfallC = atof(cs = xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("c")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
								double crossfallD = atof(cs = xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("d")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
								double crossfallT = atof(cs = xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(t12 = xercesc::XMLString::transcode("t")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
								road->addShapePolynom(crossfallStart, crossfallA, crossfallB, crossfallC, crossfallD, crossfallT);
							}
							xercesc::XMLString::release(&t7);
							xercesc::XMLString::release(&t8);
							xercesc::XMLString::release(&t9);
                        }
                    }

                    else if (xercesc::XMLString::compareIString(roadChildElement->getTagName(), t7 = xercesc::XMLString::transcode("lanes")) == 0)
                    {
                        xercesc::DOMNodeList *lanesChildrenList = roadChildElement->getChildNodes();
                        xercesc::DOMElement *lanesChildElement;
                        for (unsigned int childIndex = 0; childIndex < lanesChildrenList->getLength(); ++childIndex)
                        {
                            lanesChildElement = dynamic_cast<xercesc::DOMElement *>(lanesChildrenList->item(childIndex));
							if (lanesChildElement && xercesc::XMLString::compareIString(lanesChildElement->getTagName(),t8 =  xercesc::XMLString::transcode("laneSection")) == 0)
							{
								double laneSectionStart = atof(cs = xercesc::XMLString::transcode(lanesChildElement->getAttribute(t12 = xercesc::XMLString::transcode("s")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
								LaneSection *section = new LaneSection(road, laneSectionStart);
								road->addLaneSection(section);
								xercesc::DOMNodeList *laneSectionChildrenList = lanesChildElement->getChildNodes();
								xercesc::DOMElement *laneSectionChildElement;
								for (unsigned int childIndex = 0; childIndex < laneSectionChildrenList->getLength(); ++childIndex)
								{
									laneSectionChildElement = dynamic_cast<xercesc::DOMElement *>(laneSectionChildrenList->item(childIndex));
									if (laneSectionChildElement && (xercesc::XMLString::compareIString(laneSectionChildElement->getTagName(), t9 = xercesc::XMLString::transcode("left")) == 0
										|| xercesc::XMLString::compareIString(laneSectionChildElement->getTagName(), t10 = xercesc::XMLString::transcode("center")) == 0
										|| xercesc::XMLString::compareIString(laneSectionChildElement->getTagName(), t11 = xercesc::XMLString::transcode("right")) == 0))
									{
										xercesc::DOMNodeList *laneList = laneSectionChildElement->getChildNodes();
										xercesc::DOMElement *laneElement;
										for (unsigned int childIndex = 0; childIndex < laneList->getLength(); ++childIndex)
										{
											laneElement = dynamic_cast<xercesc::DOMElement *>(laneList->item(childIndex));
											if (laneElement && xercesc::XMLString::compareIString(laneElement->getTagName(), t13 = xercesc::XMLString::transcode("lane")) == 0)
											{
												int laneId = atoi(cs = xercesc::XMLString::transcode(laneElement->getAttribute(t12 = xercesc::XMLString::transcode("id")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
												std::string laneType(cs = xercesc::XMLString::transcode(laneElement->getAttribute(t12 = xercesc::XMLString::transcode("type")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
												std::string laneLevel(cs = xercesc::XMLString::transcode(laneElement->getAttribute(t12 = xercesc::XMLString::transcode("level")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
												Lane *lane = new Lane(laneId, laneType, laneLevel);
												section->addLane(lane);
												xercesc::DOMNodeList *laneChildrenList = laneElement->getChildNodes();
												xercesc::DOMElement *laneChildElement;
												for (unsigned int childIndex = 0; childIndex < laneChildrenList->getLength(); ++childIndex)
												{
													laneChildElement = dynamic_cast<xercesc::DOMElement *>(laneChildrenList->item(childIndex));
													if (!laneChildElement)
													{
													}

													else if (xercesc::XMLString::compareIString(laneChildElement->getTagName(), t14 = xercesc::XMLString::transcode("link")) == 0)
													{
														xercesc::DOMNodeList *linkChildrenList = laneChildElement->getChildNodes();
														xercesc::DOMElement *linkChildElement;
														for (unsigned int childIndex = 0; childIndex < linkChildrenList->getLength(); ++childIndex)
														{
															linkChildElement = dynamic_cast<xercesc::DOMElement *>(linkChildrenList->item(childIndex));
															if (!linkChildElement)
															{
															}
															else if (xercesc::XMLString::compareIString(linkChildElement->getTagName(), t15 = xercesc::XMLString::transcode("predecessor")) == 0)
															{
																int predecessorId = atoi(cs = xercesc::XMLString::transcode(linkChildElement->getAttribute(t12 = xercesc::XMLString::transcode("id")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
																lane->setPredecessor(predecessorId);
															}
															else if (xercesc::XMLString::compareIString(linkChildElement->getTagName(), t16 = xercesc::XMLString::transcode("successor")) == 0)
															{
																int successorId = atoi(cs = xercesc::XMLString::transcode(linkChildElement->getAttribute(t12 = xercesc::XMLString::transcode("id")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
																lane->setSuccessor(successorId);
															}
															xercesc::XMLString::release(&t15); xercesc::XMLString::release(&t16);
														}
													}
													else if (xercesc::XMLString::compareIString(laneChildElement->getTagName(), t15 = xercesc::XMLString::transcode("width")) == 0)
													{
														double widthStart = atof(cs = xercesc::XMLString::transcode(laneChildElement->getAttribute(t12 = xercesc::XMLString::transcode("sOffset")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
														double widthA = atof(cs = xercesc::XMLString::transcode(laneChildElement->getAttribute(t12 = xercesc::XMLString::transcode("a")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
														double widthB = atof(cs = xercesc::XMLString::transcode(laneChildElement->getAttribute(t12 = xercesc::XMLString::transcode("b")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
														double widthC = atof(cs = xercesc::XMLString::transcode(laneChildElement->getAttribute(t12 = xercesc::XMLString::transcode("c")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
														double widthD = atof(cs = xercesc::XMLString::transcode(laneChildElement->getAttribute(t12 = xercesc::XMLString::transcode("d")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
														lane->addWidth(widthStart, widthA, widthB, widthC, widthD);
													}

													else if (xercesc::XMLString::compareIString(laneChildElement->getTagName(), t16 = xercesc::XMLString::transcode("roadMark")) == 0)
													{
														double roadMarkStart = atof(cs = xercesc::XMLString::transcode(laneChildElement->getAttribute(t12 = xercesc::XMLString::transcode("sOffset")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
														double roadMarkWidth = atof(cs = xercesc::XMLString::transcode(laneChildElement->getAttribute(t12 = xercesc::XMLString::transcode("width")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
														std::string roadMarkType(cs = xercesc::XMLString::transcode(laneChildElement->getAttribute(t12 = xercesc::XMLString::transcode("type")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
														std::string roadMarkWeight(cs = xercesc::XMLString::transcode(laneChildElement->getAttribute(t12 = xercesc::XMLString::transcode("weight")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
														std::string roadMarkColor(cs = xercesc::XMLString::transcode(laneChildElement->getAttribute(t12 = xercesc::XMLString::transcode("color")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
														std::string roadMarkLaneChange(cs = xercesc::XMLString::transcode(laneChildElement->getAttribute(t12 = xercesc::XMLString::transcode("laneChange")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
														lane->addRoadMark(new RoadMark(roadMarkStart, roadMarkWidth, roadMarkType, roadMarkWeight, roadMarkColor, roadMarkLaneChange));
													}
													// Neu Andreas 27-11-2012
													else if (xercesc::XMLString::compareIString(laneChildElement->getTagName(), t17 = xercesc::XMLString::transcode("speed")) == 0)
													{
														double speedLimitStart = atof(cs = xercesc::XMLString::transcode(laneChildElement->getAttribute(t12 = xercesc::XMLString::transcode("sOffset")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
														double speedLimitNum = atof(cs = xercesc::XMLString::transcode(laneChildElement->getAttribute(t12 = xercesc::XMLString::transcode("max")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
														double factor = 1.0;
														if (laneChildElement->getAttribute(t18 = xercesc::XMLString::transcode("unit")))
														{
															std::string speedUnit = cs = xercesc::XMLString::transcode(laneChildElement->getAttribute(t12 = xercesc::XMLString::transcode("unit"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
															if (speedUnit == "mph")
																factor = 0.44704;
															if (speedUnit == "km/h")
																factor = 0.277778;
														}
														lane->addSpeedLimit(speedLimitStart, speedLimitNum*factor);
														xercesc::XMLString::release(&t18);
													}
													else if (xercesc::XMLString::compareIString(laneChildElement->getTagName(),t18 =  xercesc::XMLString::transcode("height")) == 0)
													{
														double heightStart = atof(cs = xercesc::XMLString::transcode(laneChildElement->getAttribute(t12 = xercesc::XMLString::transcode("sOffset")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
														double inner = atof(cs = xercesc::XMLString::transcode(laneChildElement->getAttribute(t12 = xercesc::XMLString::transcode("inner")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
														double outer = atof(cs = xercesc::XMLString::transcode(laneChildElement->getAttribute(t12 = xercesc::XMLString::transcode("outer")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);

														lane->addHeight(heightStart, inner, outer);
													}
													xercesc::XMLString::release(&t13);
													xercesc::XMLString::release(&t14);
													xercesc::XMLString::release(&t15);
													xercesc::XMLString::release(&t16);
													xercesc::XMLString::release(&t17);
													xercesc::XMLString::release(&t18);
												}
											}
											xercesc::XMLString::release(&t13);
										}
									}
									xercesc::XMLString::release(&t9);
									xercesc::XMLString::release(&t10);
									xercesc::XMLString::release(&t11);
									if (laneSectionChildElement && (xercesc::XMLString::compareIString(laneSectionChildElement->getTagName(), t9 = xercesc::XMLString::transcode("left")) == 0
										|| xercesc::XMLString::compareIString(laneSectionChildElement->getTagName(), t10 = xercesc::XMLString::transcode("right")) == 0))
									{
										xercesc::DOMNodeList *batterList = laneSectionChildElement->getChildNodes();
										xercesc::DOMElement *batterElement;
										for (unsigned int childIndex = 0; childIndex < batterList->getLength(); ++childIndex)
										{
											batterElement = dynamic_cast<xercesc::DOMElement *>(batterList->item(childIndex));
											if (batterElement && xercesc::XMLString::compareIString(batterElement->getTagName(), t11 = xercesc::XMLString::transcode("batter")) == 0)
											{
												int batterId = atoi(cs = xercesc::XMLString::transcode(batterElement->getAttribute(t12 = xercesc::XMLString::transcode("id")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
												std::string tessellateString(cs = xercesc::XMLString::transcode(batterElement->getAttribute(t12 = xercesc::XMLString::transcode("tessellate")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
												bool tessellate = !(tessellateString == "no" || tessellateString == "false");
												std::cout << "batter id: " << batterId << std::endl;
												Batter *batter = new Batter(batterId, tessellate);
												section->addBatter(batter);
												xercesc::DOMNodeList *batterChildrenList = batterElement->getChildNodes();
												xercesc::DOMElement *batterChildElement;
												for (unsigned int childIndex = 0; childIndex < batterChildrenList->getLength(); ++childIndex)
												{
													batterChildElement = dynamic_cast<xercesc::DOMElement *>(batterChildrenList->item(childIndex));
													if (!batterChildElement)
													{
													}
													else if (xercesc::XMLString::compareIString(batterChildElement->getTagName(), t13 = xercesc::XMLString::transcode("width")) == 0)
													{
														double widthStart = atof(cs = xercesc::XMLString::transcode(batterChildElement->getAttribute(t12 = xercesc::XMLString::transcode("sOffset")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
														double widthA = atof(cs = xercesc::XMLString::transcode(batterChildElement->getAttribute(t12 = xercesc::XMLString::transcode("a")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
														batter->addBatterWidth(widthStart, widthA);
													}

													else if (xercesc::XMLString::compareIString(batterChildElement->getTagName(), t14 = xercesc::XMLString::transcode("fall")) == 0)
													{

														double widthStart = atof(cs = xercesc::XMLString::transcode(batterChildElement->getAttribute(t12 = xercesc::XMLString::transcode("sOffset")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
														double widthA = atof(cs = xercesc::XMLString::transcode(batterChildElement->getAttribute(t12 = xercesc::XMLString::transcode("a")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
														batter->addBatterFall(widthStart, widthA);
													}
													xercesc::XMLString::release(&t13);
													xercesc::XMLString::release(&t14);
												}
											}
											xercesc::XMLString::release(&t11);
										}
									}
									xercesc::XMLString::release(&t9);
									xercesc::XMLString::release(&t10);
								}
							}

							else if (lanesChildElement && xercesc::XMLString::compareIString(lanesChildElement->getTagName(), t9 = xercesc::XMLString::transcode("laneOffset")) == 0)
							{
								double widthStart = atof(cs = xercesc::XMLString::transcode(lanesChildElement->getAttribute(t12 = xercesc::XMLString::transcode("s")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
								double widthA = atof(cs = xercesc::XMLString::transcode(lanesChildElement->getAttribute(t12 = xercesc::XMLString::transcode("a")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
								double widthB = atof(cs = xercesc::XMLString::transcode(lanesChildElement->getAttribute(t12 = xercesc::XMLString::transcode("b")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
								double widthC = atof(cs = xercesc::XMLString::transcode(lanesChildElement->getAttribute(t12 = xercesc::XMLString::transcode("c")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
								double widthD = atof(cs = xercesc::XMLString::transcode(lanesChildElement->getAttribute(t12 = xercesc::XMLString::transcode("d")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
								road->addLaneOffset(widthStart, widthA, widthB, widthC, widthD);
							}
							xercesc::XMLString::release(&t8);
							xercesc::XMLString::release(&t9);
                        }
                    }

                    else if (xercesc::XMLString::compareIString(roadChildElement->getTagName(), t8 = xercesc::XMLString::transcode("objects")) == 0)
                    {
                        xercesc::DOMNodeList *objectsChildrenList = roadChildElement->getChildNodes();
                        xercesc::DOMElement *objectElement;
                        for (unsigned int childIndex = 0; childIndex < objectsChildrenList->getLength(); ++childIndex)
                        {
                            objectElement = dynamic_cast<xercesc::DOMElement *>(objectsChildrenList->item(childIndex));
                            if (objectElement && xercesc::XMLString::compareIString(objectElement->getTagName(), t9 = xercesc::XMLString::transcode("object")) == 0)
                            {
                                std::string type(cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("type")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                if (type != "simplePole")
                                {
                                    std::string name(cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("name")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    std::string id(cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("id")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    std::string file(cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("modelFile")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    std::string textureFile(cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("textureFile")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);

                                    double s = atof(cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("s")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    double t = atof(cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("t")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    double zOffset = atof(cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("zOffset")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    double validLength = atof(cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("validLength")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);

                                    std::string orientationString(cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("orientation")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    RoadObject::OrientationType orientation = RoadObject::BOTH_DIRECTIONS;
                                    if (orientationString == "+")
                                    {
                                        orientation = RoadObject::POSITIVE_TRACK_DIRECTION;
                                    }
                                    else if (orientationString == "-")
                                    {
                                        orientation = RoadObject::NEGATIVE_TRACK_DIRECTION;
                                    }

                                    double length = atof(cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("length")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    double width = atof(cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("width")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    double radius = atof(cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("radius")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    double height = atof(cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("height")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    double hdg = atof(cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("hdg")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    double pitch = atof(cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("pitch")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    double roll = atof(cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("roll")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);

                                    RoadObject *roadObject = new RoadObject(id, file,textureFile, name, type, s, t, zOffset, validLength, orientation,
                                        length, width, radius, height, hdg, pitch, roll, road);
                                    road->addRoadObject(roadObject);

                                    xercesc::DOMNodeList *objectChildrenList = objectElement->getChildNodes();
                                    xercesc::DOMElement *objectChildElement;
                                    for (unsigned int childIndex = 0; childIndex < objectChildrenList->getLength(); ++childIndex)
                                    {
                                        objectChildElement = dynamic_cast<xercesc::DOMElement *>(objectChildrenList->item(childIndex));
                                        if (objectChildElement)
                                        {
                                            if (xercesc::XMLString::compareIString(objectChildElement->getTagName(), t10 = xercesc::XMLString::transcode("repeat")) == 0)
                                            {
                                                double repeatS = atof(cs = xercesc::XMLString::transcode(objectChildElement->getAttribute(t12 = xercesc::XMLString::transcode("s")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                                double repeatLength = atof(cs = xercesc::XMLString::transcode(objectChildElement->getAttribute(t12 = xercesc::XMLString::transcode("length")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                                double repeatDistance = atof(cs = xercesc::XMLString::transcode(objectChildElement->getAttribute(t12 = xercesc::XMLString::transcode("distance")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                                roadObject->setObjectRepeat(repeatS, repeatLength, repeatDistance);
                                            }
                                            else if (xercesc::XMLString::compareIString(objectChildElement->getTagName(), t11 = xercesc::XMLString::transcode("outline")) == 0)
                                            {
                                                xercesc::DOMNodeList *outlineChildrenList = objectChildElement->getChildNodes();
                                                xercesc::DOMElement *outlineChildElement;
                                                RoadOutline *ro = new RoadOutline();
                                                for (unsigned int childIndex = 0; childIndex < outlineChildrenList->getLength(); ++childIndex)
                                                {
                                                    outlineChildElement = dynamic_cast<xercesc::DOMElement *>(outlineChildrenList->item(childIndex));
                                                    if (outlineChildElement)
                                                    {
                                                        float height, u, v, z;
                                                        height = atof(cs = xercesc::XMLString::transcode(outlineChildElement->getAttribute(t12 = xercesc::XMLString::transcode("height")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                                        u = atof(cs = xercesc::XMLString::transcode(outlineChildElement->getAttribute(t12 = xercesc::XMLString::transcode("u")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                                        v = atof(cs = xercesc::XMLString::transcode(outlineChildElement->getAttribute(t12 = xercesc::XMLString::transcode("v")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                                        z = atof(cs = xercesc::XMLString::transcode(outlineChildElement->getAttribute(t12 = xercesc::XMLString::transcode("z")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                                        ro->push_back(RoadCornerLocal(height, u, v, z));
                                                    }
                                                }
                                                roadObject->setOutline(ro);
                                            }
                                            else if (xercesc::XMLString::compareIString(objectChildElement->getTagName(), t13 = xercesc::XMLString::transcode("userData")) == 0)
                                            {
                                                std::string code = cs = xercesc::XMLString::transcode(objectChildElement->getAttribute(t12 = xercesc::XMLString::transcode("code")));  xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                                if (code == "modelFile")
                                                {
                                                    roadObject->setFileName(cs = xercesc::XMLString::transcode(objectChildElement->getAttribute(t12 = xercesc::XMLString::transcode("value")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                                }
                                                else
                                                {
                                                    roadObject->setTexture(cs = xercesc::XMLString::transcode(objectChildElement->getAttribute(t12 = xercesc::XMLString::transcode("value")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                                }
                                            }
											xercesc::XMLString::release(&t10); xercesc::XMLString::release(&t11);
											xercesc::XMLString::release(&t13);
                                        }
                                    }
                                }
                            }
                            // Parse crosswalk
                            // <OpenDRIVE><road><objects><crosswalk> (optional, unlimited)
                            else if (objectElement && xercesc::XMLString::compareIString(objectElement->getTagName(), t10 = xercesc::XMLString::transcode("crosswalk")) == 0)
                            {
                                std::string id = ""; // required
                                std::string name = ""; // required
                                double s = 0.0; // required
                                double length = 0.0; // required
                                double crossProb = 0.5; // optional
                                double resetTime = 20.0; // optional
                                std::string type = "crosswalk"; // optional
                                int debug = 0; // optional

                                // Get attributes
                                std::string tmp = "";
                                id = cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("id"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                name = cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("name"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                s = atof(cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("s")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                length = atof(cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("length")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                tmp = cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("crossProb"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                if (tmp.length() > 0)
                                    crossProb = atof(tmp.c_str());
                                tmp = cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("resetTime"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                if (tmp.length() > 0)
                                    resetTime = atof(tmp.c_str());
                                tmp = cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("type"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                if (tmp.length() > 0)
                                    type = std::string(tmp);
                                tmp = cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("debugLvl"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                if (tmp.length() > 0)
                                    debug = atoi(tmp.c_str());

                                // Create new crosswalk object and add to roadsystem
                                Crosswalk *crosswalk = new Crosswalk(id, name, s, length, crossProb, resetTime, type, debug);
                                road->addCrosswalk(crosswalk);

                                // Parse children of crosswalk
                                // <validity/> (optional, unlimited)
                                xercesc::DOMNodeList *crosswChildrenList = objectElement->getChildNodes();
                                xercesc::DOMElement *crosswChildElement;
                                for (unsigned int childIndex = 0; childIndex < crosswChildrenList->getLength(); ++childIndex)
                                {
                                    crosswChildElement = dynamic_cast<xercesc::DOMElement *>(crosswChildrenList->item(childIndex));
                                    if (crosswChildElement && xercesc::XMLString::compareIString(crosswChildElement->getTagName(), t11 = xercesc::XMLString::transcode("validity")) == 0)
                                    {
                                        // Restrict crosswalk's span to a specific lane range (default is entire lane cross-section)
                                        int fromLane = INT_MIN;
                                        int toLane = INT_MAX;

                                        // Get attributes
                                        fromLane = atoi(cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("fromLane")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                        toLane = atoi(cs = xercesc::XMLString::transcode(objectElement->getAttribute(t12 = xercesc::XMLString::transcode("toLane")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);

                                        // Apply validity info to crosswalk
                                        crosswalk->setValidity(fromLane, toLane);
                                    }
									xercesc::XMLString::release(&t11);
                                }
                            }
							xercesc::XMLString::release(&t9); xercesc::XMLString::release(&t10);
                        }
                    }

                    else if (xercesc::XMLString::compareIString(roadChildElement->getTagName(), t9 = xercesc::XMLString::transcode("signals")) == 0)
                    {
                        xercesc::DOMNodeList *signalsChildrenList = roadChildElement->getChildNodes();
                        xercesc::DOMElement *signalElement;
                        for (unsigned int childIndex = 0; childIndex < signalsChildrenList->getLength(); ++childIndex)
                        {
                            signalElement = dynamic_cast<xercesc::DOMElement *>(signalsChildrenList->item(childIndex));
                            if (signalElement && xercesc::XMLString::compareIString(signalElement->getTagName(), t10 = xercesc::XMLString::transcode("signal")) == 0)
                            {
                                std::string id(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("id")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                std::string name(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("name")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);

                                double s = atof(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("s")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double t = atof(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("t")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double size = atof(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("size")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);

                                std::string dynamicString(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("dynamic")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                bool dynamic = false;
                                if (dynamicString == "yes")
                                {
                                    dynamic = true;
                                }
                                else if (dynamicString == "no")
                                {
                                    dynamic = false;
                                }

                                std::string orientationString(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("orientation")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                RoadSignal::OrientationType orientation = RoadSignal::BOTH_DIRECTIONS;
                                if (orientationString == "+")
                                {
                                    orientation = RoadSignal::POSITIVE_TRACK_DIRECTION;
                                }
                                else if (orientationString == "-")
                                {
                                    orientation = RoadSignal::NEGATIVE_TRACK_DIRECTION;
                                }

                                double zOffset = atof(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("zOffset")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                std::string country(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("country")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                int type = atoi(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("type")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                int subtype = atoi(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("subtype")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                std::string subclass(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("subclass")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double value = atof(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("value")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double hdg = atof(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("hOffset")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double pitch = atof(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("pitch")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double roll = atof(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("roll")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
								std::string unit(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("unit")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
								std::string text(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("text")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
								double width = atof(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("width")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
								double height = atof(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("height")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);

                                xercesc::DOMNodeList *signalChildrenList = signalElement->getChildNodes();
                                xercesc::DOMElement *signalChildElement;
                                for (unsigned int childIndex = 0; childIndex < signalChildrenList->getLength(); ++childIndex)
                                {
                                    signalChildElement = dynamic_cast<xercesc::DOMElement *>(signalChildrenList->item(childIndex));
                                    if (signalChildElement)
                                    {
                                        if (xercesc::XMLString::compareIString(signalChildElement->getTagName(), t10 = xercesc::XMLString::transcode("userData")) == 0)
                                        {
                                            std::string code = cs = xercesc::XMLString::transcode(signalChildElement->getAttribute(t12 = xercesc::XMLString::transcode("code")));  xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                            if (code == "size")
                                            {
                                                size = atof(cs = xercesc::XMLString::transcode(signalChildElement->getAttribute(t12 = xercesc::XMLString::transcode("value")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                            }
                                            else if (code == "subclass")
                                            {
                                                subclass = cs = xercesc::XMLString::transcode(signalChildElement->getAttribute(t12 = xercesc::XMLString::transcode("value"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                            } 
                                        }
										xercesc::XMLString::release(&t10);
                                    }
                                }

                                RoadSignal *roadSignal;
                                //if(type == 1000001) {
                                bool isTrafficSignal = false;
                                if ((country == "China") && (type <= 3))
                                {
                                    if (subclass.size() != 0)
                                    {
                                        std::string::const_iterator it = subclass.begin();
                                        while (it != subclass.end() && isdigit(*it)) ++it;
                                        if (it == subclass.end())
                                        {
                                            isTrafficSignal = true;
                                        }
                                    }
                                }

                                if ((type == 1000001 || type == 1000002) || isTrafficSignal)
                                {
                                    roadSignal = new TrafficLightSignal(id, name, s, t, dynamic, orientation, zOffset, country,
                                                                        type, subtype, subclass, size, value, hdg, pitch, roll, unit, text, width, height);
                                    //std::cout << "Adding traffic light signal id=" << id << std::endl;
                                }
                                else
                                {
									if (name == "")
									{
										if ((type != -1) )
										{
											if (subclass != "")
											{
												if (subtype != -1)
												{
													name = std::to_string(type) + "." + subclass + "-" + std::to_string(subtype);
												}
												else
												{
													name = std::to_string(type) + "." + subclass;
												}
											}
											else if (subtype != -1) 
											{
												name = std::to_string(type) + "-" + std::to_string(subtype);
											}
											else
											{
												name = std::to_string(type);
											}
										}

									}
                                    roadSignal = new RoadSignal(id, name, s, t, dynamic, orientation, zOffset, country,
                                                                type, subtype, subclass, size, value, hdg, pitch, roll, unit, text, width, height);
                                    //std::cout << "Adding road signal id=" << id << std::endl;
                                }
                                road->addRoadSignal(roadSignal);

                                addRoadSignal(roadSignal);
                            }
							xercesc::XMLString::release(&t10);
                        }
                    }

                    else if (xercesc::XMLString::compareIString(roadChildElement->getTagName(), t10 = xercesc::XMLString::transcode("sensors")) == 0)
                    {
                        xercesc::DOMNodeList *signalsChildrenList = roadChildElement->getChildNodes();
                        xercesc::DOMElement *signalElement;
                        for (unsigned int childIndex = 0; childIndex < signalsChildrenList->getLength(); ++childIndex)
                        {
                            signalElement = dynamic_cast<xercesc::DOMElement *>(signalsChildrenList->item(childIndex));
                            if (signalElement && xercesc::XMLString::compareIString(signalElement->getTagName(), t11 = xercesc::XMLString::transcode("sensor")) == 0)
                            {
                                std::string id(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("id")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);

                                double s = atof(cs = xercesc::XMLString::transcode(signalElement->getAttribute(t12 = xercesc::XMLString::transcode("s")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);

                                RoadSensor *roadSensor = new RoadSensor(id, s);
                                road->addRoadSensor(roadSensor);
                                addRoadSensor(roadSensor);
                            }
							xercesc::XMLString::release(&t11);
                        }
                    }

                    else if (xercesc::XMLString::compareIString(roadChildElement->getTagName(), t11 = xercesc::XMLString::transcode("surface")) == 0)
                    {
                        xercesc::DOMNodeList *surfaceChildrenList = roadChildElement->getChildNodes();
                        xercesc::DOMElement *surfaceChildElement;
                        for (unsigned int childIndex = 0; childIndex < surfaceChildrenList->getLength(); ++childIndex)
                        {
                            surfaceChildElement = dynamic_cast<xercesc::DOMElement *>(surfaceChildrenList->item(childIndex));
                            if (surfaceChildElement && xercesc::XMLString::compareIString(surfaceChildElement->getTagName(), t13 = xercesc::XMLString::transcode("CRG")) == 0)
                            {
                                std::string file(cs = xercesc::XMLString::transcode(surfaceChildElement->getAttribute(t12 = xercesc::XMLString::transcode("file")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double sStart = atof(cs = xercesc::XMLString::transcode(surfaceChildElement->getAttribute(t12 = xercesc::XMLString::transcode("sStart")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double sEnd = atof(cs = xercesc::XMLString::transcode(surfaceChildElement->getAttribute(t12 = xercesc::XMLString::transcode("sEnd")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                std::string orientString(cs = xercesc::XMLString::transcode(surfaceChildElement->getAttribute(t12 = xercesc::XMLString::transcode("orientation")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                OpenCRGSurface::SurfaceOrientation orientation = (orientString == "opposite") ? OpenCRGSurface::OPPOSITE : OpenCRGSurface::SAME;
                                double sOffset = atof(cs = xercesc::XMLString::transcode(surfaceChildElement->getAttribute(t12 = xercesc::XMLString::transcode("sOffset")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double tOffset = atof(cs = xercesc::XMLString::transcode(surfaceChildElement->getAttribute(t12 = xercesc::XMLString::transcode("tOffset")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double zOffset = atof(cs = xercesc::XMLString::transcode(surfaceChildElement->getAttribute(t12 = xercesc::XMLString::transcode("zOffset")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                std::string zScaleString(cs = xercesc::XMLString::transcode(surfaceChildElement->getAttribute(t12 = xercesc::XMLString::transcode("zScale")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double zScale = 1.0;
                                if (!zScaleString.empty())
                                {
                                    zScale = atof(zScaleString.c_str());
                                }
                                double hOffset = atof(cs = xercesc::XMLString::transcode(surfaceChildElement->getAttribute(t12 = xercesc::XMLString::transcode("hOffset")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);

                                //crgSurface *surface = new crgSurface(file, sStart, sEnd, orientation, sOffset, tOffset, zOffset, zScale);
                                OpenCRGSurface *surface = new OpenCRGSurface(file, sStart, sEnd, orientation, sOffset, tOffset, zOffset, zScale, hOffset);
                                road->addRoadSurface(surface, sStart, sEnd);
                            }
                        }
                    }
					xercesc::XMLString::release(&t2);
					xercesc::XMLString::release(&t3);
					xercesc::XMLString::release(&t4);
					xercesc::XMLString::release(&t5);
					xercesc::XMLString::release(&t6);
					xercesc::XMLString::release(&t7);
					xercesc::XMLString::release(&t8);
					xercesc::XMLString::release(&t9);
					xercesc::XMLString::release(&t10);
					xercesc::XMLString::release(&t11);
                }
            }

            else if (documentChildElement && xercesc::XMLString::compareIString(documentChildElement->getTagName(), t2 = xercesc::XMLString::transcode("header")) == 0)
            {
                header.date = cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t3 = xercesc::XMLString::transcode("date"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                header.name = cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t3 = xercesc::XMLString::transcode("name"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                header.north = atof(cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t3 = xercesc::XMLString::transcode("north")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                header.east = atof(cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t3 = xercesc::XMLString::transcode("east")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                header.south = atof(cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t3 = xercesc::XMLString::transcode("south")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                header.west = atof(cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t3 = xercesc::XMLString::transcode("west")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                header.revMajor = atoi(cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t3 = xercesc::XMLString::transcode("atoi")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                header.revMinor = atoi(cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t3 = xercesc::XMLString::transcode("atoi")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                header.version = atof(cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t3 = xercesc::XMLString::transcode("version")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                header.xoffset = atof(cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t3 = xercesc::XMLString::transcode("xoffset")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                header.yoffset = atof(cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t3 = xercesc::XMLString::transcode("yoffset")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                header.zoffset = atof(cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t3 = xercesc::XMLString::transcode("zoffset")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
            }
			xercesc::XMLString::release(&t1);
			xercesc::XMLString::release(&t2);
        }
        for (unsigned int childIndex = 0; childIndex < documentChildrenList->getLength(); ++childIndex)
        {
            documentChildElement = dynamic_cast<xercesc::DOMElement *>(documentChildrenList->item(childIndex));
            if (documentChildElement && xercesc::XMLString::compareIString(documentChildElement->getTagName(), t3 = xercesc::XMLString::transcode("controller")) == 0)
            {
                std::string controllerIdString = cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t12 = xercesc::XMLString::transcode("id"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                std::string controllerNameString = cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t12 = xercesc::XMLString::transcode("name"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                std::string script = cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t12 = xercesc::XMLString::transcode("script"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                double cycleTime = atof(cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t12 = xercesc::XMLString::transcode("cycleTime")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);

                Controller *controller = new Controller(controllerIdString, controllerNameString, script, cycleTime);
                

                xercesc::DOMNodeList *controllerChildrenList = documentChildElement->getChildNodes();
                xercesc::DOMElement *controllerChildElement;
                for (unsigned int childIndex = 0; childIndex < controllerChildrenList->getLength(); ++childIndex)
                {
                    controllerChildElement = dynamic_cast<xercesc::DOMElement *>(controllerChildrenList->item(childIndex));

                    if (controllerChildElement && xercesc::XMLString::compareIString(controllerChildElement->getTagName(), t4 = xercesc::XMLString::transcode("userData")) == 0)
                    {
                        std::string code = cs = xercesc::XMLString::transcode(controllerChildElement->getAttribute(t12 = xercesc::XMLString::transcode("code")));  xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        if (code == "cycleTime")
                        {
                            cycleTime = atof(cs = xercesc::XMLString::transcode(controllerChildElement->getAttribute(t12 = xercesc::XMLString::transcode("value")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                            if (script != "")
                            {
                                controller->setScriptParams(script, cycleTime);
                            }
                        }
                        else
                        {
                            script = cs = xercesc::XMLString::transcode(controllerChildElement->getAttribute(t12 = xercesc::XMLString::transcode("value"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        }
                    }

                    else if (controller->isScriptInitialized() && controllerChildElement && xercesc::XMLString::compareIString(controllerChildElement->getTagName(), t5 = xercesc::XMLString::transcode("control")) == 0)
                    {
                        std::string signalId = cs = xercesc::XMLString::transcode(controllerChildElement->getAttribute(t12 = xercesc::XMLString::transcode("signalId"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        std::string type = cs = xercesc::XMLString::transcode(controllerChildElement->getAttribute(t12 = xercesc::XMLString::transcode("type"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);

                        std::map<std::string, RoadSignal *>::iterator signalIt = signalIdMap.find(signalId);
                        if (signalIt != signalIdMap.end())
                        {
                            //controller->addControl(signalIt->second, type);
                            TrafficLightSignal *trafficLightSignal = dynamic_cast<TrafficLightSignal *>(signalIt->second);
                            if (trafficLightSignal)
                            {
                                Control *control = new Control(trafficLightSignal, type);
                                controller->addControl(control);
                            }
                            else
                            {
                                std::cout << "Parse OpenDRIVE: signal not a traffic light, id=" << signalId << std::endl;
                            }
                        }
                        else
                        {
                            std::cout << "Parse OpenDRIVE: Could not find signal with id=" << signalId << std::endl;
                        }
                    }
                    if (controller->isScriptInitialized() && controllerChildElement && xercesc::XMLString::compareIString(controllerChildElement->getTagName(), t6 = xercesc::XMLString::transcode("trigger")) == 0)
                    {
                        std::string sensorId = cs = xercesc::XMLString::transcode(controllerChildElement->getAttribute(t12 = xercesc::XMLString::transcode("sensorId"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        std::string function = cs = xercesc::XMLString::transcode(controllerChildElement->getAttribute(t12 = xercesc::XMLString::transcode("function"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        std::map<std::string, RoadSensor *>::iterator sensorIt = sensorIdMap.find(sensorId);
                        if (sensorIt != sensorIdMap.end())
                        {
                            controller->addTrigger(sensorIt->second, function);
                        }
                        else
                        {
                            std::cout << "Parse OpenDRIVE: Could not find sensor with id=" << sensorId << std::endl;
                        }
                    }
					xercesc::XMLString::release(&t4);
					xercesc::XMLString::release(&t5);
					xercesc::XMLString::release(&t6);
                }
                if (controller->isScriptInitialized())
                {
                    addController(controller);
                    controller->init();
                }

            }
			xercesc::XMLString::release(&t3);
        }
        for (unsigned int childIndex = 0; childIndex < documentChildrenList->getLength(); ++childIndex)
        {
            documentChildElement = dynamic_cast<xercesc::DOMElement *>(documentChildrenList->item(childIndex));
            if (documentChildElement && xercesc::XMLString::compareIString(documentChildElement->getTagName(), t3 = xercesc::XMLString::transcode("junction")) == 0)
            {
                std::string junctionId(cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t12 = xercesc::XMLString::transcode("id")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                std::string junctionName(cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t12 = xercesc::XMLString::transcode("name")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                Junction *junction = new Junction(junctionId, junctionName);
                addJunction(junction);

                xercesc::DOMNodeList *junctionChildrenList = documentChildElement->getChildNodes();
                xercesc::DOMElement *junctionChildElement;
                for (unsigned int childIndex = 0; childIndex < junctionChildrenList->getLength(); ++childIndex)
                {
                    junctionChildElement = dynamic_cast<xercesc::DOMElement *>(junctionChildrenList->item(childIndex));
                    if (!junctionChildElement)
                    {
                    }
                    else if (xercesc::XMLString::compareIString(junctionChildElement->getTagName(), t4 = xercesc::XMLString::transcode("connection")) == 0)
                    {
                        std::string connectionId(cs = xercesc::XMLString::transcode(junctionChildElement->getAttribute(t12 = xercesc::XMLString::transcode("id")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        std::string inRoadString(cs = xercesc::XMLString::transcode(junctionChildElement->getAttribute(t12 = xercesc::XMLString::transcode("incomingRoad")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        std::string connPathString(cs = xercesc::XMLString::transcode(junctionChildElement->getAttribute(t12 = xercesc::XMLString::transcode("connectingRoad")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        std::string contactPoint(cs = xercesc::XMLString::transcode(junctionChildElement->getAttribute(t12 = xercesc::XMLString::transcode("contactPoint")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        std::string numeratorString(cs = xercesc::XMLString::transcode(junctionChildElement->getAttribute(t12 = xercesc::XMLString::transcode("numerator")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        xercesc::DOMNodeList *childrenList = junctionChildElement->getChildNodes();
                        xercesc::DOMElement *childElement;
                        for (unsigned int childIndex = 0; childIndex < childrenList->getLength(); ++childIndex)
                        {
                            childElement = dynamic_cast<xercesc::DOMElement *>(childrenList->item(childIndex));
                            if (childElement && (xercesc::XMLString::compareIString(childElement->getTagName(), t5 = xercesc::XMLString::transcode("userData")) == 0))
                            {
                                numeratorString = cs = xercesc::XMLString::transcode(childElement->getAttribute(t12 = xercesc::XMLString::transcode("value")));  xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                            }
							xercesc::XMLString::release(&t5);
                        }
                        double numerator;
                        if (numeratorString.empty())
                        {
                            numerator = 1.0;
                        }
                        else
                        {
                            numerator = atof(numeratorString.c_str());
                        }

                        std::map<std::string, Road *>::iterator inRoadIt = roadIdMap.find(inRoadString);
                        if (inRoadIt == roadIdMap.end())
                        {
                            std::cerr << "Junction id " << junction->getId() << ": Connection id: " << connectionId << ": No incoming road defined with id " << inRoadString << "..." << std::endl;
                        }
                        else
                        {
                            Road *inRoad = dynamic_cast<Road *>((*inRoadIt).second);
                            if (inRoad == NULL)
                            {
                                std::cerr << "Junction id " << junction->getId() << ": Connection id: " << connectionId << ": Incoming Road with id declaration " << inRoadString << " not of type road..." << std::endl;
                            }
                            else
                            {
                                std::map<std::string, Road *>::iterator connPathIt = roadIdMap.find(connPathString);
                                if (connPathIt == roadIdMap.end())
                                {
                                    std::cerr << "Junction id " << junction->getId() << ": Connection id: " << connectionId << ": No connecting path defined with id " << connPathString << "..." << std::endl;
                                }
                                else
                                {
                                    Road *connPath = dynamic_cast<Road *>((*connPathIt).second);
                                    if (connPath == NULL)
                                    {
                                        std::cerr << "Junction id " << junction->getId() << ": Connection id: " << connectionId << ": Connecting Path with id declaration " << connPathString << " not of type road..." << std::endl;
                                    }
                                    else
                                    {
                                        int direction = 1;
                                        if (contactPoint == "start")
                                        {
                                            direction = 1;
                                        }
                                        else if (contactPoint == "end")
                                        {
                                            direction = -1;
                                        }
                                        else
                                        {
                                            std::cerr << "Junction id " << junction->getId() << ": Connection id: " << connectionId << ": No direction defined..." << std::endl;
                                        }

                                        PathConnection *conn = new PathConnection(connectionId, inRoad, connPath, direction, numerator);
                                        junction->addPathConnection(conn);

                                        xercesc::DOMNodeList *connectionChildrenList = junctionChildElement->getChildNodes();
                                        xercesc::DOMElement *connectionChildElement;
                                        for (unsigned int childIndex = 0; childIndex < connectionChildrenList->getLength(); ++childIndex)
                                        {
                                            connectionChildElement = dynamic_cast<xercesc::DOMElement *>(connectionChildrenList->item(childIndex));
                                            if (!connectionChildElement)
                                            {
                                            }
                                            else if (xercesc::XMLString::compareIString(connectionChildElement->getTagName(), t5 = xercesc::XMLString::transcode("laneLink")) == 0)
                                            {
                                                int from = atoi(cs = xercesc::XMLString::transcode(connectionChildElement->getAttribute(t12 = xercesc::XMLString::transcode("from")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                                int to = atoi(cs = xercesc::XMLString::transcode(connectionChildElement->getAttribute(t12 = xercesc::XMLString::transcode("to")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                                conn->addLaneLink(from, to);
                                            }
											xercesc::XMLString::release(&t5);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else if (xercesc::XMLString::compareIString(junctionChildElement->getTagName(), t5 = xercesc::XMLString::transcode("controller")) == 0)
                    {
                        std::string controllerId(cs = xercesc::XMLString::transcode(junctionChildElement->getAttribute(t12 = xercesc::XMLString::transcode("id")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        std::string controlType(cs = xercesc::XMLString::transcode(junctionChildElement->getAttribute(t12 = xercesc::XMLString::transcode("type")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);

                        std::map<std::string, Controller *>::iterator controllerIt = controllerIdMap.find(controllerId);
                        if (controllerIt != controllerIdMap.end())
                        {
                            junction->addJunctionController(controllerIt->second, controlType);
                        }
                    }
					xercesc::XMLString::release(&t4);
					xercesc::XMLString::release(&t5);
                }
            }
			xercesc::XMLString::release(&t3);
        }
        for (unsigned int childIndex = 0; childIndex < documentChildrenList->getLength(); ++childIndex)
        {
            documentChildElement = dynamic_cast<xercesc::DOMElement *>(documentChildrenList->item(childIndex));
            if (documentChildElement && xercesc::XMLString::compareIString(documentChildElement->getTagName(), t3 = xercesc::XMLString::transcode("fiddleyard")) == 0)
            {
                std::string fiddleyardId(cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t12 = xercesc::XMLString::transcode("id")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                std::string fiddleyardName(cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t12 = xercesc::XMLString::transcode("name")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                Fiddleyard *fiddleyard = new Fiddleyard(fiddleyardId, fiddleyardName);
                addFiddleyard(fiddleyard);

                xercesc::DOMNodeList *fiddleyardChildrenList = documentChildElement->getChildNodes();
                xercesc::DOMElement *fiddleyardChildElement;
                for (unsigned int childIndex = 0; childIndex < fiddleyardChildrenList->getLength(); ++childIndex)
                {
                    fiddleyardChildElement = dynamic_cast<xercesc::DOMElement *>(fiddleyardChildrenList->item(childIndex));
                    if (!fiddleyardChildElement)
                    {
                    }
                    else if (xercesc::XMLString::compareIString(fiddleyardChildElement->getTagName(), t4 = xercesc::XMLString::transcode("link")) == 0)
                    {
                        std::string type(cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("elementType")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        std::string id(cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("elementId")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        std::string contactPoint(cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("contactPoint")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);

                        Tarmac *tarmac = NULL;
                        if (type == "road")
                        {
                            std::map<std::string, Road *>::iterator roadMapIt = roadIdMap.find(id);
                            if (roadMapIt == roadIdMap.end())
                            {
                                std::cerr << "Fiddleyard " << fiddleyard->getName() << ": Unknown link road id: " << id << "... Ignoring..." << std::endl;
                            }
                            else
                            {
                                tarmac = dynamic_cast<Tarmac *>((*roadMapIt).second);
                            }
                        }
                        else if (type == "junction")
                        {
                            std::map<std::string, Junction *>::iterator junctionMapIt = junctionIdMap.find(id);
                            if (junctionMapIt == junctionIdMap.end())
                            {
                                std::cerr << "Fiddleyard " << fiddleyard->getName() << ": Unknown link junction id: " << id << "... Ignoring..." << std::endl;
                            }
                            else
                            {
                                tarmac = dynamic_cast<Tarmac *>((*junctionMapIt).second);
                            }
                        }
                        if (!tarmac)
                        {
                            std::cerr << "Fiddleyard " << fiddleyard->getName() << ", link tag: element id: " << id << " is not of type tarmac..." << std::endl;
                        }
                        else
                        {
                            int direction = 1;
                            if (contactPoint == "start")
                            {
                                direction = 1;
                            }
                            else if (contactPoint == "end")
                            {
                                direction = -1;
                            }
                            else
                            {
                                std::cerr << "Fiddleyard " << fiddleyard->getName() << ": link id: " << id << ": No direction defined..." << std::endl;
                            }
                            fiddleyard->setTarmacConnection(new TarmacConnection(tarmac, direction));
                        }
                    }

                    else if (xercesc::XMLString::compareIString(fiddleyardChildElement->getTagName(), t5 = xercesc::XMLString::transcode("source")) == 0)
                    {
                        std::string id(cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("id")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        int lane = atoi(cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("lane")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        double starttime = atof(cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("startTime")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        double repeattime = atof(cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("repeatTime")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        double vel = atof(cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("velocity")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        double velDev = atof(cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("velocityDeviance")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        VehicleSource *source = new VehicleSource(id, lane, starttime, repeattime, vel, velDev);
                        fiddleyard->addVehicleSource(source);

                        xercesc::DOMNodeList *sourceChildrenList = fiddleyardChildElement->getChildNodes();
                        xercesc::DOMElement *sourceChildElement;
                        for (unsigned int childIndex = 0; childIndex < sourceChildrenList->getLength(); ++childIndex)
                        {
                            sourceChildElement = dynamic_cast<xercesc::DOMElement *>(sourceChildrenList->item(childIndex));
                            if (sourceChildElement && xercesc::XMLString::compareIString(sourceChildElement->getTagName(), t6 = xercesc::XMLString::transcode("vehicle")) == 0)
                            {
                                std::string id(cs = xercesc::XMLString::transcode(sourceChildElement->getAttribute(t12 = xercesc::XMLString::transcode("id")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double numerator = atof(cs = xercesc::XMLString::transcode(sourceChildElement->getAttribute(t12 = xercesc::XMLString::transcode("numerator")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                source->addVehicleRatio(id, numerator);
                            }
							xercesc::XMLString::release(&t6);
                        }
                    }
                    else if (xercesc::XMLString::compareIString(fiddleyardChildElement->getTagName(), t6 = xercesc::XMLString::transcode("sink")) == 0)
                    {
                        std::string id(cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("id")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        int lane = atoi(cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("lane")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        fiddleyard->addVehicleSink(new VehicleSink(id, lane));
                    }
					xercesc::XMLString::release(&t4);
					xercesc::XMLString::release(&t5);
					xercesc::XMLString::release(&t6);
                }
            }
			xercesc::XMLString::release(&t3);
        }
        // CarPool
        for (unsigned int childIndex = 0; childIndex < documentChildrenList->getLength(); ++childIndex)
        {
            documentChildElement = dynamic_cast<xercesc::DOMElement *>(documentChildrenList->item(childIndex));
            if (documentChildElement && xercesc::XMLString::compareString(documentChildElement->getTagName(), t3 = xercesc::XMLString::transcode("carpool")) == 0)
            {
                //std::string carpoolId(xercesc::XMLString::transcode(documentChildElement->getAttribute(xercesc::XMLString::transcode("id"))));
                std::string carpoolName(cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t12 = xercesc::XMLString::transcode("name")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                Carpool *carpool = Carpool::Instance();
                //addFiddleyard(fiddleyard);
                xercesc::DOMNodeList *carpoolChildrenList = documentChildElement->getChildNodes();
                xercesc::DOMElement *carpoolChildElement;
                int dir = 1;
                for (unsigned int childIndex = 0; childIndex < carpoolChildrenList->getLength(); ++childIndex)
                {
                    carpoolChildElement = dynamic_cast<xercesc::DOMElement *>(carpoolChildrenList->item(childIndex));
                    if (!carpoolChildElement)
                    {
                    }
                    else if (xercesc::XMLString::compareString(carpoolChildElement->getTagName(), t4 = xercesc::XMLString::transcode("pool")) == 0)
                    {
                        std::string poolId(cs = xercesc::XMLString::transcode(carpoolChildElement->getAttribute(t12 = xercesc::XMLString::transcode("id")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        std::string poolName(cs = xercesc::XMLString::transcode(carpoolChildElement->getAttribute(t12 = xercesc::XMLString::transcode("name")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        //double repeattime = atof(cs = xercesc::XMLString::transcode(carpoolChildElement->getAttribute(t12 = xercesc::XMLString::transcode("repeatTime")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        double vel = atof(cs = xercesc::XMLString::transcode(carpoolChildElement->getAttribute(t12 = xercesc::XMLString::transcode("velocity")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        double velDev = atof(cs = xercesc::XMLString::transcode(carpoolChildElement->getAttribute(t12 = xercesc::XMLString::transcode("velocityDeviance")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        double poolNumerator = atof(cs = xercesc::XMLString::transcode(carpoolChildElement->getAttribute(t12 = xercesc::XMLString::transcode("numerator")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        Pool *pool = new Pool(poolId, poolName, vel, velDev, dir, poolNumerator);
                        dir *= -1;

                        xercesc::DOMNodeList *poolChildrenList = carpoolChildElement->getChildNodes();
                        xercesc::DOMElement *poolChildElement;
                        for (unsigned int childIndex = 0; childIndex < poolChildrenList->getLength(); ++childIndex)
                        {
                            poolChildElement = dynamic_cast<xercesc::DOMElement *>(poolChildrenList->item(childIndex));
                            if (poolChildElement && xercesc::XMLString::compareString(poolChildElement->getTagName(), t5 = xercesc::XMLString::transcode("vehicle")) == 0)
                            {
                                std::string id(cs = xercesc::XMLString::transcode(poolChildElement->getAttribute(t12 = xercesc::XMLString::transcode("id")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double numerator = atof(cs = xercesc::XMLString::transcode(poolChildElement->getAttribute(t12 = xercesc::XMLString::transcode("numerator")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                //std::cout << "ROADSYSTEM -- id: " << id << " numerator: " << numerator << std::endl;
                                pool->addVehicle(id, numerator);
                            }
							xercesc::XMLString::release(&t5);
                        }

                        carpool->addPool(pool);
                        carpool->addPoolRatio(poolId, poolNumerator);
                    }
					xercesc::XMLString::release(&t4);
                }
            }
			xercesc::XMLString::release(&t3);
        }
        //

        // Pedestrian fiddleyards (sources/sinks for pedestrians)
        for (unsigned int childIndex = 0; childIndex < documentChildrenList->getLength(); ++childIndex)
        {
            documentChildElement = dynamic_cast<xercesc::DOMElement *>(documentChildrenList->item(childIndex));
            if (documentChildElement && xercesc::XMLString::compareIString(documentChildElement->getTagName(), t3 = xercesc::XMLString::transcode("pedFiddleyard")) == 0)
            {
                std::string fiddleyardId(cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t12 = xercesc::XMLString::transcode("id")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                std::string fiddleyardName(cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t12 = xercesc::XMLString::transcode("name")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                std::string roadId = "road";
                std::string tmp = cs = xercesc::XMLString::transcode(documentChildElement->getAttribute(t12 = xercesc::XMLString::transcode("roadId"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                if (tmp.length() > 0)
                    roadId = std::string(tmp);
                PedFiddleyard *fiddleyard = new PedFiddleyard(fiddleyardId, fiddleyardName, roadId);
                addPedFiddleyard(fiddleyard);

                xercesc::DOMNodeList *fiddleyardChildrenList = documentChildElement->getChildNodes();
                xercesc::DOMElement *fiddleyardChildElement;
                for (unsigned int childIndex = 0; childIndex < fiddleyardChildrenList->getLength(); ++childIndex)
                {
                    fiddleyardChildElement = dynamic_cast<xercesc::DOMElement *>(fiddleyardChildrenList->item(childIndex));
                    if (!fiddleyardChildElement)
                    {
                    }
                    else if (xercesc::XMLString::compareIString(fiddleyardChildElement->getTagName(), t4 = xercesc::XMLString::transcode("source")) == 0)
                    {
                        // Get attributes of this source, provide defaults for any that aren't provided
                        std::string tmp = "";

                        std::string id;
                        tmp = cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("id"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        if (tmp.length() > 0)
                            id = std::string(tmp);

                        double starttime;
                        tmp = cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("startTime"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        if (tmp.length() > 0)
                            starttime = atof(tmp.c_str());
                        else
                            starttime = 0;

                        double repeattime;
                        tmp = cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("repeatTime"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        if (tmp.length() > 0)
                            repeattime = atof(tmp.c_str());
                        else
                            repeattime = 60;

                        double timedev;
                        tmp = cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("timeDeviance"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        if (tmp.length() > 0)
                            timedev = atof(tmp.c_str());
                        else
                            timedev = 0.0;

                        int lane;
                        tmp = cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("lane"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        if (tmp.length() > 0)
                            lane = atoi(tmp.c_str());
                        else
                            lane = 0;

                        int dir;
                        tmp = cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("direction"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        if (tmp.length() > 0)
                            dir = atoi(tmp.c_str());
                        else
                            dir = 0;

                        double sOff;
                        tmp = cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("sOffset"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        if (tmp.length() > 0)
                            sOff = atof(tmp.c_str());
                        else
                            sOff = 0;

                        double vOff;
                        tmp = cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("vOffset"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        if (tmp.length() > 0)
                            vOff = atof(tmp.c_str());
                        else
                            vOff = 0;

                        double vel;
                        tmp = cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("velocity"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        if (tmp.length() > 0)
                            vel = atof(tmp.c_str());
                        else
                            vel = 1;

                        double velDev;
                        tmp = cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("velocityDeviance"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        if (tmp.length() > 0)
                            velDev = atof(tmp.c_str());
                        else
                            velDev = 0;

                        double acc;
                        tmp = cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("acceleration"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        if (tmp.length() > 0)
                            acc = atof(tmp.c_str());
                        else
                            acc = vel / 2;

                        double accDev;
                        tmp = cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("accelerationDeviance"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        if (tmp.length() > 0)
                            accDev = atof(tmp.c_str());
                        else
                            accDev = 0;

                        PedSource *source = new PedSource(fiddleyard, id, starttime, repeattime, timedev, lane, dir, sOff, vOff, vel, velDev, acc, accDev);
                        fiddleyard->addSource(source);

                        xercesc::DOMNodeList *sourceChildrenList = fiddleyardChildElement->getChildNodes();
                        xercesc::DOMElement *sourceChildElement;
                        for (unsigned int childIndex = 0; childIndex < sourceChildrenList->getLength(); ++childIndex)
                        {
                            sourceChildElement = dynamic_cast<xercesc::DOMElement *>(sourceChildrenList->item(childIndex));
                            if (sourceChildElement && xercesc::XMLString::compareIString(sourceChildElement->getTagName(), t5 = xercesc::XMLString::transcode("ped")) == 0)
                            {
                                std::string id(cs = xercesc::XMLString::transcode(sourceChildElement->getAttribute(t12 = xercesc::XMLString::transcode("templateId")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                std::string num(cs = xercesc::XMLString::transcode(sourceChildElement->getAttribute(t12 = xercesc::XMLString::transcode("numerator")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                double numerator = (num.length() > 0) ? atof(num.c_str()) : 1;
                                source->addPedRatio(id, numerator);
                            }
							xercesc::XMLString::release(&t5);
                        }
                    }
                    else if (xercesc::XMLString::compareIString(fiddleyardChildElement->getTagName(), t5 = xercesc::XMLString::transcode("sink")) == 0)
                    {
                        // Get attributes of this source, provide defaults for any that aren't provided
                        std::string tmp = "";

                        std::string id;
                        tmp = cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("id"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        if (tmp.length() > 0)
                            id = std::string(tmp);
                        else
                            id = "0";

                        double prob;
                        tmp = cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("sinkProb"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        if (tmp.length() > 0)
                            prob = atof(tmp.c_str());
                        else
                            prob = 1.0;

                        int lane;
                        tmp = cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("lane"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        if (tmp.length() > 0)
                            lane = atoi(tmp.c_str());
                        else
                            lane = 0;

                        int dir;
                        tmp = cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("direction"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        if (tmp.length() > 0)
                            dir = atoi(tmp.c_str());
                        else
                            dir = 0;

                        double sOff;
                        tmp = cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("sOffset"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        if (tmp.length() > 0)
                            sOff = atof(tmp.c_str());
                        else
                            sOff = 0;

                        double vOff;
                        tmp = cs = xercesc::XMLString::transcode(fiddleyardChildElement->getAttribute(t12 = xercesc::XMLString::transcode("vOffset"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                        if (tmp.length() > 0)
                            vOff = atof(tmp.c_str());
                        else
                            vOff = 0;

                        PedSink *sink = new PedSink(fiddleyard, id, prob, lane, dir, sOff, vOff);
                        fiddleyard->addSink(sink);
                    }
					xercesc::XMLString::release(&t4);
					xercesc::XMLString::release(&t5);
                }
            }
			xercesc::XMLString::release(&t3);
        }

        for (unsigned int roadIndex = 0; roadIndex < roadVector.size(); ++roadIndex)
        {
            Road *road = roadVector[roadIndex];
            std::string junctionId(cs = xercesc::XMLString::transcode(roadDOMMap[road]->getAttribute(t12 = xercesc::XMLString::transcode("junction")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
            std::map<std::string, Junction *>::iterator junctionIt = junctionIdMap.find(junctionId);
            if (junctionId == "-1")
            {
                road->setJunction(NULL);
            }
            else if (junctionIt == junctionIdMap.end())
            {
                std::cerr << "Road " << road->getName() << ": Junction id " << junctionId << " is not in element database... Ignoring..." << std::endl;
                road->setJunction(NULL);
            }
            else
            {
                Junction *junction = dynamic_cast<Junction *>((*junctionIt).second);
                if (junction)
                {
                    road->setJunction(junction);
                }
                else
                {
                    std::cerr << "Road " << road->getName() << ": Declared id for junction " << junctionId << " is not a junction... Ignoring..." << std::endl;
                    road->setJunction(NULL);
                }
            }

            xercesc::DOMNodeList *roadChildrenList = roadDOMMap[road]->getChildNodes();
            xercesc::DOMElement *roadChildElement;
            for (unsigned int childIndex = 0; childIndex < roadChildrenList->getLength(); ++childIndex)
            {
                roadChildElement = dynamic_cast<xercesc::DOMElement *>(roadChildrenList->item(childIndex));
                if (!roadChildElement)
                {
                }
                else if (xercesc::XMLString::compareIString(roadChildElement->getTagName(), t3 = xercesc::XMLString::transcode("link")) == 0)
                {
                    xercesc::DOMNodeList *linkChildrenList = roadChildElement->getChildNodes();
                    xercesc::DOMElement *linkChildElement;
                    for (unsigned int childIndex = 0; childIndex < linkChildrenList->getLength(); ++childIndex)
                    {
                        linkChildElement = dynamic_cast<xercesc::DOMElement *>(linkChildrenList->item(childIndex));
                        if (!linkChildElement)
                        {
                        }
                        else if (xercesc::XMLString::compareIString(linkChildElement->getTagName(), t4 = xercesc::XMLString::transcode("predecessor")) == 0)
                        {
                            std::string type(cs = xercesc::XMLString::transcode(linkChildElement->getAttribute(t12 = xercesc::XMLString::transcode("elementType")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                            std::string id(cs = xercesc::XMLString::transcode(linkChildElement->getAttribute(t12 = xercesc::XMLString::transcode("elementId")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                            std::string contactPoint(cs = xercesc::XMLString::transcode(linkChildElement->getAttribute(t12 = xercesc::XMLString::transcode("contactPoint")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);

                            Tarmac *tarmac = NULL;
                            if (type == "road")
                            {
                                std::map<std::string, Road *>::iterator roadMapIt = roadIdMap.find(id);
                                if (roadMapIt == roadIdMap.end())
                                {
                                    std::cerr << "Road " << road->getName() << ": Unknown predecessor road id: " << id << "... Ignoring..." << std::endl;
                                }
                                else
                                {
                                    tarmac = dynamic_cast<Tarmac *>((*roadMapIt).second);
                                }
                            }
                            else if (type == "junction")
                            {
                                std::map<std::string, Junction *>::iterator junctionMapIt = junctionIdMap.find(id);
                                if (junctionMapIt == junctionIdMap.end())
                                {
                                    std::cerr << "Road " << road->getName() << ": Unknown predecessor junction id: " << id << "... Ignoring..." << std::endl;
                                }
                                else
                                {
                                    tarmac = dynamic_cast<Tarmac *>((*junctionMapIt).second);
                                }
                            }
                            else if (type == "fiddleyard")
                            {
                                std::map<std::string, Fiddleyard *>::iterator fiddleyardMapIt = fiddleyardIdMap.find(id);
                                if (fiddleyardMapIt == fiddleyardIdMap.end())
                                {
                                    std::cerr << "Road " << road->getName() << ": Unknown predecessor fiddleyard id: " << id << "... Ignoring..." << std::endl;
                                }
                                else
                                {
                                    tarmac = dynamic_cast<Tarmac *>(fiddleyardMapIt->second->getFiddleroad());
                                }
                            }
                            if (!tarmac)
                            {
                                std::cerr << "Road " << road->getName() << ", predecessor link tag: element id: " << id << " is not of type tarmac..." << std::endl;
                            }
                            else
                            {
                                int direction = 1;
                                if (contactPoint == "start")
                                {
                                    direction = 1;
                                }
                                else if (contactPoint == "end")
                                {
                                    direction = -1;
                                }
                                else if(type == "road")
                                {
                                    std::cerr << "Road " << road->getName() << ": Predecessor id: " << id << ": No direction defined..." << std::endl;
                                }
                                road->setPredecessorConnection(new TarmacConnection(tarmac, direction));
                            }
                        }
                        else if (xercesc::XMLString::compareIString(linkChildElement->getTagName(), t5 = xercesc::XMLString::transcode("successor")) == 0)
                        {
                            std::string type(cs = xercesc::XMLString::transcode(linkChildElement->getAttribute(t12 = xercesc::XMLString::transcode("elementType")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                            std::string id(cs = xercesc::XMLString::transcode(linkChildElement->getAttribute(t12 = xercesc::XMLString::transcode("elementId")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                            std::string contactPoint(cs = xercesc::XMLString::transcode(linkChildElement->getAttribute(t12 = xercesc::XMLString::transcode("contactPoint")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                            Tarmac *tarmac = NULL;
                            if (type == "road")
                            {
                                std::map<std::string, Road *>::iterator roadMapIt = roadIdMap.find(id);
                                if (roadMapIt == roadIdMap.end())
                                {
                                    std::cerr << "Road " << road->getName() << ": Unknown successor road id: " << id << "... Ignoring..." << std::endl;
                                }
                                else
                                {
                                    tarmac = dynamic_cast<Tarmac *>((*roadMapIt).second);
                                }
                            }
                            else if (type == "junction")
                            {
                                std::map<std::string, Junction *>::iterator junctionMapIt = junctionIdMap.find(id);
                                if (junctionMapIt == junctionIdMap.end())
                                {
                                    std::cerr << "Road " << road->getName() << ": Unknown successor junction id: " << id << "... Ignoring..." << std::endl;
                                }
                                else
                                {
                                    tarmac = dynamic_cast<Tarmac *>((*junctionMapIt).second);
                                }
                            }
                            else if (type == "fiddleyard")
                            {
                                std::map<std::string, Fiddleyard *>::iterator fiddleyardMapIt = fiddleyardIdMap.find(id);
                                if (fiddleyardMapIt == fiddleyardIdMap.end())
                                {
                                    std::cerr << "Road " << road->getName() << ": Unknown successor fiddleyard id: " << id << "... Ignoring..." << std::endl;
                                }
                                else
                                {
                                    tarmac = dynamic_cast<Tarmac *>(fiddleyardMapIt->second->getFiddleroad());
                                }
                            }
                            if (!tarmac)
                            {
                                std::cerr << "Road " << road->getName() << ", successor link tag: element id: " << id << " is not of type tarmac..." << std::endl;
                            }
                            else
                            {
                                int direction = 1;
                                if (contactPoint == "start")
                                {
                                    direction = 1;
                                }
                                else if (contactPoint == "end")
                                {
                                    direction = -1;
                                }
                                else if (type == "road")
                                {
                                    std::cerr << "Road " << road->getName() << ": Successor id: " << id << ": No direction defined..." << std::endl;
                                }
                                road->setSuccessorConnection(new TarmacConnection(tarmac, direction));
                            }
                        }
						xercesc::XMLString::release(&t4);
						xercesc::XMLString::release(&t5);
                    }
                }
				xercesc::XMLString::release(&t3);
            }
        }
    }


    //Analyzing road system
    //analyzeForCrossingJunctionPaths();
}

void RoadSystem::writeOpenDrive(std::string filename)
{
    XodrWriteRoadSystemVisitor *visitor = new XodrWriteRoadSystemVisitor();

    for (unsigned int roadIt = 0; roadIt < roadVector.size(); ++roadIt)
    {
        roadVector[roadIt]->accept(visitor);
    }
    for (unsigned int junctionIt = 0; junctionIt < junctionVector.size(); ++junctionIt)
    {
        junctionVector[junctionIt]->accept(visitor);
    }
    for (unsigned int fiddleyardIt = 0; fiddleyardIt < fiddleyardVector.size(); ++fiddleyardIt)
    {
        fiddleyardVector[fiddleyardIt]->accept(visitor);
    }
    for (unsigned int pedFiddleyardIt = 0; pedFiddleyardIt < pedFiddleyardVector.size(); ++pedFiddleyardIt)
    {
        pedFiddleyardVector[pedFiddleyardIt]->accept(visitor);
    }

    visitor->writeToFile(filename);
}

std::ostream &operator<<(std::ostream &os, RoadSystem *system)
{

    // Debuggin informations
    //os << "Debugging Information: " << std::endl;
    os << "Informations about used road system!" << std::endl;
    os << "Number of Roads: " << system->getNumRoads() << std::endl;
    for (int i = 0; i < system->getNumRoads(); ++i)
    {
        Road *road = system->getRoad(i);
        TarmacConnection *connection;
        os << "Road: " << i << ", name: " << road->getName() << ", id: " << road->getId() << ", length: " << road->getLength();
        connection = road->getPredecessorConnection();
        if (connection)
        {
            os << ", predecessor road id: " << road->getPredecessorConnection()->getConnectingTarmac()->getId() << " and name: " << road->getPredecessorConnection()->getConnectingTarmac()->getName();
        }
        else
        {
            os << ", no predecessor..." << std::endl;
        }
        connection = road->getSuccessorConnection();
        if (connection)
        {
            os << ", successor road id: " << road->getSuccessorConnection()->getConnectingTarmac()->getId() << " and name: " << road->getSuccessorConnection()->getConnectingTarmac()->getName() << std::endl;
        }
        else
        {
            os << ", no successor..." << std::endl;
        }
        Junction *junction = road->getJunction();
        if (junction)
        {
            os << "\tRoad belongs to junction id " << junction->getId() << "..." << std::endl;
        }
        else
        {
            os << "\tRoad belongs to no junction..." << std::endl;
        }

        os << "\t\tLanes of first lane section connect to:" << std::endl;
        LaneSection *firstLaneSection = road->getLaneSection(0.0);
        for (int laneIt = firstLaneSection->getTopRightLane(); laneIt <= firstLaneSection->getTopLeftLane(); ++laneIt)
        {
            os << "\t\t\tLane " << laneIt << " connects to lane " << firstLaneSection->getLanePredecessor(laneIt) << std::endl;
        }

        os << "\t\tLanes of last lane section connect to:" << std::endl;
        LaneSection *lastLaneSection = road->getLaneSection(road->getLength());
        for (int laneIt = lastLaneSection->getTopRightLane(); laneIt <= lastLaneSection->getTopLeftLane(); ++laneIt)
        {
            os << "\t\t\tLane " << laneIt << " connects to lane " << lastLaneSection->getLaneSuccessor(laneIt) << std::endl;
        }
    }

    for (int i = 0; i < system->getNumJunctions(); ++i)
    {
        Junction *junction = system->getJunction(i);
        os << "Junction: " << i << ", name: " << junction->getName() << ", id: " << junction->getId() << std::endl;
        for (int j = 0; j < junction->getNumIncomingRoads(); ++j)
        {
            Road *road = junction->getIncomingRoad(j);
            os << "\tIncoming Road id: " << road->getId() << std::endl;
            PathConnectionSet connSet = junction->getPathConnectionSet(road);
            for (PathConnectionSet::iterator connSetIt = connSet.begin(); connSetIt != connSet.end(); ++connSetIt)
            {
                PathConnection *conn = (*connSetIt);
                Road *path = conn->getConnectingPath();
                os << "\t\tPathConnection: " << conn->getId() << ", Connecting Path id: " << path->getId() << ", relativ heading: " << conn->getAngleDifference() << std::endl;
                LaneConnectionMap laneConnMap = conn->getLaneConnectionMap();
                for (std::map<int, int>::iterator laneConnIt = laneConnMap.begin(); laneConnIt != laneConnMap.end(); ++laneConnIt)
                {
                    os << "\t\t\tLane " << laneConnIt->first << " connects to lane " << laneConnIt->second << "..." << std::endl;
                }
            }
        }
    }

    for (int i = 0; i < system->getNumFiddleyards(); ++i)
    {
        Fiddleyard *fiddleyard = system->getFiddleyard(i);
        os << "Fiddleyard: " << i << ", name: " << fiddleyard->getName() << ", id: " << fiddleyard->getId() << std::endl;
        TarmacConnection *conn = fiddleyard->getTarmacConnection();
        os << "\tTarmac connection id: " << conn->getConnectingTarmac()->getId() << ", direction: " << conn->getConnectingTarmacDirection() << std::endl;
    }

    for (int i = 0; i < system->getNumPedFiddleyards(); ++i)
    {
        PedFiddleyard *fiddleyard = system->getPedFiddleyard(i);
        os << "PedFiddleyard: " << i << ", name: " << fiddleyard->getName() << ", id: " << fiddleyard->getId() << std::endl;
    }

    return os;
}

void RoadSystem::parseCinema4dXml(std::string filename)
{
	XMLCh *t1 = NULL, *t2 = NULL, *t12 = NULL;
	char *cs=NULL;
    try
    {
        xercesc::XMLPlatformUtils::Initialize();
    }
    catch (const xercesc::XMLException &toCatch)
    {
        char *message = xercesc::XMLString::transcode(toCatch.getMessage());
        std::cout << "Error during initialization! :\n" << message << std::endl;
        xercesc::XMLString::release(&message);
        return;
    }

    xercesc::XercesDOMParser *parser = new xercesc::XercesDOMParser();
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);

    try
    {
        parser->parse(filename.c_str());
    }
    catch (...)
    {
        std::cerr << "Couldn't parse Cinema4D XML-file " << filename << "!" << std::endl;
    }

    xercesc::DOMDocument *xmlDoc = parser->getDocument();
    xercesc::DOMElement *rootElement = NULL;
    if (xmlDoc)
    {
        rootElement = xmlDoc->getDocumentElement();
    }

    if (rootElement)
    {
        std::map<std::string, Road *> roadIdMap;
        std::map<std::string, Junction *> junctionIdMap;
        std::map<std::string, Fiddleyard *> fiddleyardIdMap;
        std::map<Road *, std::string> predecessorIdMap;
        std::map<Road *, std::string> successorIdMap;
        std::map<Road *, std::string> predecessorTypeMap;
        std::map<Road *, std::string> successorTypeMap;
        std::map<Road *, int> predecessorDirMap;
        std::map<Road *, int> successorDirMap;
        std::map<std::string, std::vector<Road *> > junctionRoadVectorMap;

        xercesc::DOMNodeList *splineList = rootElement->getElementsByTagName(t1 = xercesc::XMLString::transcode("obj_spline")); xercesc::XMLString::release(&t1);
        for (unsigned int splineIt = 0; splineIt < splineList->getLength(); ++splineIt)
        {
            xercesc::DOMElement *splineElement = dynamic_cast<xercesc::DOMElement *>(splineList->item(splineIt));

            int numLaneLeft = 0;
            int numLaneRight = 0;
            std::string roadId;
            std::string junctionId;
            std::string predecessorId;
            std::string successorId;
            std::string predecessorType;
            std::string successorType;
            int predecessorDir = 0;
            int successorDir = 0;

            xercesc::DOMNodeList *stringList = splineElement->getElementsByTagName(t2 = xercesc::XMLString::transcode("string")); xercesc::XMLString::release(&t2);
            std::string stringVar;
            for (unsigned int stringIt = 0; stringIt < stringList->getLength(); ++stringIt)
            {
                xercesc::DOMElement *stringElement = dynamic_cast<xercesc::DOMElement *>(stringList->item(stringIt));
                stringVar = cs = xercesc::XMLString::transcode(stringElement->getAttribute(t12 = xercesc::XMLString::transcode("v"))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                if (stringVar.find("road") == 0)
                {
                    stringVar.erase(0, 4);
                    numLaneLeft = atoi(stringVar.substr(0, 1).c_str());
                    numLaneRight = atoi(stringVar.substr(1, 1).c_str());
                    stringVar.erase(0, stringVar.find_first_of('-') + 1);
                    roadId = stringVar.substr(0, stringVar.find_first_of('-'));
                    junctionId = stringVar.substr(0, 3);
                    stringVar.erase(0, stringVar.find_first_of('-') + 1);

                    predecessorId = stringVar.substr(0, stringVar.find_first_of('-'));
                    if (predecessorId == "xxx")
                    {
                        predecessorType = "fiddleyard";
                        predecessorId = std::string("fiddleyard_pred_") + roadId;
                        Fiddleyard *fiddleyard = new Fiddleyard(predecessorId, predecessorId);
                        addFiddleyard(fiddleyard);
                        fiddleyardIdMap[predecessorId] = fiddleyard;
                        for (int laneIt = -numLaneRight; laneIt <= numLaneLeft; ++laneIt)
                        {
                            if (laneIt == 0)
                                continue;
                            std::ostringstream laneStringStream;
                            laneStringStream << laneIt;
                            fiddleyard->addVehicleSink(new VehicleSink(predecessorId + std::string("_sink_") + laneStringStream.str(), laneIt));
                        }
                        if (numLaneRight > 0)
                        {
                            fiddleyard->addVehicleSource(new VehicleSource(predecessorId + std::string("_source"), -1, 5.0, 5.0));
                        }
                    }
                    else
                    {
                        char predecessorSuffix = predecessorId[predecessorId.size() - 1];
                        predecessorId.erase(predecessorId.size() - 1, 1);
                        if (predecessorSuffix == 's')
                        {
                            predecessorType = "road";
                            predecessorDir = 1;
                        }
                        else if (predecessorSuffix == 'e')
                        {
                            predecessorType = "road";
                            predecessorDir = -1;
                        }
                        else if (predecessorSuffix == 'j')
                        {
                            predecessorType = "junction";
                        }
                    }
                    stringVar.erase(0, stringVar.find_first_of('-') + 1);

                    successorId = stringVar.substr(0, stringVar.find_first_of('-'));
                    if (successorId == "xxx")
                    {
                        successorType = "fiddleyard";
                        successorId = std::string("fiddleyard_succ_") + roadId;
                        Fiddleyard *fiddleyard = new Fiddleyard(successorId, successorId);
                        addFiddleyard(fiddleyard);
                        fiddleyardIdMap[successorId] = fiddleyard;
                        for (int laneIt = -numLaneRight; laneIt <= numLaneLeft; ++laneIt)
                        {
                            if (laneIt == 0)
                                continue;
                            std::ostringstream laneStringStream;
                            laneStringStream << laneIt;
                            fiddleyard->addVehicleSink(new VehicleSink(successorId + std::string("_sink_") + laneStringStream.str(), laneIt));
                        }
                        if (numLaneLeft > 0)
                        {
                            fiddleyard->addVehicleSource(new VehicleSource(successorId + std::string("_source"), 1, 5.0, 5.0));
                        }
                    }
                    else
                    {
                        char successorSuffix = successorId[successorId.size() - 1];
                        successorId.erase(successorId.size() - 1, 1);
                        if (successorSuffix == 's')
                        {
                            successorType = "road";
                            successorDir = 1;
                        }
                        else if (successorSuffix == 'e')
                        {
                            successorType = "road";
                            successorDir = -1;
                        }
                        else if (successorSuffix == 'j')
                        {
                            successorType = "junction";
                        }
                    }

                    break;
                }
            }
            /*
         std::cout << "Road id: " << roadId << ", numLaneLeft: " << numLaneLeft << ", numLaneRight: " << numLaneRight << ", junction: " << junctionId;
         std::cout << ", predecessorId: " << predecessorId << ", type: " << predecessorType << ", dir: " << predecessorDir;
         std::cout << ", successorId: " << successorId << ", type: " << successorType << ", dir: " << successorDir << std::endl;
         */
            std::vector<std::vector<double> > geometryMatrix;

            xercesc::DOMNodeList *vectorArrayList = splineElement->getElementsByTagName(t2 = xercesc::XMLString::transcode("vectorarray")); xercesc::XMLString::release(&t2);
            for (unsigned int vectorArrayIt = 0; vectorArrayIt < vectorArrayList->getLength(); ++vectorArrayIt)
            {
                xercesc::DOMElement *vectorArrayElement = dynamic_cast<xercesc::DOMElement *>(vectorArrayList->item(vectorArrayIt));
                //int vectorArraySize = atoi(cs = xercesc::XMLString::transcode(vectorArrayElement->getAttribute(t12 = xercesc::XMLString::transcode("size")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                xercesc::DOMNodeList *vectorList = vectorArrayElement->getElementsByTagName(t1 = xercesc::XMLString::transcode("vector")); xercesc::XMLString::release(&t1);
                for (unsigned int vectorIt = 0; vectorIt < vectorList->getLength(); ++vectorIt)
                {
                    xercesc::DOMElement *vectorElement = dynamic_cast<xercesc::DOMElement *>(vectorList->item(vectorIt));
                    std::vector<double> point;
                    point.push_back(atof(cs = xercesc::XMLString::transcode(vectorElement->getAttribute(t12 = xercesc::XMLString::transcode("z"))))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                    point.push_back(-atof(cs = xercesc::XMLString::transcode(vectorElement->getAttribute(t12 = xercesc::XMLString::transcode("x"))))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                    point.push_back(atof(cs = xercesc::XMLString::transcode(vectorElement->getAttribute(t12 = xercesc::XMLString::transcode("y"))))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                    geometryMatrix.push_back(point);
                }
            }
            double s = 0;
            for (unsigned int geometryIt = 1; geometryIt < geometryMatrix.size(); ++geometryIt)
            {
                double xdiff = geometryMatrix[geometryIt][0] - geometryMatrix[geometryIt - 1][0];
                double ydiff = geometryMatrix[geometryIt][1] - geometryMatrix[geometryIt - 1][1];
                double l = sqrt(pow(xdiff, 2) + pow(ydiff, 2));
                double hdg;
                if (xdiff > 0)
                {
                    hdg = atan(ydiff / xdiff);
                }
                else if (xdiff < 0)
                {
                    hdg = atan(ydiff / xdiff) + M_PI;
                }
                else
                {
                    hdg = (ydiff < 0) ? -0.5 * M_PI : 0.5 * M_PI;
                }

                geometryMatrix[geometryIt - 1].push_back(hdg);
                geometryMatrix[geometryIt - 1].push_back(s);
                geometryMatrix[geometryIt - 1].push_back(l);

                s += l;
            }

            Road *road = new Road(roadId, std::string("road") + roadId, s);
            addRoad(road);
            for (unsigned int geometryIt = 1; geometryIt < geometryMatrix.size(); ++geometryIt)
            {
                road->addPlanViewGeometryLine(geometryMatrix[geometryIt - 1][4],
                                              geometryMatrix[geometryIt - 1][5],
                                              geometryMatrix[geometryIt - 1][0],
                                              geometryMatrix[geometryIt - 1][1],
                                              geometryMatrix[geometryIt - 1][3]);
                road->addElevationPolynom(geometryMatrix[geometryIt - 1][4], geometryMatrix[geometryIt - 1][2], 0.0, 0.0, 0.0);
            }

            LaneSection *laneSection = new LaneSection(road,0.0);
            road->addLaneSection(laneSection);
            for (int laneIt = -numLaneRight; laneIt <= numLaneLeft; ++laneIt)
            {
                Lane *lane;
                if (laneIt == 0)
                {
                    lane = laneSection->getLane(0);
                }
                else
                {
                    lane = new Lane(laneIt, Lane::DRIVING, false);
                    laneSection->addLane(lane);
                    lane->addWidth(0.0, 3.0, 0.0, 0.0, 0.0);
                    if (predecessorDir > 0)
                    {
                        lane->setPredecessor(-laneIt);
                    }
                    else
                    {
                        lane->setPredecessor(laneIt);
                    }
                    if (successorDir < 0)
                    {
                        lane->setSuccessor(-laneIt);
                    }
                    else
                    {
                        lane->setSuccessor(laneIt);
                    }
                }

                if (laneIt == -numLaneRight)
                {
                    lane->addRoadMark(new RoadMark(0.0, 0.12, RoadMark::TYPE_SOLID, RoadMark::WEIGHT_STANDARD, RoadMark::COLOR_STANDARD, RoadMark::LANECHANGE_NONE));
                }
                else if (laneIt == numLaneLeft)
                {
                    lane->addRoadMark(new RoadMark(0.0, 0.12, RoadMark::TYPE_SOLID, RoadMark::WEIGHT_STANDARD, RoadMark::COLOR_STANDARD, RoadMark::LANECHANGE_NONE));
                }
                else
                {
                    lane->addRoadMark(new RoadMark(0.0, 0.12, RoadMark::TYPE_BROKEN, RoadMark::WEIGHT_STANDARD, RoadMark::COLOR_STANDARD, RoadMark::LANECHANGE_BOTH));
                }
            }

            roadIdMap[roadId] = road;
            predecessorIdMap[road] = predecessorId;
            predecessorTypeMap[road] = predecessorType;
            predecessorDirMap[road] = predecessorDir;
            successorIdMap[road] = successorId;
            successorTypeMap[road] = successorType;
            successorDirMap[road] = successorDir;
            junctionRoadVectorMap[junctionId].push_back(road);
        }

        for (std::map<std::string, std::vector<Road *> >::iterator junctionIt = junctionRoadVectorMap.begin(); junctionIt != junctionRoadVectorMap.end(); ++junctionIt)
        {
            if ((junctionIt->first) == "000" || (junctionIt->first).size() == 0)
                continue;
            Junction *junction = new Junction(junctionIt->first, junctionIt->first);
            addJunction(junction);
            junctionIdMap[junction->getId()] = junction;
            for (unsigned int roadIt = 0; roadIt < (junctionIt->second).size(); ++roadIt)
            {
                Road *road = (junctionIt->second)[roadIt];
                road->setJunction(junction);
                std::map<std::string, Road *>::iterator predIt = roadIdMap.find(predecessorIdMap.find(road)->second);
                if (predIt != roadIdMap.end())
                {
                    PathConnection *conn = new PathConnection(road->getId(), predIt->second, road, 1);
                    junction->addPathConnection(conn);
                    if (predecessorDirMap.find(road)->second > 0)
                    {
                        for (int laneIt = -1; laneIt >= -road->getLaneSection(0.0)->getNumLanesRight(); --laneIt)
                        {
                            conn->addLaneLink(-laneIt, laneIt);
                        }
                    }
                }
            }
        }

        for (unsigned int roadIt = 0; roadIt < this->roadVector.size(); ++roadIt)
        {
            Road *road = roadVector[roadIt];

            Tarmac *pred = NULL;
            std::string predId = predecessorIdMap.find(road)->second;
            std::string predType = predecessorTypeMap.find(road)->second;
            int predDir = predecessorDirMap.find(road)->second;
            if (predType == "road")
            {
                std::map<std::string, Road *>::iterator predIt = roadIdMap.find(predId);
                if (predIt != roadIdMap.end())
                {
                    pred = predIt->second;
                    road->setPredecessorConnection(new TarmacConnection(pred, predDir));
                }
            }
            else if (predType == "junction")
            {
                std::map<std::string, Junction *>::iterator predIt = junctionIdMap.find(predId);
                if (predIt != junctionIdMap.end())
                {
                    pred = predIt->second;
                    road->setPredecessorConnection(new TarmacConnection(pred, predDir));
                }
            }
            else if (predType == "fiddleyard")
            {
                std::map<std::string, Fiddleyard *>::iterator predIt = fiddleyardIdMap.find(predId);
                if (predIt != fiddleyardIdMap.end())
                {
                    pred = predIt->second;
                    road->setPredecessorConnection(new TarmacConnection(pred, predDir));

                    Fiddleyard *fiddleyard = predIt->second;
                    TarmacConnection *conn = new TarmacConnection(road, 1);
                    fiddleyard->setTarmacConnection(conn);
                }
            }

            Tarmac *succ = NULL;
            std::string succId = successorIdMap.find(road)->second;
            std::string succType = successorTypeMap.find(road)->second;
            int succDir = successorDirMap.find(road)->second;
            if (succType == "road")
            {
                std::map<std::string, Road *>::iterator succIt = roadIdMap.find(succId);
                if (succIt != roadIdMap.end())
                {
                    succ = succIt->second;
                    road->setSuccessorConnection(new TarmacConnection(succ, succDir));
                }
            }
            else if (succType == "junction")
            {
                std::map<std::string, Junction *>::iterator succIt = junctionIdMap.find(succId);
                if (succIt != junctionIdMap.end())
                {
                    succ = succIt->second;
                    road->setSuccessorConnection(new TarmacConnection(succ, succDir));
                }
            }
            else if (succType == "fiddleyard")
            {
                std::map<std::string, Fiddleyard *>::iterator succIt = fiddleyardIdMap.find(succId);
                if (succIt != fiddleyardIdMap.end())
                {
                    succ = succIt->second;
                    road->setSuccessorConnection(new TarmacConnection(succ, succDir));

                    Fiddleyard *fiddleyard = succIt->second;
                    TarmacConnection *conn = new TarmacConnection(road, -1);
                    fiddleyard->setTarmacConnection(conn);
                }
            }
        }
    }
}

void RoadSystem::parseLandXml(std::string filename)
{
	XMLCh *t1 = NULL, *t2 = NULL, *t3 = NULL, *t4 = NULL, *t5 = NULL, *t12 = NULL;
	char *cs = NULL;
    try
    {
        xercesc::XMLPlatformUtils::Initialize();
    }
    catch (const xercesc::XMLException &toCatch)
    {
        char *message = xercesc::XMLString::transcode(toCatch.getMessage());
        std::cout << "Error during initialization! :\n" << message << std::endl;
        xercesc::XMLString::release(&message);
        return;
    }

    xercesc::XercesDOMParser *parser = new xercesc::XercesDOMParser();
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);

    try
    {
        parser->parse(filename.c_str());
    }
    catch (...)
    {
        std::cerr << "Couldn't parse LandXML file " << filename << "!" << std::endl;
    }

    xercesc::DOMDocument *xmlDoc = parser->getDocument();
    xercesc::DOMElement *rootElement = NULL;
    if (xmlDoc)
    {
        rootElement = xmlDoc->getDocumentElement();
    }

    if (rootElement)
    {
        xercesc::DOMNodeList *RoadwayList = rootElement->getElementsByTagName(t1 = xercesc::XMLString::transcode("Roadway")); xercesc::XMLString::release(&t1);
        xercesc::DOMNodeList *AlignmentList = rootElement->getElementsByTagName(t1 = xercesc::XMLString::transcode("Alignment")); xercesc::XMLString::release(&t1);

        for (unsigned int roadwayIt = 0; roadwayIt < RoadwayList->getLength(); ++roadwayIt)
        {
            xercesc::DOMElement *roadwayElement = dynamic_cast<xercesc::DOMElement *>(RoadwayList->item(roadwayIt));
            if (roadwayElement)
            {
                std::string name(cs = xercesc::XMLString::transcode(roadwayElement->getAttribute(xercesc::XMLString::transcode("name")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                std::string alignmentRefs(cs = xercesc::XMLString::transcode(roadwayElement->getAttribute(xercesc::XMLString::transcode("alignmentRefs")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                double staStart = atof(cs = xercesc::XMLString::transcode(roadwayElement->getAttribute(xercesc::XMLString::transcode("staStart")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                double staEnd = atof(cs = xercesc::XMLString::transcode(roadwayElement->getAttribute(xercesc::XMLString::transcode("staEnd")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                Road *road = new Road(name, name, staStart + staEnd);
                addRoad(road);
                LaneSection *laneSection = new LaneSection(road,0.0);
                road->addLaneSection(laneSection);
                Lane *lane = new Lane(-1, Lane::DRIVING, false);
                laneSection->addLane(lane);
                lane->addWidth(0.0, 3.0, 0.0, 0.0, 0.0);
                for (unsigned int alignmentIt = 0; alignmentIt < AlignmentList->getLength(); ++alignmentIt)
                {
                    xercesc::DOMElement *alignmentElement = dynamic_cast<xercesc::DOMElement *>(AlignmentList->item(alignmentIt));
					char *align;
                    if (alignmentElement && std::string(align = xercesc::XMLString::transcode(alignmentElement->getAttribute(t1 = xercesc::XMLString::transcode("name")))) == alignmentRefs)
                    {
                        xercesc::DOMNodeList *CoordGeomList = alignmentElement->getElementsByTagName(t2 = xercesc::XMLString::transcode("CoordGeom")); xercesc::XMLString::release(&t2);
                        for (unsigned int coordGeomIt = 0; coordGeomIt < CoordGeomList->getLength(); ++coordGeomIt)
                        {
                            xercesc::DOMElement *coordGeomElement = dynamic_cast<xercesc::DOMElement *>(CoordGeomList->item(coordGeomIt));
                            xercesc::DOMNodeList *CoordGeomChildrenList = coordGeomElement->getChildNodes();
                            double start = 0.0;
                            double dir = 0.0;
                            for (unsigned int coordGeomChildIt = 0; coordGeomChildIt < CoordGeomChildrenList->getLength(); ++coordGeomChildIt)
                            {
                                xercesc::DOMElement *coordGeomChildElement = dynamic_cast<xercesc::DOMElement *>(CoordGeomChildrenList->item(coordGeomChildIt));
                                if (!coordGeomChildElement)
                                {
                                }

                                else if (xercesc::XMLString::compareIString(coordGeomChildElement->getTagName(), t2 = xercesc::XMLString::transcode("Line")) == 0)
                                {
                                    dir = atof(cs = xercesc::XMLString::transcode(coordGeomChildElement->getAttribute(t12 = xercesc::XMLString::transcode("dir")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    double length = atof(cs = xercesc::XMLString::transcode(coordGeomChildElement->getAttribute(t12 = xercesc::XMLString::transcode("length")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);

                                    xercesc::DOMNodeList *LineChildrenList = coordGeomChildElement->getChildNodes();
                                    for (unsigned int lineChildIt = 0; lineChildIt < LineChildrenList->getLength(); ++lineChildIt)
                                    {
                                        xercesc::DOMElement *lineChildElement = dynamic_cast<xercesc::DOMElement *>(LineChildrenList->item(lineChildIt));
                                        if (lineChildElement && xercesc::XMLString::compareIString(lineChildElement->getTagName(), t3 = xercesc::XMLString::transcode("Start")) == 0)
                                        {
                                            xercesc::DOMNodeList *StartChildrenList = lineChildElement->getChildNodes();
                                            for (unsigned int startChildIt = 0; startChildIt < StartChildrenList->getLength(); ++startChildIt)
                                            {
                                                xercesc::DOMCharacterData *startCharacterData = dynamic_cast<xercesc::DOMCharacterData *>(StartChildrenList->item(startChildIt));
                                                if (startCharacterData)
                                                {
                                                    xercesc::BaseRefVectorOf<XMLCh> *stringVector = xercesc::XMLString::tokenizeString(startCharacterData->getData());
                                                    double x = atof(cs = xercesc::XMLString::transcode(stringVector->elementAt(1))); xercesc::XMLString::release(&cs);
                                                    double y = atof(cs = xercesc::XMLString::transcode(stringVector->elementAt(0))); xercesc::XMLString::release(&cs);

                                                    road->addPlanViewGeometryLine(start, length, x, y, 2 * M_PI / 360.0 * dir);
                                                    std::cout << "Line start: x: " << x << ", y: " << y << std::endl;
                                                }
                                            }
                                        }
										xercesc::XMLString::release(&t3);
                                    }
                                    start += length;
                                }
                                else if (xercesc::XMLString::compareIString(coordGeomChildElement->getTagName(), t3 = xercesc::XMLString::transcode("Spiral")) == 0)
                                {
                                    double length = atof(cs = xercesc::XMLString::transcode(coordGeomChildElement->getAttribute(t12 = xercesc::XMLString::transcode("length")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    double radiusStart = atof(cs = xercesc::XMLString::transcode(coordGeomChildElement->getAttribute(t12 = xercesc::XMLString::transcode("radiusStart")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    double radiusEnd = atof(cs = xercesc::XMLString::transcode(coordGeomChildElement->getAttribute(t12 = xercesc::XMLString::transcode("radiusEnd")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    std::string rot(cs = xercesc::XMLString::transcode(coordGeomChildElement->getAttribute(t12 = xercesc::XMLString::transcode("rot")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    double curvSign = (rot == "cw") ? -1 : 1;

                                    xercesc::DOMNodeList *SpiralChildrenList = coordGeomChildElement->getChildNodes();
                                    for (unsigned int spiralChildIt = 0; spiralChildIt < SpiralChildrenList->getLength(); ++spiralChildIt)
                                    {
                                        xercesc::DOMElement *spiralChildElement = dynamic_cast<xercesc::DOMElement *>(SpiralChildrenList->item(spiralChildIt));
                                        if (spiralChildElement && xercesc::XMLString::compareIString(spiralChildElement->getTagName(), t4 = xercesc::XMLString::transcode("Start")) == 0)
                                        {
                                            xercesc::DOMNodeList *StartChildrenList = spiralChildElement->getChildNodes();
                                            for (unsigned int startChildIt = 0; startChildIt < StartChildrenList->getLength(); ++startChildIt)
                                            {
                                                xercesc::DOMCharacterData *startCharacterData = dynamic_cast<xercesc::DOMCharacterData *>(StartChildrenList->item(startChildIt));
                                                if (startCharacterData)
                                                {
                                                    xercesc::BaseRefVectorOf<XMLCh> *stringVector = xercesc::XMLString::tokenizeString(startCharacterData->getData());
                                                    double x = atof(cs = xercesc::XMLString::transcode(stringVector->elementAt(1))); xercesc::XMLString::release(&cs);
                                                    double y = atof(cs = xercesc::XMLString::transcode(stringVector->elementAt(0))); xercesc::XMLString::release(&cs);

                                                    road->addPlanViewGeometrySpiral(start, length, x, y, 2 * M_PI / 360.0 * dir, curvSign / radiusStart, curvSign / radiusEnd);
                                                    std::cout << "Spiral start: x: " << x << ", y: " << y << std::endl;
                                                }
                                            }
                                        }
										xercesc::XMLString::release(&t4);
                                    }
                                    start += length;
                                }
                                else if (xercesc::XMLString::compareIString(coordGeomChildElement->getTagName(), t4 = xercesc::XMLString::transcode("Curve")) == 0)
                                {
                                    double length = atof(cs = xercesc::XMLString::transcode(coordGeomChildElement->getAttribute(t12 = xercesc::XMLString::transcode("length")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    double dirStart = atof(cs = xercesc::XMLString::transcode(coordGeomChildElement->getAttribute(t12 = xercesc::XMLString::transcode("dirStart")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    double dirEnd = atof(cs = xercesc::XMLString::transcode(coordGeomChildElement->getAttribute(t12 = xercesc::XMLString::transcode("dirEnd")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    double radius = atof(cs = xercesc::XMLString::transcode(coordGeomChildElement->getAttribute(t12 = xercesc::XMLString::transcode("radius")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    std::string rot(cs = xercesc::XMLString::transcode(coordGeomChildElement->getAttribute(t12 = xercesc::XMLString::transcode("rot")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                    double curvSign = (rot == "cw") ? -1 : 1;

                                    xercesc::DOMNodeList *CurveChildrenList = coordGeomChildElement->getChildNodes();
                                    for (unsigned int curveChildIt = 0; curveChildIt < CurveChildrenList->getLength(); ++curveChildIt)
                                    {
                                        xercesc::DOMElement *curveChildElement = dynamic_cast<xercesc::DOMElement *>(CurveChildrenList->item(curveChildIt));
                                        if (curveChildElement && xercesc::XMLString::compareIString(curveChildElement->getTagName(), t5 = xercesc::XMLString::transcode("Start")) == 0)
                                        {
                                            xercesc::DOMNodeList *StartChildrenList = curveChildElement->getChildNodes();
                                            for (unsigned int startChildIt = 0; startChildIt < StartChildrenList->getLength(); ++startChildIt)
                                            {
                                                xercesc::DOMCharacterData *startCharacterData = dynamic_cast<xercesc::DOMCharacterData *>(StartChildrenList->item(startChildIt));
                                                if (startCharacterData)
                                                {
                                                    xercesc::BaseRefVectorOf<XMLCh> *stringVector = xercesc::XMLString::tokenizeString(startCharacterData->getData());
                                                    double x = atof(cs = xercesc::XMLString::transcode(stringVector->elementAt(1))); xercesc::XMLString::release(&cs);
                                                    double y = atof(cs = xercesc::XMLString::transcode(stringVector->elementAt(0))); xercesc::XMLString::release(&cs);

                                                    road->addPlanViewGeometryArc(start, length, x, y, 2 * M_PI / 360.0 * dirStart, curvSign / radius);
                                                    std::cout << "Curve start: x: " << x << ", y: " << y << std::endl;
                                                }
                                            }
                                        }
                                    }
                                    start += length;
                                    dir = dirEnd;
                                }
								xercesc::XMLString::release(&t2);
								xercesc::XMLString::release(&t3);
								xercesc::XMLString::release(&t4);
                            }
                        }
                        xercesc::DOMNodeList *ProfileList = alignmentElement->getElementsByTagName(t2 = xercesc::XMLString::transcode("Profile")); xercesc::XMLString::release(&t2);
                        for (unsigned int profileIt = 0; profileIt < ProfileList->getLength(); ++profileIt)
                        {
                            xercesc::DOMElement *profileElement = dynamic_cast<xercesc::DOMElement *>(ProfileList->item(profileIt));
                            if (profileElement)
                            {
                                xercesc::DOMNodeList *ProfAlignList = profileElement->getElementsByTagName(t3 = xercesc::XMLString::transcode("ProfAlign")); xercesc::XMLString::release(&t3);
                                for (unsigned int profAlignIt = 0; profAlignIt < ProfAlignList->getLength(); ++profAlignIt)
                                {
                                    xercesc::DOMElement *profAlignElement = dynamic_cast<xercesc::DOMElement *>(ProfAlignList->item(profAlignIt));
                                    if (profAlignElement)
                                    {
                                        xercesc::DOMNodeList *ProfAlignChildrenList = profAlignElement->getChildNodes();
                                        std::vector<double> hVec;
                                        std::vector<double> sVec;
                                        std::vector<double> lVec;

                                        for (unsigned int profAlignChildIt = 0; profAlignChildIt < ProfAlignChildrenList->getLength(); ++profAlignChildIt)
                                        {
                                            xercesc::DOMElement *profAlignChildElement = dynamic_cast<xercesc::DOMElement *>(ProfAlignChildrenList->item(profAlignChildIt));
                                            if (!profAlignChildElement)
                                            {
                                            }
                                            else if (xercesc::XMLString::compareIString(profAlignChildElement->getTagName(), t2 = xercesc::XMLString::transcode("PVI")) == 0)
                                            {
                                                lVec.push_back(0.0);

                                                xercesc::DOMNodeList *PviChildrenList = profAlignChildElement->getChildNodes();
                                                for (unsigned int pviChildIt = 0; pviChildIt < PviChildrenList->getLength(); ++pviChildIt)
                                                {
                                                    xercesc::DOMCharacterData *pviCharacterData = dynamic_cast<xercesc::DOMCharacterData *>(PviChildrenList->item(pviChildIt));
                                                    if (pviCharacterData)
                                                    {
                                                        xercesc::BaseRefVectorOf<XMLCh> *stringVector = xercesc::XMLString::tokenizeString(pviCharacterData->getData());
                                                        double h = atof(cs = xercesc::XMLString::transcode(stringVector->elementAt(1))); xercesc::XMLString::release(&cs);
                                                        double s = atof(cs = xercesc::XMLString::transcode(stringVector->elementAt(0))); xercesc::XMLString::release(&cs);
                                                        hVec.push_back(h);
                                                        sVec.push_back(s);

                                                        //road->addElevationPolynom(s,h,0.0,0.0,0.0);
                                                        break;
                                                    }
                                                }
                                            }
                                            else if (xercesc::XMLString::compareIString(profAlignChildElement->getTagName(), t3 = xercesc::XMLString::transcode("ParaCurve")) == 0)
                                            {
                                                double length = atof(cs = xercesc::XMLString::transcode(profAlignChildElement->getAttribute(t12 = xercesc::XMLString::transcode("length")))); xercesc::XMLString::release(&t12); xercesc::XMLString::release(&cs);
                                                lVec.push_back(length);

                                                xercesc::DOMNodeList *ParaCurveChildrenList = profAlignChildElement->getChildNodes();
                                                for (unsigned int paraCurveChildIt = 0; paraCurveChildIt < ParaCurveChildrenList->getLength(); ++paraCurveChildIt)
                                                {
                                                    xercesc::DOMCharacterData *paraCurveCharacterData = dynamic_cast<xercesc::DOMCharacterData *>(ParaCurveChildrenList->item(paraCurveChildIt));
                                                    if (paraCurveCharacterData)
                                                    {
                                                        xercesc::BaseRefVectorOf<XMLCh> *stringVector = xercesc::XMLString::tokenizeString(paraCurveCharacterData->getData());
                                                        double h = atof(cs = xercesc::XMLString::transcode(stringVector->elementAt(1))); xercesc::XMLString::release(&cs);
                                                        double s = atof(cs = xercesc::XMLString::transcode(stringVector->elementAt(0))); xercesc::XMLString::release(&cs);
                                                        hVec.push_back(h);
                                                        sVec.push_back(s);

                                                        //road->addElevationPolynom(s,h,0.0,0.0,0.0);
                                                        break;
                                                    }
                                                }
                                            }
											xercesc::XMLString::release(&t2);
											xercesc::XMLString::release(&t3);
                                        }
                                        double s = 0.0;
                                        double h = 0.0;
                                        if (!lVec.empty())
                                        {
                                            s = sVec[0];
                                            h = hVec[0];
                                        }
                                        for (unsigned int p = 1; p < lVec.size() - 1; ++p)
                                        {
                                            if (lVec[p] == 0.0)
                                            {
                                                road->addElevationPolynom(s, h, (hVec[p] - h) / (sVec[p] - s), 0.0, 0.0);
                                                s = sVec[p];
                                                h = hVec[p];
                                            }
                                            else
                                            {
                                                double l = lVec[p];

                                                double s1 = sVec[p] - l * 0.5;
                                                double s2 = sVec[p] + l * 0.5;
                                                double m1 = (hVec[p] - hVec[p - 1]) / (sVec[p] - sVec[p - 1]);
                                                double m2 = (hVec[p + 1] - hVec[p]) / (sVec[p + 1] - sVec[p]);
                                                double h1 = hVec[p] - m1 * l * 0.5;
                                                double h2 = hVec[p] + m2 * l * 0.5;

                                                double c = -(l * m1 - h2 + h1) / (l * l);

                                                road->addElevationPolynom(s, h, m1, 0.0, 0.0);
                                                road->addElevationPolynom(s1, h1, m1, c, 0.0);

                                                s = s2;
                                                h = h2;
                                            }
                                        }
                                        if (!lVec.empty())
                                        {
                                            int last = lVec.size() - 1;
                                            road->addElevationPolynom(s, h, (hVec[last] - h) / (sVec[last] - s), 0.0, 0.0);
                                        }
                                    }
                                }
                            }
                        }
                    }
					xercesc::XMLString::release(&align); xercesc::XMLString::release(&t1);
                }
            }
        }
    }
}

int RoadSystem::getLineLength(std::vector<double> &XVector, std::vector<double> &YVector, int startIndex, int endIndex, double delta)
{
    for (int i = startIndex + 2; i <= endIndex; i++)
    {
        //double dist;
        osg::Vec2d a(XVector[startIndex], YVector[startIndex]);
        osg::Vec2d b(XVector[i], YVector[i]);
        b = b - a;
        b.normalize();
        for (int n = startIndex + 1; n < i; n++)
        {
            osg::Vec2d p(XVector[n], YVector[n]);
            osg::Vec2d pa = p - a;
            double len = pa.length();
            double dist = ((pa / len) * b) * len;
            if (fabs(dist) > delta)
            {
                return (n - startIndex);
            }
        }
    }
    return endIndex - startIndex;
}

void RoadSystem::parseIntermapRoad(const std::string &filename, const std::string &projFromString, const std::string &projToString)
{
    std::ifstream file(filename.c_str());

    if (file.fail())
    {
        return;
    }

    projPJ pj_from, pj_to;

    if (!(pj_from = pj_init_plus(projFromString.c_str())))
    {
        return;
        std::cerr << "RoadSystem::parseIntermapRoad(): couldn't initalize projection source: " << projFromString << std::endl;
    }
    if (!(pj_to = pj_init_plus(projToString.c_str())))
    {
        std::cerr << "RoadSystem::parseIntermapRoad(): couldn't initalize projection target: " << projToString << std::endl;
        return;
    }

    while (!file.eof())
    {
        std::vector<double> XVector;
        std::vector<double> YVector;
        std::vector<double> ZVector;
        std::vector<int> FeatVector;
        std::vector<int> VFOMVector;

        std::string lineString;
        std::getline(file, lineString);
        while (!file.eof() && lineString.size() >= 3)
        {
            double longitude, latitude, altitude;
            int feature, vfom;

            size_t frontPos = 0;
            size_t rearPos = lineString.find_first_of(" ", frontPos);
            std::stringstream(lineString.substr(frontPos, rearPos)) >> longitude;
            frontPos = rearPos + 1;
            rearPos = lineString.find_first_of(" ", frontPos);
            std::stringstream(lineString.substr(frontPos, rearPos)) >> latitude;
            frontPos = rearPos + 1;
            rearPos = lineString.find_first_of(" ", frontPos);
            std::stringstream(lineString.substr(frontPos, rearPos)) >> altitude;
            frontPos = rearPos + 1;
            rearPos = lineString.find_first_of(" ", frontPos);
            std::stringstream(lineString.substr(frontPos, rearPos)) >> feature;
            frontPos = rearPos + 1;
            rearPos = lineString.find_first_of(" ", frontPos);
            std::stringstream(lineString.substr(frontPos, rearPos)) >> vfom;

            /*double longMin = 9.0916;
         double longMax = 9.116;
         double latMin = 48.7378;
         double latMax = 48.7537;
         if(longMin<=longitude && longitude<=longMax && latMin<=latitude && latitude<=latMax) {*/
            double x = longitude * DEG_TO_RAD;
            double y = latitude * DEG_TO_RAD;
            double z = altitude;
            pj_transform(pj_from, pj_to, 1, 1, &x, &y, &z);

            XVector.push_back(x /* - 1000000.0*/);
            YVector.push_back(y /* - 6190000.0*/);
            ZVector.push_back(z /*+10.0*/);
            FeatVector.push_back(feature);
            VFOMVector.push_back(vfom);
            // }

            std::getline(file, lineString);
        }

        if (XVector.size() > 1)
        {
            std::ostringstream nameStream;
            nameStream << "ROAD_" << XVector[0] << "_" << YVector[0];
            std::string name = nameStream.str();
            Road *road = new Road(name, name);
            addRoad(road);

            LaneSection *laneSection = new LaneSection(road, 0.0);
            road->addLaneSection(laneSection);

            int roadType = (int)((double)FeatVector[0] / 100.0);
            double laneWidth = 3.0;
            int numLanesLeft = 1;
            int numLanesRight = 1;
            switch (roadType)
            {
            case 1: //Highways
                laneWidth = 3.75;
                numLanesLeft = 2;
                numLanesRight = 2;
                break;
            case 2: //Major Roads
                laneWidth = 3.5;
                break;
            case 3: //Minor Roads
                laneWidth = 3.25;
                break;
            case 4: //Local Roads
                laneWidth = 3.0;
                break;
            default:
                laneWidth = 3.0;
                numLanesLeft = 1;
                numLanesRight = 1;
            }

            for (int laneNum = -numLanesRight; laneNum <= numLanesLeft; ++laneNum)
            {
                Lane *lane = new Lane(laneNum, Lane::DRIVING, false);
                laneSection->addLane(lane);
                if (laneNum == -numLanesRight || laneNum == numLanesLeft)
                {
                    lane->addRoadMark(new RoadMark(0.0, 0.12, RoadMark::TYPE_SOLID, RoadMark::WEIGHT_STANDARD, RoadMark::COLOR_STANDARD, RoadMark::LANECHANGE_BOTH));
                }
                else
                {
                    lane->addRoadMark(new RoadMark(0.0, 0.12, RoadMark::TYPE_BROKEN, RoadMark::WEIGHT_STANDARD, RoadMark::COLOR_STANDARD, RoadMark::LANECHANGE_BOTH));
                }
                if (laneNum == 0)
                {
                    continue;
                }

                lane->addWidth(0.0, laneWidth, 0.0, 0.0, 0.0);
            }

            double roadLength = 0.0;

            if (XVector.size() == 2)
            {
                double start = roadLength;
                double length = sqrt(pow(YVector[1] - YVector[0], 2) + pow(XVector[1] - XVector[0], 2));
                roadLength += length;

                road->addPlanViewGeometryLine(start, length, XVector[0], YVector[0], atan2(YVector[1] - YVector[0], XVector[1] - XVector[0]));
                road->addElevationPolynom(start, ZVector[0], (ZVector[1] - ZVector[0]) / length, 0.0, 0.0);
            }
            else
            {
                for (unsigned int i = 0; i < XVector.size() - 1; ++i)
                {
                    double start = roadLength;

                    double startHdg;
                    double endHdg;
                    double startGradient;
                    double endGradient;
                    if (i == 0)
                    {
                        endHdg = atan2(YVector[i + 2] - YVector[i], XVector[i + 2] - XVector[i]);
                        startHdg = 2.0 * atan2(YVector[i + 1] - YVector[i], XVector[i + 1] - XVector[i]) - endHdg;

                        endGradient = (ZVector[i + 2] - ZVector[i]) / sqrt(pow(XVector[i + 2] - XVector[i], 2) + pow(YVector[i + 2] - YVector[i], 2));
                        startGradient = 2.0 * (ZVector[i + 1] - ZVector[i]) / sqrt(pow(XVector[i + 1] - XVector[i], 2) + pow(YVector[i + 1] - YVector[i], 2)) - endGradient;
                    }
                    else if (i == (XVector.size() - 2))
                    {
                        startHdg = atan2(YVector[i + 1] - YVector[i - 1], XVector[i + 1] - XVector[i - 1]);
                        endHdg = 2.0 * atan2(YVector[i + 1] - YVector[i], XVector[i + 1] - XVector[i]) - startHdg;

                        startGradient = (ZVector[i + 1] - ZVector[i - 1]) / sqrt(pow(XVector[i + 1] - XVector[i - 1], 2) + pow(YVector[i + 1] - YVector[i - 1], 2));
                        endGradient = 2.0 * (ZVector[i + 1] - ZVector[i]) / sqrt(pow(XVector[i + 1] - XVector[i], 2) + pow(YVector[i + 1] - YVector[i], 2)) - startGradient;
                    }
                    else
                    {
                        startHdg = atan2(YVector[i + 1] - YVector[i - 1], XVector[i + 1] - XVector[i - 1]);
                        endHdg = atan2(YVector[i + 2] - YVector[i], XVector[i + 2] - XVector[i]);

                        startGradient = (ZVector[i + 1] - ZVector[i - 1]) / sqrt(pow(XVector[i + 1] - XVector[i - 1], 2) + pow(YVector[i + 1] - YVector[i - 1], 2));
                        endGradient = (ZVector[i + 2] - ZVector[i]) / sqrt(pow(XVector[i + 2] - XVector[i], 2) + pow(YVector[i + 2] - YVector[i], 2));
                    }

                    double endX = XVector[i + 1] - XVector[i];
                    double endY = YVector[i + 1] - YVector[i];
                    double endU = endX * cos(startHdg) + endY * sin(startHdg);
                    double endV = -endX * sin(startHdg) + endY * cos(startHdg);
                    double endSlope = tan(endHdg - startHdg);

                    //double a = 0.0;
                    double b = 0.0;
                    double c = (3.0 * endV - endSlope * endU) / pow(endU, 2);
                    double d = -(2.0 * endV - endSlope * endU) / pow(endU, 3);

                    //Cut taylor series approximation (1-degree) of arc length integral
                    double length = endU * sqrt(pow(((3.0 * d * pow(endU, 2)) / 4.0 + c * endU + b), 2) + 1.0);
                    /*if(length<0) {
                 std::cout << "Road: " << road->getId() << ", length: " << length << ", endU: " << endU << ", endX: " << endX << ", endY: " << endY << ", startHdg: " << startHdg << ", endHdg: " << endHdg << ", endSlope: " << endSlope << ", size: " << XVector.size() << std::endl;
                 }*/
                    /* lines */
                    unsigned int startIndex = 0;
                    int endIndex = 0;
                    length = 0.0;
                    while (startIndex < XVector.size() - 1)
                    {
                        endIndex = XVector.size() - 1;
                        int numLineSegments = getLineLength(XVector, YVector, startIndex, endIndex, 2.0);
                        double segmentlength = sqrt(pow(XVector[startIndex] - XVector[startIndex + numLineSegments], 2) + pow(YVector[startIndex] - YVector[startIndex + numLineSegments], 2));
                        road->addPlanViewGeometryLine(start, length, XVector[startIndex], YVector[startIndex], atan2(YVector[endIndex] - YVector[startIndex], XVector[endIndex] - XVector[startIndex]));
                        start += segmentlength;
                        startIndex += numLineSegments;
                        length += segmentlength;
                    }

                    /* polynom */
                    //               road->addPlanViewGeometryPolynom(start, length, XVector[i], YVector[i], startHdg, a, b, c, d);
                    //road->addElevationPolynom(start, ZVector[i], (ZVector[i+1]-ZVector[i])/length, 0.0, 0.0);

                    roadLength += length;

                    road->addElevationPolynom(start, ZVector[i], startGradient,
                                              -(-3.0 * ZVector[i + 1] + 3.0 * ZVector[i] + (endGradient + 2.0 * startGradient) * length) / pow(length, 2),
                                              (-2.0 * ZVector[i + 1] + 2.0 * ZVector[i] + (endGradient + startGradient) * length) / pow(length, 3));
                }
            }

            road->setLength(roadLength);
        }
    }

    pj_free(pj_from);
    pj_free(pj_to);
    //std::cout << "RoadSystem information: " << system << std::endl;
}

Vector2D RoadSystem::searchPosition(const Vector3D &worldPos, Road *&road, double &u)
{
    Vector2D pos(std::numeric_limits<float>::signaling_NaN(), std::numeric_limits<float>::signaling_NaN());
    if (road)
    {
        //std::cout << "road exists" << std::endl;
		pos = road->searchPosition(worldPos, u);
    }
    if (!pos.isNaV())
    {
        u = pos.u();
    }
    else
    {
        //std::cout << "road NOT exist" << std::endl;
		
		road = NULL;
        u = -1.0;

        std::set<Road *, bool (*)(Road *, Road *)> roadSet(Road::compare);
        for (unsigned int roadIt = 0; roadIt < roadVector.size(); ++roadIt)
        {
            Road *searchRoad = roadVector[roadIt];
	    if(searchRoad)
	    {
		osg::Geode *geode = searchRoad->getRoadGeode();
		if(geode)
		{
        	    osg::BoundingBox roadBox = geode->getBoundingBox();
        	    roadBox.zMin() = -1000000; // ignore height
        	    roadBox.zMax() = 1000000;
        	    if (roadBox.contains(osg::Vec3(worldPos.x(), worldPos.y(), worldPos.z())))
        	    {
                	roadSet.insert(searchRoad);
        	    }
		}
	    }
            //std::cout << "worldPos: " << worldPos.x() << ", " << worldPos.y() << ", " << worldPos.z() << std::endl;
            //std::cout << "BB: " << roadBox.xMin() << ", " << roadBox.yMin() << ", " << roadBox.zMin() << " - " << roadBox.xMax() << ", " << roadBox.yMax() << ", " << roadBox.zMax() << std::endl;
        }

        if (!roadSet.empty())
        {
            //std::cout << "road set not empty" << std::endl;
			
			for (std::set<Road *, bool (*)(Road *, Road *)>::iterator roadSetIt = roadSet.begin(); roadSetIt != roadSet.end(); ++roadSetIt)
            {
                pos = (*roadSetIt)->searchPosition(worldPos, -1);

                if (!pos.isNaV())
                {
                    //std::cout << "pos not a nan" << std::endl;
					road = (*roadSetIt);
                    u = pos.u();
                    break;
                }
            }
        }

        //std::cerr << "Searching on road: " << road << ", pos: " << worldPos.x() << ", " << worldPos.y() << ", " << worldPos.z() << std::endl;
    }

    return pos;
}

Vector2D RoadSystem::searchPositionFollowingRoad(const Vector3D &worldPos, Road *&road, double &u)
{
    Vector2D pos(std::numeric_limits<float>::signaling_NaN(), std::numeric_limits<float>::signaling_NaN());

    //if(road==NULL || u<0.0) return pos;
    if (road == NULL)
        return pos;

    if (u >= 0.0 && u <= road->getLength())
    {
        pos = road->searchPositionNoBorder(worldPos, u);
    }
    if (!pos.isNaV())
    {
        if (road->isJunctionPath())
        {
            const std::map<Road *, PathConnectionSet> &connSetMap = road->getJunction()->getPathConnectionSetMap();
            for (std::map<Road *, PathConnectionSet>::const_iterator connSetIt = connSetMap.begin(); connSetIt != connSetMap.end(); ++connSetIt)
            {
                for (PathConnectionSet::const_iterator connIt = connSetIt->second.begin(); connIt != connSetIt->second.end(); ++connIt)
                {
                    Vector2D newPos = (*connIt)->getConnectingPath()->searchPosition(worldPos, (*connIt)->getConnectingPath()->getLength() * 0.5);
                    if (!newPos.isNaV() && (*connIt)->getConnectingPath()->isOnRoad(newPos) && (fabs(newPos.v()) < fabs(pos.v())))
                    {
                        //std::cout << "On junction, changing position from path " << road->getId() << " to path " << (*connIt)->getConnectingPath()->getId() << ": (" << pos.u() << ", " << pos.v() << ") -> (" << newPos.u() << ", " << newPos.v() << ")" << std::endl;
                        road = (*connIt)->getConnectingPath();
                        pos = newPos;
                    }
                }
            }
        }

        u = pos.u();
        return pos;
    }
    else
    {
        //Following Road
        TarmacConnection *nextTarmacConnection = (u < 0.5 * road->getLength()) ? road->getPredecessorConnection() : road->getSuccessorConnection();
        if (!nextTarmacConnection)
            return pos;

        if (dynamic_cast<FiddleRoad *>(nextTarmacConnection->getConnectingTarmac()) != NULL)
        {
            std::cout << "----- Fiddle road!!! ------" << std::endl;
            return pos;
        }

        Road *nextRoad = dynamic_cast<Road *>(nextTarmacConnection->getConnectingTarmac());
        if (nextRoad)
        {
            double u_init = (nextTarmacConnection->getConnectingTarmacDirection() == -1) ? nextRoad->getLength() : 0.0;
            pos = nextRoad->searchPositionNoBorder(worldPos, u_init);
            if (!pos.isNaV())
            {
                road = nextRoad;
                u = pos.u();
                return pos;
            }
        }

        Road *getConnectingPath();
        int getConnectingPathDirection();
        Junction *nextJunction = dynamic_cast<Junction *>(nextTarmacConnection->getConnectingTarmac());
        if (nextJunction)
        {
            PathConnectionSet connSet = nextJunction->getPathConnectionSet(road);
            for (PathConnectionSet::iterator connSetIt = connSet.begin(); connSetIt != connSet.end(); ++connSetIt)
            {
                double u_init = ((*connSetIt)->getConnectingPathDirection() == -1) ? (*connSetIt)->getConnectingPath()->getLength() : 0.0;
                Vector2D newPos = (*connSetIt)->getConnectingPath()->searchPositionNoBorder(worldPos, u_init);
                if (!newPos.isNaV() && (*connSetIt)->getConnectingPath()->isOnRoad(newPos) && (pos.isNaV() || (fabs(newPos.v()) < fabs(pos.v()))))
                {
                    road = (*connSetIt)->getConnectingPath();
                    pos = newPos;
                    u = pos.u();
                }
            }

            return pos;
        }
        else
        {
            return pos;
        }
    }
}

std::vector<Road*> RoadSystem::searchPositionList(const Vector3D &worldPos/*, int initialRoad*/)
{
	Vector2D pos(std::numeric_limits<float>::signaling_NaN(), std::numeric_limits<float>::signaling_NaN());
	std::vector<Road*> outVector;
	std::vector<Road> initVector;
	
	double u = -1.0;
	
	std::set<Road *, bool (*)(Road *, Road *)> roadSet(Road::compare);
	for (unsigned int roadIt = 0; roadIt < roadVector.size(); ++roadIt)
	{
		Road *searchRoad = roadVector[roadIt];
		if(searchRoad)
		{
			osg::Geode *geode = searchRoad->getRoadGeode();
			if(geode)
			{
				osg::BoundingBox roadBox = geode->getBoundingBox();
				roadBox.zMin() = -1000000; // ignore height
				roadBox.zMax() = 1000000;
				if (roadBox.contains(osg::Vec3(worldPos.x(), worldPos.y(), worldPos.z())))
				{
					roadSet.insert(searchRoad);
				}
			}
		}
		//std::cout << "worldPos: " << worldPos.x() << ", " << worldPos.y() << ", " << worldPos.z() << std::endl;
		//std::cout << "BB: " << roadBox.xMin() << ", " << roadBox.yMin() << ", " << roadBox.zMin() << " - " << roadBox.xMax() << ", " << roadBox.yMax() << ", " << roadBox.zMax() << std::endl;
	}

	if (!roadSet.empty())
	{
		//std::cout << "road set not empty" << std::endl;
		for (std::set<Road *, bool (*)(Road *, Road *)>::iterator roadSetIt = roadSet.begin(); roadSetIt != roadSet.end(); ++roadSetIt)
		{
			pos = (*roadSetIt)->searchPosition(worldPos, -1);
			//std::cout << /*"roadsetiterator " << roadSetIt << */" pos: " << pos.x() << " " << pos.y() << std::endl;
			if (!pos.isNaV())
			{
				Road* road = (*roadSetIt);
				outVector.push_back(road);
			}
		}
	}
	
	return outVector;
}

void RoadSystem::analyzeForCrossingJunctionPaths()
{
    if (roadVector.size() == 0)
        return;

    std::vector<Road *>::iterator outerRoadEndIt = roadVector.end() - 1;

    for (std::vector<Road *>::iterator outerRoadIt = roadVector.begin(); outerRoadIt != outerRoadEndIt; ++outerRoadIt)
    {
        for (std::vector<Road *>::iterator innerRoadIt = outerRoadIt + 1; innerRoadIt != roadVector.end(); ++innerRoadIt)
        {
            if ((*outerRoadIt)->getJunction() != (*innerRoadIt)->getJunction())
                continue;
            std::cout << "RoadSystem::analyzeForCrossingJunctionPaths(): Road "
                      << (*outerRoadIt)->getId() << " in same junction as road " << (*innerRoadIt)->getId() << std::endl;

            double u1 = (*outerRoadIt)->getLength() * 0.5;
            double u2 = (*innerRoadIt)->getLength() * 0.5;

            for (unsigned int i = 0; i < 10; ++i)
            {
                Vector3D r1 = (*outerRoadIt)->getChordLinePlanViewPoint(u1);
                Vector3D r2 = (*innerRoadIt)->getChordLinePlanViewPoint(u2);
                double dd = (r1.x() - r2.x()) * (r1.x() - r2.x()) + (r1.y() - r2.y()) * (r1.y() - r2.y());
                /*double tan1 = tan(r1[2]);
            double tan2 = tan(r2[2]);
            double dxdu1 = sqrt(tan1*tan1/(tan1*tan1+1.0));
            double dydu1 = sqrt(1.0/(tan1*tan1+1.0));
            double dxdu2 = sqrt(tan2*tan2/(tan2*tan2+1.0));
            double dydu2 = sqrt(1.0/(tan2*tan2+1.0));*/
                double dxdu1 = cos(r1[2]);
                double dydu1 = sin(r1[2]);
                double dxdu2 = cos(r2[2]);
                double dydu2 = sin(r2[2]);
                double ddddu1 = 2.0 * (dxdu1 * (r1.x() - r2.x()) + dydu1 * (r1.y() - r2.y()));
                double ddddu2 = 2.0 * (dxdu2 * (r1.x() - r2.x()) + dydu2 * (r1.y() - r2.y()));
                double inv_mddddu = 1.0 / (ddddu1 * ddddu1 + ddddu2 * ddddu2);
                u1 -= ddddu1 * inv_mddddu * dd;
                u2 -= ddddu2 * inv_mddddu * dd;
                std::cout << "\t i: " << i << ", u1: " << u1 << ", u2: " << u2 << ", d: " << sqrt(dd) << std::endl;
                std::cout << "\t\t ddddu1: " << ddddu1 << ", ddddu2: " << ddddu2 << ", inv_mddddu: " << inv_mddddu << std::endl;
            }
        }
    }
}

void RoadSystem::update(const double &dt)
{
    /*for(unsigned int i=0; i<junctionVector.size(); ++i) {
      junctionVector[i]->update(dt);
   }*/

    /*for(unsigned int i=0; i<signalVector.size(); ++i) {
      signalVector[i]->update(dt);
   }*/

    for (unsigned int i = 0; i < controllerVector.size(); ++i)
    {
        controllerVector[i]->update(dt);
    }
}

Road *RoadLineSegment::getRoad()
{
    return road;
}

RoadLineSegment::RoadLineSegment()
{
    this->setRoad(NULL);
    this->set_smax(-1.0);
    this->set_smin(-1.0);
    //std::cout << "smax " << this->get_smax() << " smin " << this->get_smin() << std::endl;
}

void RoadLineSegment::setRoad(Road *r)
{
    road = r;
}

double RoadLineSegment::get_smax()
{
    return smax;
}

double RoadLineSegment::get_smin()
{
    return smin;
}

void RoadLineSegment::set_smax(double s_max)
{
    smax = s_max;
}

void RoadLineSegment::set_smin(double s_min)
{
    smin = s_min;
}

void RoadLineSegment::check_s(double s)
{
    //std::cout << ">> old smin: "  << smin << " smax: " << smax;
    if (this->smin == -1 && this->smax == -1)
    {
        smin = s;
        smax = s;
    }
    else if (this->smin > s)
        smin = s;
    else if (this->smax < s)
        smax = s;
    //std::cout << " ||| new smin: " << smin << " smax: " << smax << " ---- s: "<< s << std::endl;
}

void RoadSystem::scanStreets()
{
    int numRoads;
    numRoads = (Instance())->getNumRoads();
    //std::cout << std::endl << "scanStreets() | getNumRoads = " << numRoads << std::endl;
    //      if(coVRMSController::instance()->isMaster()) {
    std::cout << ">> Analysing RoadSystem..." << std::endl;
    //      }

    //GröÃ des StraÃnnetzes ermitteln
    for (int count = 0; count < numRoads; count++)
    {
        Road *current_road = (Instance())->getRoad(count);
        double length = 0.0;
        while (length < current_road->getLength())
        {
            double x = current_road->getCenterLinePoint(length)[0];
            double y = current_road->getCenterLinePoint(length)[1];

            if (count == 0 && length == 0.0)
            {
                x_min = x;
                x_max = x;
                y_min = y;
                y_max = y;
            }
            else
            {
                if (x < x_min)
                {
                    x_min = x;
                    //std::cout << "..x_min........Road: " << count << " s-Pos: " << length << " x-Pos: " << x << " y-Pos: " << y << " ||| x_min: " << x_min << " x_max: " << x_max << " | y_min: " << y_min << " y_max: " << y_max << std::endl;
                }
                else if (x > x_max)
                {
                    x_max = x;
                }

                if (y < y_min)
                {
                    y_min = y;
                }
                else if (y > y_max)
                {
                    y_max = y;
                }
            }
            length += 0.5;
        }
    }

    //      if(coVRMSController::instance()->isMaster()) {
    std::cout << "... x_min: " << x_min << " x_max: " << x_max << " | y_min: " << y_min << " y_max: " << y_max << std::endl;
    //      }
    double delta_x = x_max - x_min;
    double delta_y = y_max - y_min;
    //std::cout << "... delta x: " << delta_x << " | delta y: " << delta_y << std::endl;
    //
    //        // Anzahl der Kacheln
    _tiles_x = (int)(delta_x / _tile_width); // Anzahl der Kacheln
    _tiles_y = (int)(delta_y / _tile_width);

    // Kacheln mit leeren Elementen füllen
    for (int init_y = 0; init_y <= _tiles_y; init_y++)
    {
        for (int init_x = 0; init_x <= _tiles_x; init_x++)
        {
            std::vector<std::list<RoadLineSegment *> > empty_vec;
            rls_vector.push_back(empty_vec);
            std::list<RoadLineSegment *> empty_list;
            (rls_vector.at(init_x)).push_back(empty_list);
        }
    }

    // Kacheln mit Elementen befüllen
    for (int count = 0; count < numRoads; count++)
    { // Alle StraÃn durchlaufen
        Road *current_road = (Instance())->getRoad(count);
        double length = 0.0;

        while (length < current_road->getLength())
        { //s-Koordinate der jeweiligen StraÃ abfahren
            RoadLineSegment *rls = new RoadLineSegment();
            double x = current_road->getCenterLinePoint(length)[0]; // s-Koordinate in Weltkoordinaten wandeln
            double y = current_road->getCenterLinePoint(length)[1];

            //Ermitteln der Kachel, in welcher sich die aktuelle StraÃ an der Position s (bzw. x/y in WK) befindet.
            osg::Vec2d current_tile = get_tile(x, y);
            if (check_position(current_tile[0], current_tile[1]))
            {
                std::list<RoadLineSegment *>::iterator it;
                bool inList = false;

                // alle Elementer der Liste innerhalb EINER Kachel durchgehen
                for (it = ((rls_vector[current_tile[0]])[current_tile[1]]).begin(); it != ((rls_vector[current_tile[0]])[current_tile[1]]).end(); ++it)
                {
                    if (strcmp((((*it)->getRoad())->getId()).c_str(), (current_road->getId()).c_str()) == 0)
                    {
                        (*it)->check_s(length); //falls die StraÃ vorhanden ist, soll s überprüft werden um s_min und s_max innerhalb der aktuellen Kachel zu bestimmen
                        inList = true;
                        break;
                    }
                }
                if (inList == false)
                { //falls die StraÃ noch nicht in der Liste ist, wird sie angfügt
                    rls->setRoad(current_road);
                    rls->check_s(length);
                    (rls_vector.at(current_tile[0])).at(current_tile[1]).push_back(rls);
                }
            }
            length += 0.5;
        }
    }
    /*
 *         if (_tiles_y <= 500 && _tiles_x <= 500) {
 *                         //for (int i=0; i<=_tiles_y;i++) {
 *                                         for (int i=_tiles_y; i>=0;i--) {
 *                                                                 for (int j=0; j<=_tiles_x;j++) {
 *                                                                                                 if (((rls_vector.at(j)).at(i)).size() == 0) std::cout << "-";
 *                                                                                                                                 else std::cout << ((rls_vector.at(j)).at(i)).size();
 *                                                                                                                                                         }
 *                                                                                                                                                                                 std::cout << std::endl;
 *                                                                                                                                                                                                 }
 *                                                                                                                                                                                                         }
 *                                                                                                                                                                                                                 std::cout << "... delta x: " << delta_x << " | delta y: " << delta_y << std::endl << " _tiles_x " << _tiles_x  << " _tiles_y " << _tiles_y << std::endl;
 *                                                                                                                                                                                                                 */
}
std::list<RoadLineSegment *> RoadSystem::getRLS_List(int x, int y)
{
    //std::cout << " ------- getRLS_List ------ || x: " << x << " , y: " << y << " (" << _tiles_x << "," << _tiles_y << ")" << std::endl;
    return ((rls_vector.at(x)).at(y));
}

bool RoadSystem::check_position(int x, int y)
{
    if (x > (_tiles_x) || x < 0 || y > (_tiles_y) || y < 0)
    {
        //std::cout << " XXXXX Kachel existiert nicht !!! XXXX " << std::endl;
        return false;
    }
    return true;
}

osg::Vec2d RoadSystem::get_tile(double x, double y)
{
    //if (x < x_min || x > x_max || y < y_min || y > y_max)return osg::Vec2(-1,-1);
    //        //else {
    current_tile_x = (int)(x - x_min) / _tile_width;
    current_tile_y = (int)(y - y_min) / _tile_width;
    return osg::Vec2(current_tile_x, current_tile_y);
    //}
}
