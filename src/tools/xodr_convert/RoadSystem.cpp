/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMWriter.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/util/PlatformUtils.hpp>

#include "RoadSystem.h"

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

void RoadSystem::addRoad(Road *road)
{
    roadVector.push_back(road);
    roadIdMap[road->getId()] = road;
}

void RoadSystem::addJunction(Junction *junction)
{
    junctionVector.push_back(junction);
    junctionIdMap[junction->getId()] = junction;
}

Road *RoadSystem::getRoad(int i)
{
    return roadVector[i];
}
int RoadSystem::getNumRoads()
{
    return roadVector.size();
}

Junction *RoadSystem::getJunction(int i)
{
    return junctionVector[i];
}

int RoadSystem::getNumJunctions()
{
    return junctionVector.size();
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
    xercesc::DOMElement *rootElement = NULL;
    if (xmlDoc)
    {
        rootElement = xmlDoc->getDocumentElement();
    }

    if (rootElement)
    {
        Road *road;
        std::map<Road *, xercesc::DOMElement *> roadDOMMap;
        xercesc::DOMNodeList *documentChildrenList = rootElement->getChildNodes();
        xercesc::DOMElement *documentChildElement;
        for (int childIndex = 0; childIndex < documentChildrenList->getLength(); ++childIndex)
        {
            documentChildElement = dynamic_cast<xercesc::DOMElement *>(documentChildrenList->item(childIndex));
            if (documentChildElement && xercesc::XMLString::compareString(documentChildElement->getTagName(), xercesc::XMLString::transcode("road")) == 0)
            {
                std::string roadIdString = xercesc::XMLString::transcode(documentChildElement->getAttribute(xercesc::XMLString::transcode("id")));
                std::string roadNameString = xercesc::XMLString::transcode(documentChildElement->getAttribute(xercesc::XMLString::transcode("name")));
                double roadLength = atof(xercesc::XMLString::transcode(documentChildElement->getAttribute(xercesc::XMLString::transcode("length"))));
                road = new Road(roadIdString, roadNameString, roadLength);
                addRoad(road);
                roadDOMMap[road] = documentChildElement;

                xercesc::DOMNodeList *roadChildrenList = documentChildElement->getChildNodes();
                xercesc::DOMElement *roadChildElement;
                for (int childIndex = 0; childIndex < roadChildrenList->getLength(); ++childIndex)
                {
                    roadChildElement = dynamic_cast<xercesc::DOMElement *>(roadChildrenList->item(childIndex));
                    if (!roadChildElement)
                    {
                        //std::cerr << "A not-an-element in road tag";
                        //std::cerr << ", type of node: " << xercesc::XMLString::transcode(roadChildElement->getNodeName()) << std::endl;
                    }
                    else if (xercesc::XMLString::compareString(roadChildElement->getTagName(), xercesc::XMLString::transcode("type")) == 0)
                    {
                        double typeStart;
                        std::string typeName;

                        typeStart = atof(xercesc::XMLString::transcode(roadChildElement->getAttribute(xercesc::XMLString::transcode("s"))));
                        typeName = xercesc::XMLString::transcode(roadChildElement->getAttribute(xercesc::XMLString::transcode("type")));
                        road->addRoadType(typeStart, typeName);
                    }
                    else if (xercesc::XMLString::compareString(roadChildElement->getTagName(), xercesc::XMLString::transcode("planView")) == 0)
                    {
                        double geometryStart;
                        double geometryX;
                        double geometryY;
                        double geometryHdg;
                        double geometryLength;
                        xercesc::DOMNodeList *planViewChildrenList = roadChildElement->getChildNodes();
                        xercesc::DOMElement *planViewChildElement;
                        for (int childIndex = 0; childIndex < planViewChildrenList->getLength(); ++childIndex)
                        {
                            planViewChildElement = dynamic_cast<xercesc::DOMElement *>(planViewChildrenList->item(childIndex));
                            if (planViewChildElement && xercesc::XMLString::compareString(planViewChildElement->getTagName(), xercesc::XMLString::transcode("geometry")) == 0)
                            {
                                geometryStart = atof(xercesc::XMLString::transcode(planViewChildElement->getAttribute(xercesc::XMLString::transcode("s"))));
                                geometryX = atof(xercesc::XMLString::transcode(planViewChildElement->getAttribute(xercesc::XMLString::transcode("x"))));
                                geometryY = atof(xercesc::XMLString::transcode(planViewChildElement->getAttribute(xercesc::XMLString::transcode("y"))));
                                geometryHdg = atof(xercesc::XMLString::transcode(planViewChildElement->getAttribute(xercesc::XMLString::transcode("hdg"))));
                                geometryLength = atof(xercesc::XMLString::transcode(planViewChildElement->getAttribute(xercesc::XMLString::transcode("length"))));
                                xercesc::DOMNodeList *curveList = planViewChildElement->getChildNodes();
                                xercesc::DOMElement *curveElement;
                                for (int curveIndex = 0; curveIndex < curveList->getLength(); ++curveIndex)
                                {
                                    curveElement = dynamic_cast<xercesc::DOMElement *>(curveList->item(curveIndex));
                                    if (!curveElement)
                                    {
                                        //std::cerr << "A not-an-element in plan view geometry tag with start s: " << geometryStart;
                                        //std::cerr << ", type of node: " << xercesc::XMLString::transcode(curveElement->getNodeName()) << std::endl;
                                        //std::cerr << "Content of node: " << xercesc::XMLString::transcode(curveElement->getTextContent()) << std::endl;
                                    }
                                    else if (xercesc::XMLString::compareString(curveElement->getTagName(), xercesc::XMLString::transcode("line")) == 0)
                                    {
                                        //std::cerr << "Added line geometry at s: " << geometryStart << std::endl;
                                        road->addPlanViewGeometryLine(geometryStart, geometryLength, geometryX, geometryY, geometryHdg);
                                    }
                                    else if (xercesc::XMLString::compareString(curveElement->getTagName(), xercesc::XMLString::transcode("spiral")) == 0)
                                    {
                                        double curveCurvStart = atof(xercesc::XMLString::transcode(curveElement->getAttribute(xercesc::XMLString::transcode("curvStart"))));
                                        double curveCurvEnd = atof(xercesc::XMLString::transcode(curveElement->getAttribute(xercesc::XMLString::transcode("curvEnd"))));
                                        road->addPlanViewGeometrySpiral(geometryStart, geometryLength, geometryX, geometryY, geometryHdg, curveCurvStart, curveCurvEnd);
                                    }
                                    else if (xercesc::XMLString::compareString(curveElement->getTagName(), xercesc::XMLString::transcode("arc")) == 0)
                                    {
                                        double curveCurvature = atof(xercesc::XMLString::transcode(curveElement->getAttribute(xercesc::XMLString::transcode("curvature"))));
                                        road->addPlanViewGeometryArc(geometryStart, geometryLength, geometryX, geometryY, geometryHdg, curveCurvature);
                                    }
                                    else if (xercesc::XMLString::compareString(curveElement->getTagName(), xercesc::XMLString::transcode("poly3")) == 0)
                                    {
                                        double curveA = atof(xercesc::XMLString::transcode(curveElement->getAttribute(xercesc::XMLString::transcode("a"))));
                                        double curveB = atof(xercesc::XMLString::transcode(curveElement->getAttribute(xercesc::XMLString::transcode("b"))));
                                        double curveC = atof(xercesc::XMLString::transcode(curveElement->getAttribute(xercesc::XMLString::transcode("c"))));
                                        double curveD = atof(xercesc::XMLString::transcode(curveElement->getAttribute(xercesc::XMLString::transcode("d"))));
                                        road->addPlanViewGeometryPolynom(geometryStart, geometryLength, geometryX, geometryY, geometryHdg, curveA, curveB, curveC, curveD);
                                    }
                                }
                            }
                        }
                    }
                    else if (xercesc::XMLString::compareString(roadChildElement->getTagName(), xercesc::XMLString::transcode("elevationProfile")) == 0)
                    {
                        xercesc::DOMNodeList *elevationProfileChildrenList = roadChildElement->getChildNodes();
                        xercesc::DOMElement *elevationProfileChildElement;
                        for (int childIndex = 0; childIndex < elevationProfileChildrenList->getLength(); ++childIndex)
                        {
                            elevationProfileChildElement = dynamic_cast<xercesc::DOMElement *>(elevationProfileChildrenList->item(childIndex));
                            if (elevationProfileChildElement && xercesc::XMLString::compareString(elevationProfileChildElement->getTagName(), xercesc::XMLString::transcode("elevation")) == 0)
                            {
                                double elevationStart = atof(xercesc::XMLString::transcode(elevationProfileChildElement->getAttribute(xercesc::XMLString::transcode("s"))));
                                double elevationA = atof(xercesc::XMLString::transcode(elevationProfileChildElement->getAttribute(xercesc::XMLString::transcode("a"))));
                                double elevationB = atof(xercesc::XMLString::transcode(elevationProfileChildElement->getAttribute(xercesc::XMLString::transcode("b"))));
                                double elevationC = atof(xercesc::XMLString::transcode(elevationProfileChildElement->getAttribute(xercesc::XMLString::transcode("c"))));
                                double elevationD = atof(xercesc::XMLString::transcode(elevationProfileChildElement->getAttribute(xercesc::XMLString::transcode("d"))));
                                road->addElevationPolynom(elevationStart, elevationA, elevationB, elevationC, elevationD);
                            }
                        }
                    }
                    else if (xercesc::XMLString::compareString(roadChildElement->getTagName(), xercesc::XMLString::transcode("lateralProfile")) == 0)
                    {
                        xercesc::DOMNodeList *lateralProfileChildrenList = roadChildElement->getChildNodes();
                        xercesc::DOMElement *lateralProfileChildElement;
                        for (int childIndex = 0; childIndex < lateralProfileChildrenList->getLength(); ++childIndex)
                        {
                            lateralProfileChildElement = dynamic_cast<xercesc::DOMElement *>(lateralProfileChildrenList->item(childIndex));
                            if (lateralProfileChildElement && xercesc::XMLString::compareString(lateralProfileChildElement->getTagName(), xercesc::XMLString::transcode("superelevation")) == 0)
                            {
                                double superelevationStart = atof(xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(xercesc::XMLString::transcode("s"))));
                                double superelevationA = atof(xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(xercesc::XMLString::transcode("a"))));
                                double superelevationB = atof(xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(xercesc::XMLString::transcode("b"))));
                                double superelevationC = atof(xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(xercesc::XMLString::transcode("c"))));
                                double superelevationD = atof(xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(xercesc::XMLString::transcode("d"))));
                                road->addSuperelevationPolynom(superelevationStart, superelevationA, superelevationB, superelevationC, superelevationD);
                            }
                            else if (lateralProfileChildElement && xercesc::XMLString::compareString(lateralProfileChildElement->getTagName(), xercesc::XMLString::transcode("crossfall")) == 0)
                            {
                                double crossfallStart = atof(xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(xercesc::XMLString::transcode("s"))));
                                double crossfallA = atof(xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(xercesc::XMLString::transcode("a"))));
                                double crossfallB = atof(xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(xercesc::XMLString::transcode("b"))));
                                double crossfallC = atof(xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(xercesc::XMLString::transcode("c"))));
                                double crossfallD = atof(xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(xercesc::XMLString::transcode("d"))));
                                std::string crossfallSide(xercesc::XMLString::transcode(lateralProfileChildElement->getAttribute(xercesc::XMLString::transcode("side"))));
                                road->addCrossfallPolynom(crossfallStart, crossfallA, crossfallB, crossfallC, crossfallD, crossfallSide);
                            }
                        }
                    }

                    else if (xercesc::XMLString::compareString(roadChildElement->getTagName(), xercesc::XMLString::transcode("lanes")) == 0)
                    {
                        xercesc::DOMNodeList *lanesChildrenList = roadChildElement->getChildNodes();
                        xercesc::DOMElement *lanesChildElement;
                        for (int childIndex = 0; childIndex < lanesChildrenList->getLength(); ++childIndex)
                        {
                            lanesChildElement = dynamic_cast<xercesc::DOMElement *>(lanesChildrenList->item(childIndex));
                            if (lanesChildElement && xercesc::XMLString::compareString(lanesChildElement->getTagName(), xercesc::XMLString::transcode("laneSection")) == 0)
                            {
                                double laneSectionStart = atof(xercesc::XMLString::transcode(lanesChildElement->getAttribute(xercesc::XMLString::transcode("s"))));
                                LaneSection *section = new LaneSection(laneSectionStart);
                                road->addLaneSection(section);
                                xercesc::DOMNodeList *laneSectionChildrenList = lanesChildElement->getChildNodes();
                                xercesc::DOMElement *laneSectionChildElement;
                                for (int childIndex = 0; childIndex < laneSectionChildrenList->getLength(); ++childIndex)
                                {
                                    laneSectionChildElement = dynamic_cast<xercesc::DOMElement *>(laneSectionChildrenList->item(childIndex));
                                    if (laneSectionChildElement && (xercesc::XMLString::compareString(laneSectionChildElement->getTagName(), xercesc::XMLString::transcode("left")) == 0
                                                                    || xercesc::XMLString::compareString(laneSectionChildElement->getTagName(), xercesc::XMLString::transcode("center")) == 0
                                                                    || xercesc::XMLString::compareString(laneSectionChildElement->getTagName(), xercesc::XMLString::transcode("right")) == 0))
                                    {
                                        xercesc::DOMNodeList *laneList = laneSectionChildElement->getChildNodes();
                                        xercesc::DOMElement *laneElement;
                                        for (int childIndex = 0; childIndex < laneList->getLength(); ++childIndex)
                                        {
                                            laneElement = dynamic_cast<xercesc::DOMElement *>(laneList->item(childIndex));
                                            if (laneElement && xercesc::XMLString::compareString(laneElement->getTagName(), xercesc::XMLString::transcode("lane")) == 0)
                                            {
                                                int laneId = atoi(xercesc::XMLString::transcode(laneElement->getAttribute(xercesc::XMLString::transcode("id"))));
                                                std::string laneType(xercesc::XMLString::transcode(laneElement->getAttribute(xercesc::XMLString::transcode("type"))));
                                                std::string laneLevel(xercesc::XMLString::transcode(laneElement->getAttribute(xercesc::XMLString::transcode("level"))));
                                                Lane *lane = new Lane(laneId, laneType, laneLevel);
                                                section->addLane(lane);
                                                xercesc::DOMNodeList *laneChildrenList = laneElement->getChildNodes();
                                                xercesc::DOMElement *laneChildElement;
                                                for (int childIndex = 0; childIndex < laneChildrenList->getLength(); ++childIndex)
                                                {
                                                    laneChildElement = dynamic_cast<xercesc::DOMElement *>(laneChildrenList->item(childIndex));
                                                    if (!laneChildElement)
                                                    {
                                                    }
                                                    else if (xercesc::XMLString::compareString(laneChildElement->getTagName(), xercesc::XMLString::transcode("link")) == 0)
                                                    {
                                                        xercesc::DOMNodeList *linkChildrenList = laneChildElement->getChildNodes();
                                                        xercesc::DOMElement *linkChildElement;
                                                        for (int childIndex = 0; childIndex < linkChildrenList->getLength(); ++childIndex)
                                                        {
                                                            linkChildElement = dynamic_cast<xercesc::DOMElement *>(linkChildrenList->item(childIndex));
                                                            if (!linkChildElement)
                                                            {
                                                            }
                                                            else if (xercesc::XMLString::compareString(linkChildElement->getTagName(), xercesc::XMLString::transcode("predecessor")) == 0)
                                                            {
                                                                int predecessorId = atoi(xercesc::XMLString::transcode(linkChildElement->getAttribute(xercesc::XMLString::transcode("id"))));
                                                                lane->setPredecessor(predecessorId);
                                                            }
                                                            else if (xercesc::XMLString::compareString(linkChildElement->getTagName(), xercesc::XMLString::transcode("successor")) == 0)
                                                            {
                                                                int successorId = atoi(xercesc::XMLString::transcode(linkChildElement->getAttribute(xercesc::XMLString::transcode("id"))));
                                                                lane->setSuccessor(successorId);
                                                            }
                                                        }
                                                    }
                                                    else if (xercesc::XMLString::compareString(laneChildElement->getTagName(), xercesc::XMLString::transcode("width")) == 0)
                                                    {
                                                        double widthStart = atof(xercesc::XMLString::transcode(laneChildElement->getAttribute(xercesc::XMLString::transcode("sOffset"))));
                                                        double widthA = atof(xercesc::XMLString::transcode(laneChildElement->getAttribute(xercesc::XMLString::transcode("a"))));
                                                        double widthB = atof(xercesc::XMLString::transcode(laneChildElement->getAttribute(xercesc::XMLString::transcode("b"))));
                                                        double widthC = atof(xercesc::XMLString::transcode(laneChildElement->getAttribute(xercesc::XMLString::transcode("c"))));
                                                        double widthD = atof(xercesc::XMLString::transcode(laneChildElement->getAttribute(xercesc::XMLString::transcode("d"))));
                                                        lane->addWidth(widthStart, widthA, widthB, widthC, widthD);
                                                    }
                                                    else if (xercesc::XMLString::compareString(laneChildElement->getTagName(), xercesc::XMLString::transcode("roadMark")) == 0)
                                                    {
                                                        double roadMarkStart = atof(xercesc::XMLString::transcode(laneChildElement->getAttribute(xercesc::XMLString::transcode("sOffset"))));
                                                        double roadMarkWidth = atof(xercesc::XMLString::transcode(laneChildElement->getAttribute(xercesc::XMLString::transcode("width"))));
                                                        std::string roadMarkType(xercesc::XMLString::transcode(laneChildElement->getAttribute(xercesc::XMLString::transcode("type"))));
                                                        std::string roadMarkWeight(xercesc::XMLString::transcode(laneChildElement->getAttribute(xercesc::XMLString::transcode("weight"))));
                                                        std::string roadMarkColor(xercesc::XMLString::transcode(laneChildElement->getAttribute(xercesc::XMLString::transcode("color"))));
                                                        std::string roadMarkLaneChange(xercesc::XMLString::transcode(laneChildElement->getAttribute(xercesc::XMLString::transcode("laneChange"))));
                                                        lane->addRoadMark(new RoadMark(roadMarkStart, roadMarkWidth, roadMarkType, roadMarkWeight, roadMarkColor, roadMarkLaneChange));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for (int childIndex = 0; childIndex < documentChildrenList->getLength(); ++childIndex)
        {
            documentChildElement = dynamic_cast<xercesc::DOMElement *>(documentChildrenList->item(childIndex));
            if (documentChildElement && xercesc::XMLString::compareString(documentChildElement->getTagName(), xercesc::XMLString::transcode("junction")) == 0)
            {
                std::string junctionId(xercesc::XMLString::transcode(documentChildElement->getAttribute(xercesc::XMLString::transcode("id"))));
                std::string junctionName(xercesc::XMLString::transcode(documentChildElement->getAttribute(xercesc::XMLString::transcode("name"))));
                Junction *junction = new Junction(junctionId, junctionName);
                addJunction(junction);

                xercesc::DOMNodeList *junctionChildrenList = documentChildElement->getChildNodes();
                xercesc::DOMElement *junctionChildElement;
                for (int childIndex = 0; childIndex < junctionChildrenList->getLength(); ++childIndex)
                {
                    junctionChildElement = dynamic_cast<xercesc::DOMElement *>(junctionChildrenList->item(childIndex));
                    if (!junctionChildElement)
                    {
                    }
                    else if (xercesc::XMLString::compareString(junctionChildElement->getTagName(), xercesc::XMLString::transcode("connection")) == 0)
                    {
                        std::string connectionId(xercesc::XMLString::transcode(junctionChildElement->getAttribute(xercesc::XMLString::transcode("id"))));
                        std::string inRoadString(xercesc::XMLString::transcode(junctionChildElement->getAttribute(xercesc::XMLString::transcode("incomingRoad"))));
                        std::string connPathString(xercesc::XMLString::transcode(junctionChildElement->getAttribute(xercesc::XMLString::transcode("connectingRoad"))));
                        std::string contactPoint(xercesc::XMLString::transcode(junctionChildElement->getAttribute(xercesc::XMLString::transcode("contactPoint"))));

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

                                        PathConnection *conn = new PathConnection(connectionId, inRoad, connPath, direction);
                                        junction->addPathConnection(conn);

                                        xercesc::DOMNodeList *connectionChildrenList = junctionChildElement->getChildNodes();
                                        xercesc::DOMElement *connectionChildElement;
                                        for (int childIndex = 0; childIndex < connectionChildrenList->getLength(); ++childIndex)
                                        {
                                            connectionChildElement = dynamic_cast<xercesc::DOMElement *>(connectionChildrenList->item(childIndex));
                                            if (!connectionChildElement)
                                            {
                                            }
                                            else if (xercesc::XMLString::compareString(connectionChildElement->getTagName(), xercesc::XMLString::transcode("laneLink")) == 0)
                                            {
                                                int from = atoi(xercesc::XMLString::transcode(connectionChildElement->getAttribute(xercesc::XMLString::transcode("from"))));
                                                int to = atoi(xercesc::XMLString::transcode(connectionChildElement->getAttribute(xercesc::XMLString::transcode("to"))));
                                                conn->addLaneLink(from, to);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for (int roadIndex = 0; roadIndex < roadVector.size(); ++roadIndex)
        {
            Road *road = roadVector[roadIndex];
            std::string junctionId(xercesc::XMLString::transcode(roadDOMMap[road]->getAttribute(xercesc::XMLString::transcode("junction"))));
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
            for (int childIndex = 0; childIndex < roadChildrenList->getLength(); ++childIndex)
            {
                roadChildElement = dynamic_cast<xercesc::DOMElement *>(roadChildrenList->item(childIndex));
                if (!roadChildElement)
                {
                }
                else if (xercesc::XMLString::compareString(roadChildElement->getTagName(), xercesc::XMLString::transcode("link")) == 0)
                {
                    xercesc::DOMNodeList *linkChildrenList = roadChildElement->getChildNodes();
                    xercesc::DOMElement *linkChildElement;
                    for (int childIndex = 0; childIndex < linkChildrenList->getLength(); ++childIndex)
                    {
                        linkChildElement = dynamic_cast<xercesc::DOMElement *>(linkChildrenList->item(childIndex));
                        if (!linkChildElement)
                        {
                        }
                        else if (xercesc::XMLString::compareString(linkChildElement->getTagName(), xercesc::XMLString::transcode("predecessor")) == 0)
                        {
                            std::string type(xercesc::XMLString::transcode(linkChildElement->getAttribute(xercesc::XMLString::transcode("elementType"))));
                            std::string id(xercesc::XMLString::transcode(linkChildElement->getAttribute(xercesc::XMLString::transcode("elementId"))));
                            std::string contactPoint(xercesc::XMLString::transcode(linkChildElement->getAttribute(xercesc::XMLString::transcode("contactPoint"))));

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
                                else
                                {
                                    std::cerr << "Road " << road->getName() << ": Predecessor id: " << id << ": No direction defined..." << std::endl;
                                }
                                road->setPredecessorConnection(new TarmacConnection(tarmac, direction));
                            }
                        }
                        else if (xercesc::XMLString::compareString(linkChildElement->getTagName(), xercesc::XMLString::transcode("successor")) == 0)
                        {
                            std::string type(xercesc::XMLString::transcode(linkChildElement->getAttribute(xercesc::XMLString::transcode("elementType"))));
                            std::string id(xercesc::XMLString::transcode(linkChildElement->getAttribute(xercesc::XMLString::transcode("elementId"))));
                            std::string contactPoint(xercesc::XMLString::transcode(linkChildElement->getAttribute(xercesc::XMLString::transcode("contactPoint"))));
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
                                else
                                {
                                    std::cerr << "Road " << road->getName() << ": Successor id: " << id << ": No direction defined..." << std::endl;
                                }
                                road->setSuccessorConnection(new TarmacConnection(tarmac, direction));
                            }
                        }
                    }
                }
            }
        }
    }

    xercesc::XMLPlatformUtils::Terminate();
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

    return os;
}
