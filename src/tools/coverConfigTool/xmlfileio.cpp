/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 ** xmlfileio.cpp
 ** 2004-02-04, Matthias Feurer
 ****************************************************************************/

#include <qstring.h>
#include <qfile.h>
#include <qtextstream.h>
#include <qdom.h>
#include "xmlfileio.h"
#include "covergeneral.h"
#include "host.h"
#include "projectionarea.h"
//#include <iostream>
#include "covise.h"

XMLFileIO::XMLFileIO()
{
    hostMap = 0;
    projMap = 0;
    genSets = 0;
    fileName = QString();
    domDoc.setContent(QString("<COVERConfigTool/>"));
}

HostMap *XMLFileIO::getHostMap()
{
    return hostMap;
}

ProjectionAreaMap *XMLFileIO::getProjMap()
{
    return projMap;
}

CoverGeneral *XMLFileIO::getGeneralSettings()
{
    return genSets;
}

Tracking *XMLFileIO::getTracking()
{
    return tracking;
}

void XMLFileIO::setHostMap(HostMap *hm)
{
    hostMap = hm;
}

void XMLFileIO::setProjMap(ProjectionAreaMap *pm)
{
    projMap = pm;
}

void XMLFileIO::setGeneralSettings(CoverGeneral *gs)
{
    genSets = gs;
}

void XMLFileIO::setTracking(Tracking *t)
{
    tracking = t;
}

bool XMLFileIO::saveXMLFile(QString name)
{
    fileName = name;
    QDomDocument newDoc("COVERConfigTool");
    QDomElement root = newDoc.createElement("COVERConfigTool");
    newDoc.appendChild(root);

    // first add projection areas...
    QDomElement projMapElement = newDoc.createElement("ProjMap");
    root.appendChild(projMapElement);
    ProjectionAreaMap::Iterator projIt;

    // iterate over all projection areas
    for (projIt = projMap->begin(); projIt != projMap->end(); ++projIt)
    {
        QDomElement proj = newDoc.createElement("projectionArea");
        //QDomElement projWidth = newDoc.createElement("projWidth");

        projMapElement.appendChild(proj);
        //proj.appendChild(projName);

        //projName.addAttribute(projIt.data().getName());

        //proj.appendChild(projWidth);
        //projName.addAttribute(projIt.data().getWidth());

        //proj.appendChild(projHeight);
        //projName.addAttribute(projIt.data().getHeight());

        proj.addAttribute("name", projIt.data().getName());
        proj.addAttribute("type", projIt.data().getType());
        proj.addAttribute("width", projIt.data().getWidth());
        proj.addAttribute("height", projIt.data().getHeight());
        proj.addAttribute("originX", projIt.data().getOriginX());
        proj.addAttribute("originY", projIt.data().getOriginY());
        proj.addAttribute("originZ", projIt.data().getOriginZ());
        proj.addAttribute("rotation_h", projIt.data().getRotation_h());
        proj.addAttribute("rotation_p", projIt.data().getRotation_p());
        proj.addAttribute("rotation_r", projIt.data().getRotation_r());
    }

    // now add host information
    QDomElement hostMapElement = newDoc.createElement("HostMap");
    root.appendChild(hostMapElement);
    PipeMap *pipeMap;
    WindowMap *winMap;
    ChannelMap *chMap;
    HostMap::Iterator hostIt;
    PipeMap::Iterator pipeIt;
    WindowMap::Iterator winIt;
    ChannelMap::Iterator chIt;

    // iterate over all projection areas
    for (hostIt = hostMap->begin(); hostIt != hostMap->end(); ++hostIt)
    {
        QDomElement hostElement = newDoc.createElement("host");
        hostMapElement.appendChild(hostElement);

        hostElement.addAttribute("name", hostIt.data().getName());
        hostElement.addAttribute("isControlHost", hostIt.data().isControlHost());
        hostElement.addAttribute("isMasterHost", hostIt.data().isMasterHost());

        if (hostIt.data().isMasterHost())
            hostElement.addAttribute("masterInterface", hostIt.data().getMasterInterface());
        else
            hostElement.addAttribute("masterInterface", "");

        hostElement.addAttribute("trackingSystem", hostIt.data().getTrackingString());
        hostElement.addAttribute("monoView", hostIt.data().getMonoView());

        // get all pipes of host
        pipeMap = hostIt.data().getPipeMap();
        if (pipeMap != 0)
        {
            for (pipeIt = pipeMap->begin(); pipeIt != pipeMap->end(); ++pipeIt)
            {
                QDomElement pipeElement = newDoc.createElement("pipe");
                hostElement.appendChild(pipeElement);

                pipeElement.addAttribute("name", pipeIt.key());
                pipeElement.addAttribute("index", pipeIt.data().getIndex());
                pipeElement.addAttribute("hardPipe", pipeIt.data().getHardPipe());
                pipeElement.addAttribute("display", pipeIt.data().getDisplay());

                winMap = pipeIt.data().getWindowMap();
                if (winMap != 0)
                {
                    for (winIt = winMap->begin(); winIt != winMap->end(); ++winIt)
                    {
                        QDomElement winElement = newDoc.createElement("window");
                        pipeElement.appendChild(winElement);

                        winElement.addAttribute("index", winIt.data().getIndex());
                        winElement.addAttribute("name", winIt.data().getName());
                        winElement.addAttribute("softPipeNo", winIt.data().getSoftPipeNo());
                        winElement.addAttribute("originX", winIt.data().getOriginX());
                        winElement.addAttribute("originY", winIt.data().getOriginY());
                        winElement.addAttribute("width", winIt.data().getWidth());
                        winElement.addAttribute("height", winIt.data().getHeight());

                        chMap = winIt.data().getChannelMap();
                        if (chMap != 0)
                        {
                            for (chIt = chMap->begin(); chIt != chMap->end(); ++chIt)
                            {
                                QDomElement chElement = newDoc.createElement("channel");
                                winElement.appendChild(chElement);

                                chElement.addAttribute("index", chIt.data().getIndex());
                                chElement.addAttribute("name", chIt.data().getName());
                                chElement.addAttribute("left", chIt.data().getLeft());
                                chElement.addAttribute("right", chIt.data().getRight());
                                chElement.addAttribute("bottom", chIt.data().getBottom());
                                chElement.addAttribute("top", chIt.data().getTop());

                                if (chIt.data().getProjectionArea() != 0)
                                {
                                    chElement.addAttribute("projectionArea",
                                                           chIt.data().getProjectionArea()->getName());
                                }
                                else
                                    chElement.addAttribute("projectionArea", "");
                            }
                        }
                    }
                }
            }
        }
    }

    // now add general information
    QDomElement generalElement = newDoc.createElement("GeneralSettings");
    root.appendChild(generalElement);

    if (genSets != 0)
    {
        generalElement.addAttribute("stereoMode", genSets->getStereoMode());

        generalElement.addAttribute("viewerPosX", genSets->getViewerPosX());
        generalElement.addAttribute("viewerPosY", genSets->getViewerPosY());
        generalElement.addAttribute("viewerPosZ", genSets->getViewerPosZ());

        generalElement.addAttribute("floorHeight", genSets->getFloorHeight());
        generalElement.addAttribute("stepSize", genSets->getStepSize());

        generalElement.addAttribute("menuPosX", genSets->getMenuPosX());
        generalElement.addAttribute("menuPosY", genSets->getMenuPosY());
        generalElement.addAttribute("menuPosZ", genSets->getMenuPosZ());

        generalElement.addAttribute("menuOrient_h", genSets->getMenuOrient_h());
        generalElement.addAttribute("menuOrient_p", genSets->getMenuOrient_p());
        generalElement.addAttribute("menuOrient_r", genSets->getMenuOrient_r());

        generalElement.addAttribute("menuSize", genSets->getMenuSize());
        generalElement.addAttribute("sceneSize", genSets->getSceneSize());

        // MultiPC
        generalElement.addAttribute("syncMode", genSets->getSyncModeString());
        generalElement.addAttribute("syncProcess", genSets->getSyncProcessString());
        generalElement.addAttribute("serialDevice", genSets->getSerialDevice());
    }

    // add information about tracking system
    QDomElement trackerElement = newDoc.createElement("TrackerConfig");
    root.appendChild(trackerElement);

    if (tracking != 0)
    {
        trackerElement.addAttribute("noConnectedSensors", tracking->getNoSensors());
        trackerElement.addAttribute("adrHeadSensor", tracking->getAdrHeadSensor());
        trackerElement.addAttribute("adrHandSensor", tracking->getAdrHandSensor());

        trackerElement.addAttribute("transmitterOffsetX", tracking->getTransmitterOffsetX());
        trackerElement.addAttribute("transmitterOffsetY", tracking->getTransmitterOffsetY());
        trackerElement.addAttribute("transmitterOffsetZ", tracking->getTransmitterOffsetZ());
        trackerElement.addAttribute("transmitterOrientH", tracking->getTransmitterOrientH());
        trackerElement.addAttribute("transmitterOrientP", tracking->getTransmitterOrientP());
        trackerElement.addAttribute("transmitterOrientR", tracking->getTransmitterOrientR());

        trackerElement.addAttribute("headSensorOffsetX", tracking->getHeadSensorOffsetX());
        trackerElement.addAttribute("headSensorOffsetY", tracking->getHeadSensorOffsetY());
        trackerElement.addAttribute("headSensorOffsetZ", tracking->getHeadSensorOffsetZ());
        trackerElement.addAttribute("headSensorOrientH", tracking->getHeadSensorOrientH());
        trackerElement.addAttribute("headSensorOrientP", tracking->getHeadSensorOrientP());
        trackerElement.addAttribute("headSensorOrientR", tracking->getHeadSensorOrientR());

        trackerElement.addAttribute("handSensorOffsetX", tracking->getHandSensorOffsetX());
        trackerElement.addAttribute("handSensorOffsetY", tracking->getHandSensorOffsetY());
        trackerElement.addAttribute("handSensorOffsetZ", tracking->getHandSensorOffsetZ());
        trackerElement.addAttribute("handSensorOrientH", tracking->getHandSensorOrientH());
        trackerElement.addAttribute("handSensorOrientP", tracking->getHandSensorOrientP());
        trackerElement.addAttribute("handSensorOrientR", tracking->getHandSensorOrientR());

        trackerElement.addAttribute("trackerType", tracking->getTrackerType());
        trackerElement.addAttribute("xDir", tracking->getXDir());
        trackerElement.addAttribute("yDir", tracking->getYDir());
        trackerElement.addAttribute("zDir", tracking->getZDir());

        trackerElement.addAttribute("fieldCorrectionX", tracking->getLinearMagneticFieldCorrectionX());
        trackerElement.addAttribute("fieldCorrectionY", tracking->getLinearMagneticFieldCorrectionY());
        trackerElement.addAttribute("fieldCorrectionZ", tracking->getLinearMagneticFieldCorrectionZ());

        trackerElement.addAttribute("interpolationFile", tracking->getInterpolationFile());
        trackerElement.addAttribute("debugTracking", tracking->getDebugTracking());
        trackerElement.addAttribute("debugButtons", tracking->getDebugButtons());
        trackerElement.addAttribute("debugStation", tracking->getDebugStation());
    }

    //  QString xmlString = newDoc.toString();
    QString xmlString = newDoc.toString();

    QFile file(fileName);
    if (file.open(IO_WriteOnly))
    {
        QTextStream stream(&file);

        stream << xmlString << "\n";
        file.close();
        return true;
    }

    return false;
}

/*------------------------------------------------------------------------------
 ** loadXMLFile():
 **   loads config information from xml-file.
 **
 **   Parameters:
 **     name:              the file name of the xml file
 **     message:           return message.
 **
 **   Return value:        message if we could load xml file.
 **
 **   PRE:                 all parameter objects are not null!
 **
 **   POST:                the objects contain correct informations.
 **
-------------------------------------------------------------------------------*/
bool XMLFileIO::loadXMLFile(QString name,
                            QString *message)
{
    XMLFileIO::projMap = new ProjectionAreaMap();
    XMLFileIO::hostMap = new HostMap();
    XMLFileIO::genSets = new CoverGeneral();
    XMLFileIO::tracking = new Tracking();

    fileName = name;
    QDomDocument doc;

    QFile file(fileName);
    if (!file.open(IO_ReadOnly))
    {
        (*message) = "Could not open file.";
        return false;
    }
    if (!doc.setContent(&file))
    {
        file.close();
        (*message) = "Could not parse document";
        return false;
    }
    file.close();

    // read DOM

    QDomElement root = doc.documentElement();
    traverseNode(root);
    //cout<<"after traverseNode.: "<<endl;

    //saveXMLFile("test_load.xml");
    (*message) = "Settings successfully loaded.";

    return true;
}

// private

/*------------------------------------------------------------------------------
 ** traverseNode():
 **   traverses a QDomNode recurively.
 **
 **   Parameters:
 **     node:              the node to traverse
 **
-------------------------------------------------------------------------------*/
void XMLFileIO::traverseNode(const QDomNode &node)
{
    QDomNode child = node.firstChild();
    while (!child.isNull())
    {
        if (child.isElement())
        {
            //cout<<"child is element. "<<endl;
            QDomElement childElement = child.toElement();
            if (childElement.tagName() == "ProjMap")
            {
                //cout<<"child is ProjMap. "<<endl;
            }
            else if (childElement.tagName() == "projectionArea")
            {
                //cout<<"child is projArea. "<<endl;
                ProjectionArea proj = ProjectionArea();
                proj.setName(childElement.attribute("name"));
                proj.setType((ProjType)childElement.attribute("type").toInt());
                proj.setWidth(childElement.attribute("width").toInt());
                proj.setHeight(childElement.attribute("height").toInt());
                proj.setOrigin(childElement.attribute("originX").toInt(),
                               childElement.attribute("originY").toInt(),
                               childElement.attribute("originZ").toInt());
                proj.setRotation(childElement.attribute("rotation_h").toDouble(),
                                 childElement.attribute("rotation_p").toDouble(),
                                 childElement.attribute("rotation_r").toDouble());

                (*projMap)[proj.getName()] = proj;
            }
            else if (childElement.tagName() == "HostMap")
            {
                // only traverse children of HostMap
                //cout<<"child is HostMap. "<<endl;
            }
            else if (childElement.tagName() == "host")
            {
                //cout<<"child is Host. "<<endl;
                Host h = Host();
                h.setName(childElement.attribute("name"));
                h.setControlHost((bool)childElement.attribute("isControlHost").toInt());
                h.setMasterHost((bool)childElement.attribute("isMasterHost").toInt());
                h.setMasterInterface(childElement.attribute("masterInterface"));
                h.setTrackingSystemString(childElement.attribute("trackingSystem"));
                h.setMonoView(childElement.attribute("monoView"));

                (*hostMap)[h.getName()] = h;
            }
            else if (childElement.tagName() == "pipe")
            {
                //cout<<"child is Pipe. "<<endl;
                Host h = Host();
                Pipe p = Pipe();
                QDomElement hostEl = child.parentNode().toElement();
                h = (*hostMap)[hostEl.attribute("name")];

                //p.setName(childElement.attribute("name"));
                p.setIndex(childElement.attribute("index").toInt());
                p.setHardPipe(childElement.attribute("hardPipe").toInt());
                p.setDisplay(childElement.attribute("display"));

                (*h.getPipeMap())[childElement.attribute("name")] = p;
                (*hostMap)[h.getName()] = h;
            }
            else if (childElement.tagName() == "window")
            {
                //cout<<"child is Window. "<<endl;
                Host h = Host();
                Pipe p = Pipe();
                Window w = Window();
                QDomElement pipeEl = child.parentNode().toElement();
                QDomElement hostEl = child.parentNode().parentNode().toElement();
                h = (*hostMap)[hostEl.attribute("name")];
                p = (*h.getPipeMap())[pipeEl.attribute("name")];

                w.setIndex(childElement.attribute("index").toInt());
                w.setName(childElement.attribute("name"));
                w.setSoftPipeNo(childElement.attribute("softPipeNo").toInt());
                w.setOriginX(childElement.attribute("originX").toInt());
                w.setOriginY(childElement.attribute("originY").toInt());
                w.setWidth(childElement.attribute("width").toInt());
                w.setHeight(childElement.attribute("height").toInt());

                (*p.getWindowMap())[w.getName()] = w;
                (*h.getPipeMap())[pipeEl.attribute("name")] = p;
                (*hostMap)[h.getName()] = h;
            }
            else if (childElement.tagName() == "channel")
            {
                //cout<<"child is channel. "<<endl;
                Host h = Host();
                Pipe p = Pipe();
                Window w = Window();
                Channel ch = Channel();
                QDomElement winEl = child.parentNode().toElement();
                QDomElement pipeEl = child.parentNode().parentNode().toElement();
                QDomElement hostEl = child.parentNode().parentNode().parentNode().toElement();
                h = (*hostMap)[hostEl.attribute("name")];
                p = (*h.getPipeMap())[pipeEl.attribute("name")];
                w = (*p.getWindowMap())[winEl.attribute("name")];

                ch.setIndex(childElement.attribute("index").toInt());
                ch.setName(childElement.attribute("name"));
                ch.setLeft(childElement.attribute("left").toInt());
                ch.setRight(childElement.attribute("right").toInt());
                ch.setBottom(childElement.attribute("bottom").toInt());
                ch.setTop(childElement.attribute("top").toInt());

                // ADD PROJ AREA TO CHANNEL!!!!!!! ----> TO DO!
                if (projMap->find(childElement.attribute("projectionArea")) != projMap->end())
                {
                    ch.setProjectionArea(&(*projMap)[childElement.attribute("projectionArea")]);
                }

                (*w.getChannelMap())[ch.getName()] = ch;
                (*p.getWindowMap())[w.getName()] = w;
                (*h.getPipeMap())[pipeEl.attribute("name")] = p;
                (*hostMap)[h.getName()] = h;
            }
            else if (childElement.tagName() == "GeneralSettings")
            {
                //cout<<"child is general settings. "<<endl;
                Host h = Host();

                genSets->setStereoMode(childElement.attribute("stereoMode"));
                genSets->setViewerPosX(childElement.attribute("viewerPosX").toInt());
                genSets->setViewerPosY(childElement.attribute("viewerPosY").toInt());
                genSets->setViewerPosZ(childElement.attribute("viewerPosZ").toInt());

                genSets->setFloorHeight(childElement.attribute("floorHeight").toInt());
                genSets->setStepSize(childElement.attribute("stepSize").toInt());
                genSets->setSceneSize(childElement.attribute("sceneSize").toInt());

                genSets->setMenuPosX(childElement.attribute("menuPosX").toInt());
                genSets->setMenuPosY(childElement.attribute("menuPosY").toInt());
                genSets->setMenuPosZ(childElement.attribute("menuPosZ").toInt());

                genSets->setMenuOrient_h(childElement.attribute("menuOrient_h").toDouble());
                genSets->setMenuOrient_p(childElement.attribute("menuOrient_p").toDouble());
                genSets->setMenuOrient_r(childElement.attribute("menuOrient_r").toDouble());

                genSets->setSyncModeString(childElement.attribute("syncMode"));
                genSets->setSyncProcessString(childElement.attribute("syncProcess"));
                genSets->setSerialDevice(childElement.attribute("serialDevice"));
            }
            else if (childElement.tagName() == "TrackerConfig")
            {
                //cout<<"child is TrackerConfig. "<<endl;
                Host h = Host();

                tracking->setNoSensors(childElement.attribute("noConnectedSensors").toInt());
                tracking->setAdrHeadSensor(childElement.attribute("adrHeadSensor").toInt());
                tracking->setAdrHandSensor(childElement.attribute("adrHandSensor").toInt());

                tracking->setTransmitterOffsetX(childElement.attribute("transmitterOffsetX").toDouble());
                tracking->setTransmitterOffsetY(childElement.attribute("transmitterOffsetY").toDouble());
                tracking->setTransmitterOffsetZ(childElement.attribute("transmitterOffsetZ").toDouble());
                tracking->setTransmitterOrientH(childElement.attribute("transmitterOrientH").toDouble());
                tracking->setTransmitterOrientP(childElement.attribute("transmitterOrientP").toDouble());
                tracking->setTransmitterOrientR(childElement.attribute("transmitterOrientR").toDouble());

                tracking->setHeadSensorOffsetX(childElement.attribute("headSensorOffsetX").toDouble());
                tracking->setHeadSensorOffsetY(childElement.attribute("headSensorOffsetY").toDouble());
                tracking->setHeadSensorOffsetZ(childElement.attribute("headSensorOffsetZ").toDouble());
                tracking->setHeadSensorOrientH(childElement.attribute("headSensorOrientH").toDouble());
                tracking->setHeadSensorOrientP(childElement.attribute("headSensorOrientP").toDouble());
                tracking->setHeadSensorOrientR(childElement.attribute("headSensorOrientR").toDouble());

                tracking->setHandSensorOffsetX(childElement.attribute("handSensorOffsetX").toDouble());
                tracking->setHandSensorOffsetY(childElement.attribute("handSensorOffsetY").toDouble());
                tracking->setHandSensorOffsetZ(childElement.attribute("handSensorOffsetZ").toDouble());
                tracking->setHandSensorOrientH(childElement.attribute("handSensorOrientH").toDouble());
                tracking->setHandSensorOrientP(childElement.attribute("handSensorOrientP").toDouble());
                tracking->setHandSensorOrientR(childElement.attribute("handSensorOrientR").toDouble());

                tracking->setTrackerType(TrackerType(childElement.attribute("trackerType").toInt()));
                tracking->setDirections(DirectionType(childElement.attribute("xDir").toInt()),
                                        DirectionType(childElement.attribute("yDir").toInt()),
                                        DirectionType(childElement.attribute("zDir").toInt()));
                tracking->setLinearMagneticFieldCorrection(childElement.attribute("fieldCorrectionX").toDouble(),
                                                           childElement.attribute("fieldCorrectionY").toDouble(),
                                                           childElement.attribute("fieldCorrectionZ").toDouble());
                tracking->setInterpolationFile(childElement.attribute("interpolationFile"));
                tracking->setDebugTracking(DebugTrackingType(childElement.attribute("debugTracking").toInt()));
                tracking->setDebugButtons(childElement.attribute("debugButtons").toInt());
                tracking->setDebugStation(childElement.attribute("debugStation").toInt());
            }
            else
                cout << "child is none. " << endl;
        }
        else
        {
            //cout<<"child is NOT element. "<<endl;
        }
        traverseNode(child);
        child = child.nextSibling();
    }
}
