/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Uwe Woessner (c) 2013
**   <woessner@hlrs.de.de>
**   03/2013
**
**************************************************************************/

#include "osmimport.hpp"
#include "importsettings.hpp"
#include "ui_osmimport.h"
#include <QDomDocument>
#include <QFile>

#include "projectwidget.hpp"
#include "src/gui/projectionsettings.hpp"
#include "src/gui/importsettings.hpp"
#include "src/graph/topviewgraph.hpp"

// Data //
#include "src/data/visitors/boundingboxvisitor.hpp"
#include "src/data/projectdata.hpp"
#include "src/data/commands/projectdatacommands.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/scenerysystem/scenerysystem.hpp"

// Settings //
//
#include "src/settings/projectsettings.hpp"
//################//
// CONSTRUCTOR    //
//################//

OsmImport::OsmImport()
    : ui(new Ui::OsmImport)
    , init_(false)
{
    ui->setupUi(this);

    nam = new QNetworkAccessManager(this);
    QObject::connect(nam, SIGNAL(finished(QNetworkReply *)),
                     this, SLOT(finishedSlot(QNetworkReply *)));
    ui->minLatSpin->setDecimals(8);
    ui->minLatSpin->setMaximum(360);
    ui->minLatSpin->setMinimum(-360);
    ui->minLatSpin->setValue(48.709145);
    //ui->minLatSpin->setValue(31.23122);
    ui->minLongSpin->setDecimals(8);
    ui->minLongSpin->setMaximum(360);
    ui->minLongSpin->setMinimum(-360);
    ui->minLongSpin->setValue(9.24237);
    //ui->minLongSpin->setValue(121.46355);
    ui->maxLatSpin->setDecimals(8);
    ui->maxLatSpin->setMaximum(360);
    ui->maxLatSpin->setMinimum(-360);
    ui->maxLatSpin->setValue(48.727294);
    //ui->maxLatSpin->setValue(31.236);
    ui->maxLongSpin->setMaximum(360);
    ui->maxLongSpin->setMinimum(-360);
    ui->maxLongSpin->setDecimals(8);
     ui->maxLongSpin->setValue(9.291937);
    //ui->maxLongSpin->setValue(121.476);

    connect(this, SIGNAL(accepted()), this, SLOT(okPressed()));
    // Done //
    //
    init_ = true;
}

OsmImport::~OsmImport()
{
    delete ui;
}

/** \brief imports a OSM file.
*
*/
bool
OsmImport::importOSMFile(const QString &fileName)
{
    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return false;
 
    QDomDocument doc;
    if (!doc.setContent(&file)) {
        file.close();
        return false;
    }
    file.close();
    return parseDoc(doc);
}

bool OsmImport::parseDoc(QDomDocument &doc)
{

    QDomNodeList list = doc.elementsByTagName("node");
    for (int i = 0; i < list.count(); i++)
    {
        nodes.append(osmNode(list.at(i).toElement()));
    }
    list = doc.elementsByTagName("way");
    for (int i = 0; i < list.count(); i++)
    {
        ways.append(osmWay(list.at(i).toElement(), nodes));
    }
    
    bool importPrimary();
    bool importSecondary();
    bool importTertiary();
    bool importService();
    bool importPath();
    bool importSteps();
    bool importTrack();
    bool importFootway();
    bool importResidential();
    bool importLiving_street();
    bool importCycleway();
    bool importTurning_circle();
    bool importPedestrian();
    bool importUnclassified();
    bool doPrimary = ImportSettings::instance()->importPrimary();
    bool doSecondary = ImportSettings::instance()->importSecondary();
    bool doTertiary = ImportSettings::instance()->importTertiary();
    bool doService = ImportSettings::instance()->importService();
    bool doPath = ImportSettings::instance()->importPath();
    bool doSteps = ImportSettings::instance()->importSteps();
    bool doTrack = ImportSettings::instance()->importTrack();
    bool doFootpath = ImportSettings::instance()->importFootway();
    bool doResidential = ImportSettings::instance()->importResidential();
    bool doLiving_street = ImportSettings::instance()->importLiving_street();
    bool doCycleway = ImportSettings::instance()->importCycleway();
    bool doTurning_circle = ImportSettings::instance()->importTurning_circle();
    bool doPedestrian = ImportSettings::instance()->importPedestrian();
    bool doUnclassified = ImportSettings::instance()->importUnclassified();
    for (int i = 0; i < ways.count(); i++)
    {
        osmWay::wayType t = ways.at(i).type;

        if ((t == osmWay::primary && doPrimary) ||
            (t == osmWay::secondary && doSecondary) ||
            (t == osmWay::tertiary && doTertiary) ||
            (t == osmWay::service && doService) ||
            (t == osmWay::path && doPath) ||
            (t == osmWay::steps && doSteps) ||
            (t == osmWay::track && doTrack) ||
            (t == osmWay::footway && doFootpath) ||
            (t == osmWay::residential && doResidential) ||
            (t == osmWay::living_street && doLiving_street) ||
            (t == osmWay::cycleway && doCycleway) ||
            (t == osmWay::turning_circle && doTurning_circle) ||
            (t == osmWay::pedestrian && doPedestrian) ||
            (t == osmWay::unclassified && doUnclassified))
        {
            project->XVector = ways.at(i).XVector;
            project->YVector = ways.at(i).YVector;
            project->ZVector = ways.at(i).ZVector;
            if (project->XVector.size() > 0)
            {
                project->addLineStrip(ways.at(i).name);
            }
        }
    }
    ways.clear();
    nodes.clear();
    // resize
    BoundingBoxVisitor *visitor = new BoundingBoxVisitor();
    project->getProjectData()->getRoadSystem()->accept(visitor);
    project->getProjectData()->getScenerySystem()->accept(visitor);
    QRectF box = visitor->getBoundingBox();
    SetProjectDimensionsCommand *command = new SetProjectDimensionsCommand(project->getProjectData(), box.bottom() + 0.1 * box.height(), box.top() - 0.1 * box.height(), box.right() + 0.1 * box.width(), box.left() - 0.1 * box.width());
    project->getProjectSettings()->executeCommand(command);

    return true;
}

void OsmImport::finishedSlot(QNetworkReply *reply)
{
    // Reading attributes of the reply
    // e.g. the HTTP status code
    QVariant statusCodeV = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute);
    // Or the target URL if it was a redirect:
    QVariant redirectionTargetUrl = reply->attribute(QNetworkRequest::RedirectionTargetAttribute);
    // see CS001432 on how to handle this
    project->numLineStrips = 0;
    // no error received?
    if (reply->error() == QNetworkReply::NoError)
    {
        // read data from QNetworkReply here
        QByteArray bytes = reply->readAll();

        QDomDocument doc;
        doc.setContent(bytes);
        parseDoc(doc);
    }
    else
    {
        // handle errors here
    }
    project->getTopviewGraph()->updateSceneSize();
    QApplication::restoreOverrideCursor();
    // We receive ownership of the reply object
    // and therefore need to handle deletion.
    reply->deleteLater();
}

osmNode::osmNode()
{
    latitude = 0;
    longitude = 0;
    id = 0;
}

osmNode::osmNode(const osmNode &n)
{
    latitude = n.latitude;
    longitude = n.longitude;
    id = n.id;
}
osmNode::osmNode(QDomElement element)
{
    id = element.attribute("id").toUInt();
    latitude = element.attribute("lat").toDouble();
    longitude = element.attribute("lon").toDouble();
}

void osmNode::getCoordinates(double &x, double &y, double &z) const
{
    y = latitude * DEG_TO_RAD;
    x = longitude * DEG_TO_RAD;
    z = 0.0;
    ProjectionSettings::instance()->transform(x, y, z);
}
osmWay::osmWay()
{
}

osmWay::osmWay(const osmWay &w)
{
    type = w.type;
    XVector = w.XVector;
    YVector = w.YVector;
    ZVector = w.ZVector;
    name = w.name;
}
osmWay::osmWay(QDomElement element, QVector<osmNode> &nodes)
{
    type = unknown;
    numLanes = 2;
    maxSpeed = -1;
    bridge=false;
    name = "id" + element.attribute("id");
    QDomNodeList list = element.elementsByTagName("nd");
    for (int i = 0; i < list.count(); i++)
    {
        QDomElement ele = list.at(i).toElement();
        unsigned int ref = ele.attribute("ref").toUInt();
        for (int n = 0; n < nodes.count(); n++)
        {
            if (nodes.at(n).id == ref)
            {
                double x, y, z;
                nodes.at(n).getCoordinates(x, y, z);
                XVector.push_back(x);
                YVector.push_back(y);
                ZVector.push_back(0.0);
                break;
            }
        }
    }

    list = element.elementsByTagName("tag");
    for (int i = 0; i < list.count(); i++)
    {
        QDomElement ele = list.at(i).toElement();
        QString k = ele.attribute("k");
        QString v = ele.attribute("v");
        if (k == "name")
        {
            name = v;
        }
        else if (k == "lanes")
        {
            numLanes = v.toInt();
        }
        else if (k == "maxspeed")
        {
            maxSpeed = v.toInt();
        }
        else if (k == "bridge")
        {
            if (v == "yes")
            {
                bridge = true;
            }
        }
        else if (k == "highway")
        {
            //type = highway;
            if (v == "residential")
            {
                type = residential;
            }
            else if (v == "footway")
            {
                type = footway;
            }
            else if (v == "track")
            {
                type = track;
            }
            else if (v == "steps")
            {
                type = steps;
            }
            else if (v == "primary" || v == "primary_link")
            {
                type = secondary;
            }
            else if (v == "secondary"|| v == "secondary_link")
            {
                type = secondary;
            }
            else if (v == "tertiary"|| v == "tertiary_link")
            {
                type = tertiary;
            }
            else if (v == "service"|| v == "service_link")
            {
                type = service;
            }
            else if (v == "path")
            {
                type = path;
            }
            else if (v == "living_street")
            {
                type = living_street;
            }
            else if (v == "cycleway")
            {
                type = cycleway;
            }
            else if (v == "turning_circle")
            {
                type = turning_circle;
            }
            else if (v == "pedestrian")
            {
                type = pedestrian;
            }
            else if (v == "unclassified")
            {
                type = unclassified;
            }
        }
        else if (k == "building")
        {
            type = building;
        }
    }
}

//################//
// FUNCTIONS      //
//################//

//################//
// SLOTS          //
//################//

void
OsmImport::on_preview_released()
{
    /*
	QString boxString;

	boxString = QString::number(ui->minLongSpin->value(),'f',10)+QString(",")+ QString::number(ui->minLatSpin->value(),'f',10)+QString(",")+ QString::number(ui->maxLongSpin->value(),'f',10)+QString(",")+ QString::number(ui->maxLatSpin->value(),'f',10);
   
	QUrl url("http://api.openstreetmap.org/api/0.6/map?bbox="+boxString);
    QNetworkReply* reply = nam->get(QNetworkRequest(url));
	*/
    // maybe we need to check the reply and compare it to the reply we get in the slot to check whether it is for the correct request.
}

void
OsmImport::okPressed()
{
    QString boxString;

    boxString = QString::number(ui->minLongSpin->value(), 'f', 10) + QString(",") + QString::number(ui->minLatSpin->value(), 'f', 10) + QString(",") + QString::number(ui->maxLongSpin->value(), 'f', 10) + QString(",") + QString::number(ui->maxLatSpin->value(), 'f', 10);

    //QUrl url("http://api.openstreetmap.org/api/0.6/map?bbox="+boxString);

    QUrl url("http://www.overpass-api.de/api/xapi?map?bbox=" + boxString);
    QNetworkReply *reply = nam->get(QNetworkRequest(url));
}
