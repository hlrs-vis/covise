/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   12.05.2010
**
**************************************************************************/

#include "prototypemanager.hpp"

// Data //
//
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// IO //
//
#include "src/io/domparser.hpp"

// Qt //
//
#include <QFile>
#include <QMessageBox>
#include <QApplication>

#include "math.h"

//####################//
//                    //
// PROTOTYPEMANAGER   //
//                    //
//####################//

PrototypeManager::PrototypeManager(QObject *parent)
    : QObject(parent)
{
}

PrototypeManager::~PrototypeManager()
{
    foreach (PrototypeContainer<RSystemElementRoad *> *prototype, roadPrototypes_)
    {
        delete prototype;
    }
    foreach (PrototypeContainer<RoadSystem *> *prototype, roadSystemPrototypes_)
    {
        delete prototype;
    }
}

void
PrototypeManager::addRoadPrototype(const QString &name, const QIcon &icon, RSystemElementRoad *road, PrototypeManager::PrototypeType type,QString &system,QString &typeName,QString &lanes)
{
    roadPrototypes_.insert(type, new PrototypeContainer<RSystemElementRoad *>(name, icon, road,system,typeName,lanes));
}

void
PrototypeManager::addRoadSystemPrototype(const QString &name, const QIcon &icon, RoadSystem *roadSystem)
{
    roadSystemPrototypes_.append(new PrototypeContainer<RoadSystem *>(name, icon, roadSystem));
}

bool
PrototypeManager::loadPrototypes(const QString &fileName)
{
    // Print //
    //
    qDebug("Loading file: " + fileName.toUtf8());

    // Open file //
    //
    QFile file(fileName);
    if (!file.open(QFile::ReadOnly | QFile::Text))
    {
        //		QMessageBox::warning(this, tr("ODD"), tr("Cannot read file %1:\n%2.")
        //		.arg(fileName)
        //		.arg(file.errorString()));
        qDebug("Loading file failed: " + fileName.toUtf8());
        return false;
    }

    // Parse file //
    //
    QApplication::setOverrideCursor(Qt::WaitCursor);
    DomParser *parser = new DomParser(NULL);
    parser->parsePrototypes(&file); // parser calls addPrototype()
    delete parser;

    // Close file //
    //
    QApplication::restoreOverrideCursor();
    file.close();
    return true;
}

/// find prototypes of a certain type with an appropriate typeName, otherwise return the first one of that type as a fallback.

RSystemElementRoad *PrototypeManager::getRoadPrototype(PrototypeManager::PrototypeType type,QString combinedType)
{
    RSystemElementRoad *rp=NULL;
    QString systemName="osm";
    QString typeName="unknown";
    QString laneNumbers="1";
    QStringList sl = combinedType.split(':');
    if(sl.size()>0)
        systemName = sl[0];
    if(sl.size()>1)
        typeName = sl[1];
    if(sl.size()>2)
        laneNumbers = sl[2];
    QList<PrototypeContainer<RSystemElementRoad *> *> values = roadPrototypes_.values(type);
    for (int i = 0; i < values.size(); ++i)
    {
        PrototypeContainer<RSystemElementRoad *> *ptc = values.at(i);
        if(ptc->getSystemName() == systemName && ptc->getTypeName() == typeName)
        {
            rp = ptc->getPrototype();
            if( ptc->getLaneNumbers() == laneNumbers)
                break;
        }
    }
    if(rp == NULL) // if we did not find the type take one with the same number of lanes
    {
        for (int i = 0; i < values.size(); ++i)
        {
            PrototypeContainer<RSystemElementRoad *> *ptc = values.at(i);
            if(ptc->getSystemName() == systemName && ptc->getLaneNumbers() == laneNumbers )
            {
                rp = ptc->getPrototype();
                break;
            }
        }
    }
    if(rp == NULL)
        rp = values.first()->getPrototype();
    return rp;
}