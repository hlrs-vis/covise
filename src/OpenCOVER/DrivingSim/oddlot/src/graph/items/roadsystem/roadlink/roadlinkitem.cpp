/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/18/2010
**
**************************************************************************/

#include "roadlinkitem.hpp"

// Data //
//
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/rsystemelementjunction.hpp"
#include "src/data/roadsystem/junctionconnection.hpp"
#include "src/data/roadsystem/roadlink.hpp"

// Items //
//
#include "roadlinkroaditem.hpp"
#include "roadlinkhandle.hpp"

// Utils //
//
#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>

#define DISTANCE 2.0

RoadLinkItem::RoadLinkItem(RoadLinkRoadItem *parent, RoadLink *roadLink)
    : GraphElement(parent, roadLink)
    , roadLink_(roadLink)
    , type_(roadLink->getRoadLinkType())
    , parentRoadItem_(parent)
    , parentRoad_(roadLink->getParentRoad())
    , targetRoad_(NULL)
    , targetJunction_(NULL)
    , roadLinkHandle_(NULL)
{
    init();
}

/*! No RoadLink: Observe road instead.
*
*/
RoadLinkItem::RoadLinkItem(RoadLinkRoadItem *parent, RoadLink::RoadLinkType roadLinkType)
    : GraphElement(parent, parent->getRoad())
    , roadLink_(NULL)
    , type_(roadLinkType)
    , parentRoadItem_(parent)
    , parentRoad_(parent->getRoad())
    , targetRoad_(NULL)
    , targetJunction_(NULL)
    , roadLinkHandle_(NULL)
{
    init();
}

RoadLinkItem::~RoadLinkItem()
{
    // Observer //
    //
    if (targetRoad_)
    {
        targetRoad_->detachObserver(this);
    }

    if (targetJunction_)
    {
        targetJunction_->detachObserver(this);

        foreach (RSystemElementRoad *path, paths_)
        {
            path->detachObserver(this);
        }
    }
}

void
RoadLinkItem::init()
{
    // Observer //
    //
    if (roadLink_ && roadLink_->getElementType() == "road")
    {
        targetRoad_ = roadLink_->getProjectData()->getRoadSystem()->getRoad(roadLink_->getElementId());
        if (!targetRoad_)
        {
            qDebug("%s", tr("Error 1011191315! RoadLinkItem::init() road not found: ").append(roadLink_->getElementId()).toUtf8().constData());
        }
        else
        {
            targetRoad_->attachObserver(this);
        }
    }
    else if (roadLink_ && roadLink_->getElementType() == "junction")
    {
        targetJunction_ = roadLink_->getProjectData()->getRoadSystem()->getJunction(roadLink_->getElementId());
        if (!targetJunction_)
        {
            qDebug("%s", tr("Error 1011191316! RoadLinkItem::init() junction not found: ").append(roadLink_->getElementId()).toUtf8().constData());
        }
        else
        {
            targetJunction_->attachObserver(this);

            foreach (JunctionConnection *connection, targetJunction_->getConnections(parentRoad_->getID()))
            {
                RSystemElementRoad *path = connection->getProjectData()->getRoadSystem()->getRoad(connection->getConnectingRoad());
                if (!path)
                {
                    qDebug("%s", tr("Error 1011191314! RoadLinkItem::init() road not found: ").append(connection->getConnectingRoad()).toUtf8().constData());
                }
                else
                {
                    path->attachObserver(this);
                    paths_.append(path);
                }
            }
        }
    }

    // Handle //
    //
    roadLinkHandle_ = new RoadLinkHandle(this);

    // Color and Path //
    //
    QPen thePen;
    thePen.setWidth(2);
    thePen.setCosmetic(true);
    setPen(thePen);

    updateColor();
    createPath();
}

RoadLink::RoadLinkType
RoadLinkItem::getRoadLinkType() const
{
    return type_;
}

void
RoadLinkItem::updateType()
{
    if (roadLink_)
    {
        qDebug("%s", tr("Error 1011261638! RoadLinkItem::updateType() not implemented yet.").toUtf8().constData());
        //		type_ = roadLink_->getRoadLinkType();
        //		updateTransform();
        //		updateColor();
    }
}

void
RoadLinkItem::updateParentRoad()
{
    if (roadLink_)
    {
        qDebug("%s", tr("Error 1011261639! RoadLinkItem::updateParentRoad() not implemented yet.").toUtf8().constData());
        //		parentRoad_ = roadLink_->getParentRoad();
        //		updateTransform();
        //		updateColor();
    }
}

void
RoadLinkItem::updatePathList()
{
    if (roadLink_ && roadLink_->getElementType() == "junction")
    {
        QList<RSystemElementRoad *> newPaths;

        foreach (JunctionConnection *connection, targetJunction_->getConnections(parentRoad_->getID()))
        {
            RSystemElementRoad *path = connection->getProjectData()->getRoadSystem()->getRoad(connection->getConnectingRoad());
            if (!path)
            {
                qDebug("%s", tr("Error 1011261640! RoadLinkItem::updatePathList() road not found: ").append(connection->getConnectingRoad()).toUtf8().constData());
            }
            else
            {
                newPaths.append(path);
                if (!paths_.contains(path))
                {
                    path->attachObserver(this); // newly added path
                }
            }
        }

        foreach (RSystemElementRoad *path, paths_)
        {
            if (!newPaths.contains(path))
            {
                path->detachObserver(this); // newly removed path
            }
        }

        paths_ = newPaths;
    }

    // Redraw //
    //
    createPath();
}

void
RoadLinkItem::updateTransform() // called by parent road item
{
    if (type_ == RoadLink::DRL_PREDECESSOR)
    {
        roadLinkHandle_->setPos(parentRoad_->getGlobalPoint(DISTANCE, parentRoad_->getMaxWidth(0.0) + DISTANCE));
        //		roadLinkHandle_->setPos(parentRoad_->getGlobalPoint(DISTANCE, parentRoad_->getMaxWidth(0.0)*0.5));
        roadLinkHandle_->setRotation(parentRoad_->getGlobalHeading(0.0) + 180.0);
        createPath();
    }
    else if (type_ == RoadLink::DRL_SUCCESSOR)
    {
        roadLinkHandle_->setPos(parentRoad_->getGlobalPoint(parentRoad_->getLength() - DISTANCE, parentRoad_->getMinWidth(parentRoad_->getLength() - DISTANCE) - DISTANCE));
        //		roadLinkHandle_->setPos(parentRoad_->getGlobalPoint(parentRoad_->getLength()-DISTANCE, parentRoad_->getMinWidth(parentRoad_->getLength()-DISTANCE)*0.5));
        roadLinkHandle_->setRotation(parentRoad_->getGlobalHeading(parentRoad_->getLength()));
        createPath();
    }
}

//################//
// GRAPHICS       //
//################//

void
RoadLinkItem::updateColor()
{
    QPen thePen = pen();
    if (roadLink_)
    {
        if (roadLink_->isLinkValid())
        {
            // TODO: Check if road pos == target(s) pos

            roadLinkHandle_->setBrush(QBrush(ODD::instance()->colors()->brightGreen()));
            roadLinkHandle_->setPen(QPen(ODD::instance()->colors()->darkGreen()));
            thePen.setColor(ODD::instance()->colors()->darkGreen());
        }
        else
        {
            roadLinkHandle_->setBrush(QBrush(ODD::instance()->colors()->brightRed()));
            roadLinkHandle_->setPen(QPen(ODD::instance()->colors()->darkRed()));
            thePen.setColor(ODD::instance()->colors()->darkRed());
        }
    }
    else
    {
        roadLinkHandle_->setBrush(QBrush(ODD::instance()->colors()->brightOrange()));
        roadLinkHandle_->setPen(QPen(ODD::instance()->colors()->darkOrange()));
        thePen.setColor(ODD::instance()->colors()->darkOrange());
    }
    setPen(thePen);
}

void
RoadLinkItem::createPath()
{
    QPainterPath thePath;

    if (roadLink_)
    {
        QPointF handlePoint;
        if (type_ == RoadLink::DRL_PREDECESSOR)
        {
            handlePoint = parentRoad_->getGlobalPoint(DISTANCE, parentRoad_->getMaxWidth(DISTANCE) + DISTANCE);
        }
        else if (type_ == RoadLink::DRL_SUCCESSOR)
        {
            handlePoint = parentRoad_->getGlobalPoint(parentRoad_->getLength() - DISTANCE, parentRoad_->getMinWidth(parentRoad_->getLength() - DISTANCE) - DISTANCE);
        }
        else
        {
            qDebug("Error 1011191317! RoadLinkItem::createPath() unkown link type.");
            setPath(thePath);
            return;
        }

        // Road //
        //
        if (targetRoad_)
        {
            thePath.moveTo(handlePoint);

            if (roadLink_->getContactPoint() == "start")
            {
                thePath.lineTo(targetRoad_->getGlobalPoint(DISTANCE, targetRoad_->getMinWidth(DISTANCE) - DISTANCE));
            }
            else if (roadLink_->getContactPoint() == "end")
            {
                thePath.lineTo(targetRoad_->getGlobalPoint(targetRoad_->getLength() - DISTANCE, targetRoad_->getMaxWidth(targetRoad_->getLength() - DISTANCE) + DISTANCE));
            }
            else
            {
                qDebug("Error 1011191318! RoadLinkItem::createPath() unkown contact point.");
                setPath(thePath);
                return;
            }
        }

        // Junction //
        //
        else if (targetJunction_)
        {
            foreach (JunctionConnection *connection, targetJunction_->getConnections(parentRoad_->getID()))
            {
                RSystemElementRoad *targetRoad = connection->getProjectData()->getRoadSystem()->getRoad(connection->getConnectingRoad());
                if (!targetRoad)
                {
                    qDebug("%s", tr("Error 1011191310! RoadLinkItem::createPath() road not found: ").append(connection->getConnectingRoad()).toUtf8().constData());
                    continue;
                }

                thePath.moveTo(handlePoint);

                if (connection->getContactPoint() == "start")
                {
                    thePath.lineTo(targetRoad->getGlobalPoint(DISTANCE, targetRoad->getMinWidth(DISTANCE) - DISTANCE));
                }
                else if (connection->getContactPoint() == "end")
                {
                    thePath.lineTo(targetRoad->getGlobalPoint(targetRoad->getLength() - DISTANCE, targetRoad->getMaxWidth(targetRoad->getLength() - DISTANCE) + DISTANCE));
                }
                else
                {
                    qDebug("Error 1011191319! RoadLinkItem::createPath() unkown contact point.");
                    setPath(thePath);
                    return;
                }
            }
        }
    }

    setPath(thePath);
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
*/
void
RoadLinkItem::updateObserver()
{
    // Parent //
    //
    GraphElement::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    if (roadLink_)
    {
        // RoadLink //
        //
        int changes = roadLink_->getRoadLinkChanges();
        if (changes & RoadLink::CRL_TypeChanged)
        {
            updateType();
        }

        if (changes & RoadLink::CRL_ParentRoadChanged)
        {
            updateParentRoad();
        }

        if ((changes & RoadLink::CRL_ContactPointChanged)
            || (changes & RoadLink::CRL_IdChanged))
        {
            //			createPath();
            //			updateColor();
        }
    }
    else
    {
        // Road (but no link yet) //
        //
        int changes = parentRoad_->getRoadChanges();
        if ((changes & RSystemElementRoad::CRD_PredecessorChange)
            && (type_ == RoadLink::DRL_PREDECESSOR))
        {
            registerForDeletion(); // life is over!
            return;
        }

        if ((changes & RSystemElementRoad::CRD_SuccessorChange)
            && (type_ == RoadLink::DRL_SUCCESSOR))
        {
            registerForDeletion(); // time to say goodbye
            return;
        }
    }

    // Target road //
    //
    if (targetRoad_)
    {
        int changes = targetRoad_->getRoadChanges();
        if ((changes & RSystemElementRoad::CRD_LengthChange)
            || (changes & RSystemElementRoad::CRD_ShapeChange)
            || (changes & RSystemElementRoad::CRD_TrackSectionChange))
        {
            createPath();
        }
    }

    // Target junction //
    //
    bool doUpdatePathList = false;
    if (targetJunction_)
    {
        int changes = targetJunction_->getJunctionChanges();
        if ((changes & RSystemElementJunction::CJN_ConnectionChanged))
        {
            doUpdatePathList = true;
        }
        else
        {
            foreach (RSystemElementRoad *path, paths_)
            {
                int changes = path->getRoadChanges();
                if ((changes & RSystemElementRoad::CRD_LengthChange)
                    || (changes & RSystemElementRoad::CRD_ShapeChange)
                    || (changes & RSystemElementRoad::CRD_TrackSectionChange))
                {
                    doUpdatePathList = true;
                    break;
                }
            }
        }
    }
    if (doUpdatePathList)
    {
        updatePathList();
    }
}
