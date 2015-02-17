/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   08.07.2010
**
**************************************************************************/

#ifndef ROADSYSTEMCOMMANDS_HPP
#define ROADSYSTEMCOMMANDS_HPP

// 2000

#include "datacommand.hpp"

#include <QList>
#include <QTransform>
#include <QPointF>
#include <QMultiMap>

class RoadSystem;

class RSystemElement;
class RSystemElementRoad;
class RSystemElementController;
class RSystemElementJunction;
class RSystemElementFiddleyard;

//#########################//
// AddRoadSystemPrototypeCommand //
//#########################//

class AddRoadSystemPrototypeCommand : public DataCommand
{
public:
    explicit AddRoadSystemPrototypeCommand(RoadSystem *roadSystem, const RoadSystem *prototypeRoadSystem, const QPointF &deltaPos, double deltaHeadingDegrees, DataCommand *parent = NULL);
    virtual ~AddRoadSystemPrototypeCommand();

    virtual int id() const
    {
        return 0x2001;
    }

    virtual void undo();
    virtual void redo();

private:
    AddRoadSystemPrototypeCommand(); /* not allowed */
    AddRoadSystemPrototypeCommand(const AddRoadSystemPrototypeCommand &); /* not allowed */
    AddRoadSystemPrototypeCommand &operator=(const AddRoadSystemPrototypeCommand &); /* not allowed */

private:
    // RoadSystem //
    //
    RoadSystem *roadSystem_;

    // RoadSystemElements //
    //
    QList<RSystemElementRoad *> newRoads_;
    QList<RSystemElementController *> newControllers_;
    QList<RSystemElementJunction *> newJunctions_;
    QList<RSystemElementFiddleyard *> newFiddleyards_;
};

//#########################//
// SetRSystemElementIdCommand //
//#########################//

class SetRSystemElementIdCommand : public DataCommand
{
public:
    explicit SetRSystemElementIdCommand(RoadSystem *roadSystem, RSystemElement *element, const QString &Id, const QString &name, DataCommand *parent = NULL);
    virtual ~SetRSystemElementIdCommand();

    virtual int id() const
    {
        return 0x2001;
    }

    virtual void undo();
    virtual void redo();

private:
    SetRSystemElementIdCommand(); /* not allowed */
    SetRSystemElementIdCommand(const SetRSystemElementIdCommand &); /* not allowed */
    SetRSystemElementIdCommand &operator=(const SetRSystemElementIdCommand &); /* not allowed */

private:
    // RSystemElement //
    //
    RSystemElement *element_;

    RoadSystem *roadSystem_;

    QString oldId_;
    QString newId_;
    QString oldName_;
    QString newName_;
};

//#########################//
// AddToJunctionCommand //
//#########################//

class AddToJunctionCommand : public DataCommand
{
public:
    explicit AddToJunctionCommand(RoadSystem *roadSystem, RSystemElementRoad *road, RSystemElementJunction *junction, DataCommand *parent = NULL);
    virtual ~AddToJunctionCommand();

    virtual int id() const
    {
        return 0x2002;
    }

    virtual void undo();
    virtual void redo();

private:
    AddToJunctionCommand(); /* not allowed */
    AddToJunctionCommand(const AddToJunctionCommand &); /* not allowed */
    AddToJunctionCommand &operator=(const AddToJunctionCommand &); /* not allowed */

private:
    // RSystemElement //
    //
    RoadSystem *roadSystem_;
    RSystemElementRoad *road_;
    RSystemElementJunction *junction_;
    QString oldJunctionID_;
};

//#########################//
// RemoveFromJunctionCommand //
//#########################//

class RemoveFromJunctionCommand : public DataCommand
{
public:
    explicit RemoveFromJunctionCommand(RSystemElementJunction *junction, RSystemElementRoad *road, DataCommand *parent = NULL);
    virtual ~RemoveFromJunctionCommand();

    virtual int id() const
    {
        return 0x2004;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveFromJunctionCommand(); /* not allowed */
    RemoveFromJunctionCommand(const RemoveFromJunctionCommand &); /* not allowed */
    RemoveFromJunctionCommand &operator=(const RemoveFromJunctionCommand &); /* not allowed */

private:
    // RSystemElement //
    //
    RSystemElementRoad *road_;
    RSystemElementJunction *junction_, *oldJunction_;
};

#endif // ROADSYSTEMCOMMANDS_HPP
