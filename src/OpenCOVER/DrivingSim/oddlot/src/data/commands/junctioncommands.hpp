/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   21.05.2010
**
**************************************************************************/

#ifndef JUNCTIONCOMMANDS_HPP
#define JUNCTIONCOMMANDS_HPP

// 1000

#include "datacommand.hpp"

#include <QMap>

class RSystemElementJunction;
class RSystemElementRoad;
class RoadSystem;
class JunctionConnection;
class RoadLink;

//#########################//
// RemoveJunctionCommand //
//#########################//

class RemoveJunctionCommand : public DataCommand
{
public:
    explicit RemoveJunctionCommand(RSystemElementJunction *junction, DataCommand *parent = NULL);
    virtual ~RemoveJunctionCommand();

    virtual int id() const
    {
        return 0x1011;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveJunctionCommand(); /* not allowed */
    RemoveJunctionCommand(const RemoveJunctionCommand &); /* not allowed */
    RemoveJunctionCommand &operator=(const RemoveJunctionCommand &); /* not allowed */

private:
    RSystemElementJunction *junction_;
    RoadSystem *roadSystem_;

    // JunctionConnections //
    //
    QMultiMap<QString, JunctionConnection *> connections_;

    // Predecessors, Successors //
    //
    QMap<RSystemElementRoad *, RoadLink *> predecessors_;
    QMap<RSystemElementRoad *, RoadLink *> successors_;
};

//#########################//
// NewJunctionCommand //
//#########################//

class NewJunctionCommand : public DataCommand
{
public:
    explicit NewJunctionCommand(RSystemElementJunction *newJunction, RoadSystem *roadSystem, DataCommand *parent = NULL);
    virtual ~NewJunctionCommand();

    virtual int id() const
    {
        return 0x1004;
    }

    virtual void undo();
    virtual void redo();

private:
    NewJunctionCommand(); /* not allowed */
    NewJunctionCommand(const NewJunctionCommand &); /* not allowed */
    NewJunctionCommand &operator=(const NewJunctionCommand &); /* not allowed */

private:
    RSystemElementJunction *newJunction_;
    RoadSystem *roadSystem_;
};

//#########################//
// AddConnectionCommand //
//#########################//

class AddConnectionCommand : public DataCommand
{
public:
    explicit AddConnectionCommand(RSystemElementJunction *junction, JunctionConnection *connection, DataCommand *parent = NULL);
    virtual ~AddConnectionCommand();

    virtual int id() const
    {
        return 0x1004;
    }

    virtual void undo();
    virtual void redo();

private:
    AddConnectionCommand(); /* not allowed */
    AddConnectionCommand(const AddConnectionCommand &); /* not allowed */
    AddConnectionCommand &operator=(const AddConnectionCommand &); /* not allowed */

private:
    RSystemElementJunction *junction_;
    JunctionConnection *connection_;
};

//##############################//
// SetConnectionLaneLinkCommand //
//##############################//

class SetConnectionLaneLinkCommand : public DataCommand
{
public:
    explicit SetConnectionLaneLinkCommand(JunctionConnection *connection, int from, int to, DataCommand *parent = NULL);
    virtual ~SetConnectionLaneLinkCommand();

    virtual int id() const
    {
        return 0x1004;
    }

    virtual void undo();
    virtual void redo();

private:
    SetConnectionLaneLinkCommand(); /* not allowed */
    SetConnectionLaneLinkCommand(const SetConnectionLaneLinkCommand &); /* not allowed */
    SetConnectionLaneLinkCommand &operator=(const SetConnectionLaneLinkCommand &); /* not allowed */

private:
    JunctionConnection *connection_;
    int oldFrom_;
    int newFrom_;
    int to_;
};

//##############################//
// SetConnectionLaneLinkCommand //
//##############################//

class RemoveConnectionLaneLinksCommand : public DataCommand
{
public:
    explicit RemoveConnectionLaneLinksCommand(JunctionConnection *connection, DataCommand *parent = NULL);
    virtual ~RemoveConnectionLaneLinksCommand();

    virtual int id() const
    {
        return 0x1004;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveConnectionLaneLinksCommand(); /* not allowed */
    RemoveConnectionLaneLinksCommand(const RemoveConnectionLaneLinksCommand &); /* not allowed */
    RemoveConnectionLaneLinksCommand &operator=(const RemoveConnectionLaneLinksCommand &); /* not allowed */

private:
    JunctionConnection *connection_;
    QMap<int, int> oldLaneLinks_;
};


#endif // JUNCTIONCOMMANDS_HPP
