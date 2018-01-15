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

#ifndef CONTROLLERCOMMANDS_HPP
#define CONTROLLERCOMMANDS_HPP

// 1000

#include "datacommand.hpp"

#include "src/data/roadsystem/rsystemelementcontroller.hpp"

#include <QMap>

class RoadSystem;
class Signal;


//#########################//
// AddControllerCommand //
//#########################//

class AddControllerCommand : public DataCommand
{
public:
    explicit AddControllerCommand(RSystemElementController *controller, RoadSystem *roadSystem, DataCommand *parent = NULL);
    virtual ~AddControllerCommand();

    virtual int id() const
    {
        return 0x1300;
    }

    virtual void undo();
    virtual void redo();

private:
    AddControllerCommand(); /* not allowed */
    AddControllerCommand(const AddControllerCommand &); /* not allowed */
    AddControllerCommand &operator=(const AddControllerCommand &); /* not allowed */

private:
    RSystemElementController *controller_;
    RoadSystem *roadSystem_;
};

//#########################//
// RemoveControllerCommand //
//#########################//

class RemoveControllerCommand : public DataCommand
{
public:
    explicit RemoveControllerCommand(RSystemElementController *controller, RoadSystem *roadSystem, DataCommand *parent = NULL);
    virtual ~RemoveControllerCommand();

    virtual int id() const
    {
        return 0x1320;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveControllerCommand(); /* not allowed */
    RemoveControllerCommand(const RemoveControllerCommand &); /* not allowed */
    RemoveControllerCommand &operator=(const RemoveControllerCommand &); /* not allowed */

private:
    RSystemElementController *controller_;
    RoadSystem *roadSystem_;
};

//#########################//
// SetControllerPropertiesCommand //
//#########################//

class SetControllerPropertiesCommand : public DataCommand
{
public:
    explicit SetControllerPropertiesCommand(RSystemElementController *controller, const QString &id, const QString &name, int sequence, const QString &script, double cycleTime, DataCommand *parent = NULL);
    explicit SetControllerPropertiesCommand(RSystemElementController *controller, const QString &id, const QString &name, int sequence, const RSystemElementController::ControllerUserData &controllerUserData, DataCommand *parent = NULL);
    virtual ~SetControllerPropertiesCommand();

    virtual int id() const
    {
        return 0x1340;
    }

    virtual void undo();
    virtual void redo();

private:
    SetControllerPropertiesCommand(); /* not allowed */
    SetControllerPropertiesCommand(const SetControllerPropertiesCommand &); /* not allowed */
    SetControllerPropertiesCommand &operator=(const SetControllerPropertiesCommand &); /* not allowed */

private:
    RSystemElementController *controller_;
    QString newId_;
    QString newName_;
    int newSequence_;
    RSystemElementController::ControllerUserData newControllerUserData_;
    QList<ControlEntry *> newControlEntries_;

    QString oldId_;
    QString oldName_;
    int oldSequence_;
    RSystemElementController::ControllerUserData oldControllerUserData_;   
    QList<ControlEntry *> oldControlEntries_;
};

//#########################//
// AddControlEntryCommand //
//#########################//

class AddControlEntryCommand : public DataCommand
{
public:
    explicit AddControlEntryCommand(RSystemElementController *controller, ControlEntry *controlEntry, Signal *signal, DataCommand *parent = NULL);
    virtual ~AddControlEntryCommand();

    virtual int id() const
    {
        return 0x1360;
    }

    virtual void undo();
    virtual void redo();

private:
    AddControlEntryCommand(); /* not allowed */
    AddControlEntryCommand(const AddControlEntryCommand &); /* not allowed */
    AddControlEntryCommand &operator=(const AddControlEntryCommand &); /* not allowed */

private:
    RSystemElementController *controller_;
    ControlEntry *controlEntry_;
    Signal *signal_;
};

//#########################//
// DelControlEntryCommand //
//#########################//

class DelControlEntryCommand : public DataCommand
{
public:
    explicit DelControlEntryCommand(RSystemElementController *controller, ControlEntry *controlEntry, Signal *signal, DataCommand *parent = NULL);
    virtual ~DelControlEntryCommand();

    virtual int id() const
    {
        return 0x1380;
    }

    virtual void undo();
    virtual void redo();

private:
    DelControlEntryCommand(); /* not allowed */
    DelControlEntryCommand(const DelControlEntryCommand &); /* not allowed */
    DelControlEntryCommand &operator=(const DelControlEntryCommand &); /* not allowed */

private:
    RSystemElementController *controller_;
    ControlEntry *controlEntry_;
    Signal *signal_;
};



#endif // CONTROLLERCOMMANDS_HPP
