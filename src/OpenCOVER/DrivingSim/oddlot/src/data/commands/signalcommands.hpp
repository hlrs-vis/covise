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

#ifndef SIGNALCOMMANDS_HPP
#define SIGNALCOMMANDS_HPP

// 1000

#include "datacommand.hpp"

#include "src/data/roadsystem/sections/signalobject.hpp"
#include "src/data/roadsystem/sections/objectobject.hpp"
#include "src/data/roadsystem/sections/bridgeobject.hpp"

#include <QMap>

class RoadSystem;
class RSystemElementRoad;
class MoveRoadSectionCommand;

//#########################//
// AddSignalCommand //
//#########################//

class AddSignalCommand : public DataCommand
{
public:
    explicit AddSignalCommand(Signal *signal, RSystemElementRoad *road, DataCommand *parent = NULL);
    virtual ~AddSignalCommand();

    virtual int id() const
    {
        return 0x1011;
    }

    virtual void undo();
    virtual void redo();

private:
    AddSignalCommand(); /* not allowed */
    AddSignalCommand(const AddSignalCommand &); /* not allowed */
    AddSignalCommand &operator=(const AddSignalCommand &); /* not allowed */

private:
    Signal *signal_;
    RSystemElementRoad *road_;
};

//#########################//
// RemoveSignalCommand //
//#########################//

class RemoveSignalCommand : public DataCommand
{
public:
    explicit RemoveSignalCommand(Signal *signal, RSystemElementRoad *road, DataCommand *parent = NULL);
    virtual ~RemoveSignalCommand();

    virtual int id() const
    {
        return 0x1011;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveSignalCommand(); /* not allowed */
    RemoveSignalCommand(const RemoveSignalCommand &); /* not allowed */
    RemoveSignalCommand &operator=(const RemoveSignalCommand &); /* not allowed */

private:
    Signal *signal_;
    RSystemElementRoad *road_;
};

//#########################//
// SetSignalPropertiesCommand //
//#########################//

class SetSignalPropertiesCommand : public DataCommand
{
public:
    explicit SetSignalPropertiesCommand(Signal *signal, const QString &id, const QString &name, double t, bool dynamic, Signal::OrientationType orientation, double zOffset, const QString &country, int type, const QString &typeSubclass, int subtype, double value, bool pole, int size, int fromLane, int toLane, double probability = 0.0, double resetTime = 0.0, DataCommand *parent = NULL);
    explicit SetSignalPropertiesCommand(Signal *signal, const QString &id, const QString &name, Signal::SignalProperties &signalProps, Signal::Validity &validLanes, Signal::SignalUserData &userData, DataCommand *parent = NULL);
    virtual ~SetSignalPropertiesCommand();

    virtual int id() const
    {
        return 0x1011;
    }

    virtual void undo();
    virtual void redo();

private:
    SetSignalPropertiesCommand(); /* not allowed */
    SetSignalPropertiesCommand(const SetSignalPropertiesCommand &); /* not allowed */
    SetSignalPropertiesCommand &operator=(const SetSignalPropertiesCommand &); /* not allowed */

private:
    Signal *signal_;
    RSystemElementRoad *road_;
    QString newId_;
    QString newName_;
    QString oldId_;
    QString oldName_;
    Signal::SignalProperties newSignalProps_;
    Signal::SignalProperties oldSignalProps_;
    Signal::Validity newValidity_;
    Signal::Validity oldValidity_;
    Signal::SignalUserData newUserData_;
    Signal::SignalUserData oldUserData_;
};

//#########################//
// AddObjectCommand //
//#########################//

class AddObjectCommand : public DataCommand
{
public:
    explicit AddObjectCommand(Object *object, RSystemElementRoad *road, DataCommand *parent = NULL);
    virtual ~AddObjectCommand();

    virtual int id() const
    {
        return 0x1011;
    }

    virtual void undo();
    virtual void redo();

private:
    AddObjectCommand(); /* not allowed */
    AddObjectCommand(const AddObjectCommand &); /* not allowed */
    AddObjectCommand &operator=(const AddObjectCommand &); /* not allowed */

private:
    Object *object_;
    RSystemElementRoad *road_;
};

//#########################//
// RemoveObjectCommand //
//#########################//

class RemoveObjectCommand : public DataCommand
{
public:
    explicit RemoveObjectCommand(Object *object, RSystemElementRoad *road, DataCommand *parent = NULL);
    virtual ~RemoveObjectCommand();

    virtual int id() const
    {
        return 0x1011;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveObjectCommand(); /* not allowed */
    RemoveObjectCommand(const RemoveObjectCommand &); /* not allowed */
    RemoveObjectCommand &operator=(const RemoveObjectCommand &); /* not allowed */

private:
    Object *object_;
    RSystemElementRoad *road_;
};

//#########################//
// SetObjectPropertiesCommand //
//#########################//

class SetObjectPropertiesCommand : public DataCommand
{
public:
    explicit SetObjectPropertiesCommand(Object *object, const QString &id, const QString &name, const QString &type, double t, double zOffset, double validLength, Object::ObjectOrientation orientation, double length, double width, double radius, double height, double hdg, double pitch, double roll, bool pole, double repeatS, double repeatLength, double repeatDistance, const QString &textureFile, DataCommand *parent = NULL);
    explicit SetObjectPropertiesCommand(Object *object, const QString &id, const QString &name, Object::ObjectProperties &objectProps, Object::ObjectRepeatRecord &objectRepeat, const QString &textureFile, DataCommand *parent = NULL);
    virtual ~SetObjectPropertiesCommand();

    virtual int id() const
    {
        return 0x1011;
    }

    virtual void undo();
    virtual void redo();

private:
    SetObjectPropertiesCommand(); /* not allowed */
    SetObjectPropertiesCommand(const SetObjectPropertiesCommand &); /* not allowed */
    SetObjectPropertiesCommand &operator=(const SetObjectPropertiesCommand &); /* not allowed */

private:
    Object *object_;
    QString newId_;
    QString newName_;
    QString newTextureFile_;
    QString oldId_;
    QString oldName_;
    QString oldTextureFile_;
    Object::ObjectProperties newObjectProps_;
    Object::ObjectProperties oldObjectProps_;
    Object::ObjectRepeatRecord newObjectRepeat_;
    Object::ObjectRepeatRecord oldObjectRepeat_;
};

//#########################//
// AddBridgeCommand //
//#########################//

class AddBridgeCommand : public DataCommand
{
public:
    explicit AddBridgeCommand(Bridge *bridge, RSystemElementRoad *road, DataCommand *parent = NULL);
    virtual ~AddBridgeCommand();

    virtual int id() const
    {
        return 0x1011;
    }

    virtual void undo();
    virtual void redo();

private:
    AddBridgeCommand(); /* not allowed */
    AddBridgeCommand(const AddBridgeCommand &); /* not allowed */
    AddBridgeCommand &operator=(const AddBridgeCommand &); /* not allowed */

private:
    Bridge *bridge_;
    RSystemElementRoad *road_;
};

//#########################//
// RemoveBridgeCommand //
//#########################//

class RemoveBridgeCommand : public DataCommand
{
public:
    explicit RemoveBridgeCommand(Bridge *bridge, RSystemElementRoad *road, DataCommand *parent = NULL);
    virtual ~RemoveBridgeCommand();

    virtual int id() const
    {
        return 0x1011;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveBridgeCommand(); /* not allowed */
    RemoveBridgeCommand(const RemoveBridgeCommand &); /* not allowed */
    RemoveBridgeCommand &operator=(const RemoveBridgeCommand &); /* not allowed */

private:
    Bridge *bridge_;
    RSystemElementRoad *road_;
};

//#########################//
// SetBridgePropertiesCommand //
//#########################//

class SetBridgePropertiesCommand : public DataCommand
{
public:
    explicit SetBridgePropertiesCommand(Bridge *bridge, const QString &id, const QString &file, const QString &name, int type, double length, DataCommand *parent = NULL);
    virtual ~SetBridgePropertiesCommand();

    virtual int id() const
    {
        return 0x1012;
    }

    virtual void undo();
    virtual void redo();

private:
    SetBridgePropertiesCommand(); /* not allowed */
    SetBridgePropertiesCommand(const SetBridgePropertiesCommand &); /* not allowed */
    SetBridgePropertiesCommand &operator=(const SetBridgePropertiesCommand &); /* not allowed */

private:
    Bridge *bridge_;
    QString newId_;
    QString newName_;
    QString newFile_;
    QString oldId_;
    QString oldName_;
    QString oldFile_;
    int newType_;
    int oldType_;
    double newLength_;
    double oldLength_;
};

#endif // SIGNALCOMMANDS_HPP
