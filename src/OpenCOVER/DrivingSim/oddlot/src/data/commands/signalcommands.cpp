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

#include "signalcommands.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/sections/roadsection.hpp"

#include "src/data/commands/roadsectioncommands.hpp"

//#########################//
// AddSignalCommand //
//#########################//

AddSignalCommand::AddSignalCommand(Signal *signal, RSystemElementRoad *road, DataCommand *parent)
    : DataCommand(parent)
    , signal_(signal)
    , road_(road)
{
    // Check for validity //
    //
    if (!signal || !road_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("AddSignalCommand: Internal error! No road specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("AddSignal"));
    }
}

/*! \brief .
*
*/
AddSignalCommand::~AddSignalCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        delete signal_;
    }
    else
    {
        // nothing to be done (signal is now owned by the road)
    }
}

/*! \brief .
*
*/
void
AddSignalCommand::redo()
{
    road_->addSignal(signal_);

    setRedone();
}

/*! \brief
*
*/
void
AddSignalCommand::undo()
{
    road_->delSignal(signal_);

    setUndone();
}

//#########################//
// RemoveSignalCommand //
//#########################//

RemoveSignalCommand::RemoveSignalCommand(Signal *signal, RSystemElementRoad *road, DataCommand *parent)
    : DataCommand(parent)
    , signal_(signal)
    , road_(road)
{
    // Check for validity //
    //
    if (!signal || !road_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("RemoveSignalCommand: Internal error! No road specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("RemoveSignal"));
    }
}

/*! \brief .
*
*/
RemoveSignalCommand::~RemoveSignalCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        // nothing to be done (signal is now owned by the road)
    }
    else
    {
        delete signal_;
    }
}

/*! \brief .
*
*/
void
RemoveSignalCommand::redo()
{
    road_->delSignal(signal_);

    setRedone();
}

/*! \brief
*
*/
void
RemoveSignalCommand::undo()
{
    road_->addSignal(signal_);

    setUndone();
}

//#########################//
// SetSignalPropertiesCommand //
//#########################//
SetSignalPropertiesCommand::SetSignalPropertiesCommand(Signal *signal, const QString &id, const QString &name, double t, bool dynamic, Signal::OrientationType orientation, double zOffset, const QString &country, int type, const QString &typeSubclass, int subtype, double value, bool pole, int size, int fromLane, int toLane, double probability, double resetTime, DataCommand *parent)
    : DataCommand(parent)
    , newId_(id)
    , newName_(name)
    , signal_(signal)
{
    // Check for validity //
    //
    if (!signal)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("SetSignalPropertiesCommand: Internal error! No signal specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("SetProperties"));
    }

    road_ = signal_->getParentRoad();

    oldId_ = signal_->getId();
    oldName_ = signal_->getName();

    newSignalProps_.t = t;
    newSignalProps_.dynamic = dynamic;
    newSignalProps_.orientation = orientation;
    newSignalProps_.zOffset = zOffset;
    newSignalProps_.country = country;
    newSignalProps_.type = type;
    newSignalProps_.typeSubclass = typeSubclass;
    newSignalProps_.subtype = subtype;
    newSignalProps_.value = value;
    newSignalProps_.pole = pole;
    newSignalProps_.size = size;
    newValidity_.fromLane = fromLane;
    newValidity_.toLane = toLane;
    newUserData_.crossProb = probability;
    newUserData_.resetTime = resetTime;

    oldSignalProps_.t = signal_->getT();
    oldSignalProps_.dynamic = signal_->getDynamic();
    oldSignalProps_.orientation = signal_->getOrientation();
    oldSignalProps_.zOffset = signal_->getZOffset();
    oldSignalProps_.country = signal_->getCountry();
    oldSignalProps_.type = signal_->getType();
    oldSignalProps_.subtype = signal_->getSubtype();
    oldSignalProps_.value = signal_->getValue();
    oldSignalProps_.pole = signal_->getPole();
    oldSignalProps_.size = signal_->getSize();
    oldValidity_.fromLane = signal_->getValidFromLane();
    oldValidity_.toLane = signal_->getValidToLane();
    oldUserData_.crossProb = signal->getCrossingProbability();
    oldUserData_.resetTime = resetTime;
}

SetSignalPropertiesCommand::SetSignalPropertiesCommand(Signal *signal, const QString &id, const QString &name, const Signal::SignalProperties &signalProps, const Signal::Validity &validLanes, const Signal::SignalUserData &userData, DataCommand *parent)
    : DataCommand(parent)
    , newId_(id)
    , newName_(name)
    , signal_(signal)
    , newSignalProps_(signalProps)
    , newValidity_(validLanes)
    , newUserData_(userData)
{
    // Check for validity //
    //
    if (!signal)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("SetSignalPropertiesCommand: Internal error! No signal specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("SetSignalProperties"));
    }

    oldId_ = signal_->getId();
    oldName_ = signal_->getName();
    oldSignalProps_ = signal_->getProperties();
    oldValidity_ = signal_->getValidity();
    oldUserData_ = signal_->getSignalUserData();
}

/*! \brief .
*
*/
SetSignalPropertiesCommand::~SetSignalPropertiesCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
    }
    else
    {
        // nothing to be done (signal is now owned by the road)
    }
}

/*! \brief .
*
*/
void
SetSignalPropertiesCommand::redo()
{

    //	road_->delSignal(signal_);

    signal_->setId(newId_);
    signal_->setName(newName_);
    signal_->setProperties(newSignalProps_);
    signal_->setValidity(newValidity_);
    signal_->setSignalUserData(newUserData_);
    //	road_->addSignal(signal_);

    signal_->addSignalChanges(Signal::CEL_ParameterChange);

    setRedone();
}

/*! \brief
*
*/
void
SetSignalPropertiesCommand::undo()
{
    //	road_->delSignal(signal_);

    signal_->setId(oldId_);
    signal_->setName(oldName_);
    signal_->setProperties(oldSignalProps_);
    signal_->setValidity(oldValidity_);
    signal_->setSignalUserData(oldUserData_);

    //	road_->addSignal(signal_);

    signal_->addSignalChanges(Signal::CEL_ParameterChange);

    setUndone();
}

//#########################//
// AddObjectCommand //
//#########################//

AddObjectCommand::AddObjectCommand(Object *object, RSystemElementRoad *road, DataCommand *parent)
    : DataCommand(parent)
    , object_(object)
    , road_(road)
{
    // Check for validity //
    //
    if (!object || !road_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("AddObjectCommand: Internal error! No road specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("AddObject"));
    }
}

/*! \brief .
*
*/
AddObjectCommand::~AddObjectCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        delete object_;
    }
    else
    {
        // nothing to be done (object is now owned by the road)
    }
}

/*! \brief .
*
*/
void
AddObjectCommand::redo()
{
    road_->addObject(object_);

    setRedone();
}

/*! \brief
*
*/
void
AddObjectCommand::undo()
{
    road_->delObject(object_);

    setUndone();
}

//#########################//
// RemoveObjectCommand //
//#########################//

RemoveObjectCommand::RemoveObjectCommand(Object *object, RSystemElementRoad *road, DataCommand *parent)
    : DataCommand(parent)
    , object_(object)
    , road_(road)
{
    // Check for validity //
    //
    if (!object || !road_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("RemoveObjectCommand: Internal error! No road specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("RemoveObject"));
    }
}

/*! \brief .
*
*/
RemoveObjectCommand::~RemoveObjectCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        // nothing to be done (object is now owned by the road)
    }
    else
    {
        delete object_;
    }
}

/*! \brief .
*
*/
void
RemoveObjectCommand::redo()
{
    road_->delObject(object_);

    setRedone();
}

/*! \brief
*
*/
void
RemoveObjectCommand::undo()
{
    road_->addObject(object_);

    setUndone();
}

//#########################//
// SetObjectPropertiesCommand //
//#########################//
SetObjectPropertiesCommand::SetObjectPropertiesCommand(Object *object, const QString &id, const QString &name, const QString &type, double t, double zOffset,
                                                       double validLength, Object::ObjectOrientation orientation, double length, double width, double radius, double height, double hdg, double pitch, double roll, bool pole,
                                                       double repeatS, double repeatLength, double repeatDistance, const QString &textureFile, DataCommand *parent)
    : DataCommand(parent)
    , newId_(id)
    , newName_(name)
    , newTextureFile_(textureFile)
    , object_(object)
{
    // Check for validity //
    //
    if (!object)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("SetObjectPropertiesCommand: Internal error! No object specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("SetProperties"));
    }

    oldId_ = object_->getId();
    oldName_ = object_->getName();
    oldTextureFile_ = object_->getTextureFileName();
    newObjectProps_.t = t;
    newObjectProps_.orientation = orientation;
    newObjectProps_.zOffset = zOffset;
    newObjectProps_.type = type;
    newObjectProps_.validLength = validLength;
    newObjectProps_.length = length;
    newObjectProps_.width = width;
    newObjectProps_.radius = radius;
    newObjectProps_.height = height;
    newObjectProps_.hdg = hdg;
    newObjectProps_.pitch = pitch;
    newObjectProps_.roll = roll;
    newObjectProps_.pole = pole;
    newObjectRepeat_.s = repeatS;
    newObjectRepeat_.length = repeatLength;
    newObjectRepeat_.distance = repeatDistance;

    oldObjectProps_.t = object_->getT();
    oldObjectProps_.orientation = object_->getOrientation();
    oldObjectProps_.zOffset = object_->getzOffset();
    oldObjectProps_.type = object_->getType();
    oldObjectProps_.validLength = object->getValidLength();
    oldObjectProps_.length = object->getLength();
    oldObjectProps_.width = object->getWidth();
    oldObjectProps_.radius = object->getRadius();
    oldObjectProps_.height = object->getHeight();
    oldObjectProps_.hdg = object->getHeading();
    oldObjectProps_.pitch = object->getPitch();
    oldObjectProps_.roll = object->getRoll();
    oldObjectProps_.pole = object->getPole();
    oldObjectRepeat_.s = object->getRepeatS();
    oldObjectRepeat_.length = object->getRepeatLength();
    oldObjectRepeat_.distance = object->getRepeatDistance();
}

SetObjectPropertiesCommand::SetObjectPropertiesCommand(Object *object, const QString &id, const QString &name, Object::ObjectProperties &objectProps, Object::ObjectRepeatRecord &objectRepeat, const QString &textureFile, DataCommand *parent)
    : DataCommand(parent)
    , newId_(id)
    , newName_(name)
    , newTextureFile_(textureFile)
    , object_(object)
    , newObjectProps_(objectProps)
    , newObjectRepeat_(objectRepeat)
{
    // Check for validity //
    //
    if (!object)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("SetObjectPropertiesCommand: Internal error! No object specified."));
        return;
    }

    oldId_ = object_->getId();
    oldName_ = object_->getName();
    oldTextureFile_ = object_->getTextureFileName();
    oldObjectProps_ = object_->getProperties();
    oldObjectRepeat_ = object_->getRepeatProperties();

    setValid();
    setText(QObject::tr("SetObjectProperties"));
}

/*! \brief .
*
*/
SetObjectPropertiesCommand::~SetObjectPropertiesCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
    }
    else
    {
        // nothing to be done (object is now owned by the road)
    }
}

/*! \brief .
*
*/
void
SetObjectPropertiesCommand::redo()
{
    object_->setId(newId_);
    object_->setName(newName_);
    object_->setModelFileName(newName_);
    object_->setTextureFileName(newTextureFile_);
    object_->setProperties(newObjectProps_);
    object_->setRepeatProperties(newObjectRepeat_);

    object_->addObjectChanges(Object::CEL_ParameterChange);

    setRedone();
}

/*! \brief
*
*/
void
SetObjectPropertiesCommand::undo()
{
    object_->setId(oldId_);
    object_->setName(oldName_);
    object_->setModelFileName(oldName_);
    object_->setTextureFileName(oldTextureFile_);
    object_->setProperties(oldObjectProps_);
    object_->setRepeatProperties(oldObjectRepeat_);

    object_->addObjectChanges(Object::CEL_ParameterChange);

    setUndone();
}

//#########################//
// AddBridgeCommand //
//#########################//

AddBridgeCommand::AddBridgeCommand(Bridge *bridge, RSystemElementRoad *road, DataCommand *parent)
    : DataCommand(parent)
    , bridge_(bridge)
    , road_(road)
{
    // Check for validity //
    //
    if (!bridge || !road_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("AddBridgeCommand: Internal error! No road specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("AddBridge"));
    }
}

/*! \brief .
*
*/
AddBridgeCommand::~AddBridgeCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        delete bridge_;
    }
    else
    {
        // nothing to be done (bridge is now owned by the road)
    }
}

/*! \brief .
*
*/
void
AddBridgeCommand::redo()
{
    road_->addBridge(bridge_);

    setRedone();
}

/*! \brief
*
*/
void
AddBridgeCommand::undo()
{
    road_->delBridge(bridge_);

    setUndone();
}

//#########################//
// RemoveBridgeCommand //
//#########################//

RemoveBridgeCommand::RemoveBridgeCommand(Bridge *bridge, RSystemElementRoad *road, DataCommand *parent)
    : DataCommand(parent)
    , bridge_(bridge)
    , road_(road)
{
    // Check for validity //
    //
    if (!bridge || !road_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("RemoveBridgeCommand: Internal error! No road specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("RemoveBridge"));
    }
}

/*! \brief .
*
*/
RemoveBridgeCommand::~RemoveBridgeCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        // nothing to be done (bridge is now owned by the road)
    }
    else
    {
        delete bridge_;
    }
}

/*! \brief .
*
*/
void
RemoveBridgeCommand::redo()
{
    road_->delBridge(bridge_);

    setRedone();
}

/*! \brief
*
*/
void
RemoveBridgeCommand::undo()
{
    road_->addBridge(bridge_);

    setUndone();
}

//#########################//
// SetBridgePropertiesCommand //
//#########################//
SetBridgePropertiesCommand::SetBridgePropertiesCommand(Bridge *bridge, const QString &id, const QString &file, const QString &name, int type, double length, DataCommand *parent)
    : DataCommand(parent)
    , newId_(id)
    , newName_(name)
    , newFile_(file)
    , bridge_(bridge)
    , newType_(type)
    , newLength_(length)
{
    // Check for validity //
    //
    if (!bridge)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("SetBridgePropertiesCommand: Internal error! No bridge specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("SetProperties"));
    }

    oldId_ = bridge_->getId();
    oldName_ = bridge_->getName();
    oldFile_ = bridge_->getFileName();
    oldType_ = bridge_->getType();
    oldLength_ = bridge_->getLength();
}

/*! \brief .
*
*/
SetBridgePropertiesCommand::~SetBridgePropertiesCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
    }
    else
    {
        // nothing to be done (bridge is now owned by the road)
    }
}

/*! \brief .
*
*/
void
SetBridgePropertiesCommand::redo()
{
    bridge_->setId(newId_);
    bridge_->setName(newName_);
    bridge_->setFileName(newFile_);
    bridge_->setType(newType_);
    bridge_->setLength(newLength_);

    bridge_->addBridgeChanges(Bridge::CEL_ParameterChange);

    setRedone();
}

/*! \brief
*
*/
void
SetBridgePropertiesCommand::undo()
{
    bridge_->setId(oldId_);
    bridge_->setName(oldName_);
    bridge_->setFileName(oldFile_);
    bridge_->setType(oldType_);
    bridge_->setLength(oldLength_);

    bridge_->addBridgeChanges(Bridge::CEL_ParameterChange);

    setUndone();
}
