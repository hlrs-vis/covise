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
 //       delete signal_;
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
SetSignalPropertiesCommand::SetSignalPropertiesCommand(Signal *signal, const QString &id, const QString &name, double t, bool dynamic, Signal::OrientationType orientation, double zOffset, const QString &country, const QString &type, const QString &typeSubclass, const QString &subtype, double value, double hOffset, double pitch, double roll, const QString &unit, const QString &text, double width, double height, bool pole, int size, int fromLane, int toLane, double probability, double resetTime, DataCommand *parent)
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
    newSignalProps_.subtype = subtype;
    newSignalProps_.value = value;
    newSignalProps_.hOffset = hOffset;
    newSignalProps_.pitch = pitch;
    newSignalProps_.roll = roll;
	newSignalProps_.unit = unit;
	newSignalProps_.text = text;
	newSignalProps_.width = width;
	newSignalProps_.height = height;
    newSignalProps_.pole = pole;

    newValidity_.fromLane = fromLane;
    newValidity_.toLane = toLane;

    newUserData_.crossProb = probability;
    newUserData_.resetTime = resetTime;
    newUserData_.size = size;
    newUserData_.typeSubclass = typeSubclass;

    oldSignalProps_.t = signal_->getT();
    oldSignalProps_.dynamic = signal_->getDynamic();
    oldSignalProps_.orientation = signal_->getOrientation();
    oldSignalProps_.zOffset = signal_->getZOffset();
    oldSignalProps_.country = signal_->getCountry();
    oldSignalProps_.type = signal_->getType();
    oldSignalProps_.subtype = signal_->getSubtype();

    oldSignalProps_.value = signal_->getValue();
    oldSignalProps_.hOffset = signal_->getHeading();
    oldSignalProps_.pitch = signal_->getPitch();
	oldSignalProps_.roll = signal_->getRoll();
	oldSignalProps_.unit = signal_->getUnit();
	oldSignalProps_.text = signal_->getText();
	oldSignalProps_.width = signal_->getWidth();
	oldSignalProps_.height = signal_->getHeight();
    oldSignalProps_.pole = signal_->getPole();

    oldValidity_.fromLane = signal_->getValidFromLane();
    oldValidity_.toLane = signal_->getValidToLane();
    oldUserData_.crossProb = signal->getCrossingProbability();
    oldUserData_.resetTime = signal->getResetTime();
    oldUserData_.size = signal_->getSize();
    oldUserData_.typeSubclass = signal_->getTypeSubclass();
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

    if ((newSignalProps_.type != oldSignalProps_.type) || (newSignalProps_.subtype != oldSignalProps_.subtype) || (newUserData_.typeSubclass != oldUserData_.typeSubclass))
    {
        signal_->addSignalChanges(Signal::CEL_TypeChange);
    }
    else
    {
        signal_->addSignalChanges(Signal::CEL_ParameterChange);
    }

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
    if ((newSignalProps_.type != oldSignalProps_.type) || (newSignalProps_.subtype != oldSignalProps_.subtype) || (newUserData_.typeSubclass != oldUserData_.typeSubclass))
    {
        signal_->addSignalChanges(Signal::CEL_TypeChange);
    }
    else
    {
        signal_->addSignalChanges(Signal::CEL_ParameterChange);
    }

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
//        delete object_;
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


SetObjectPropertiesCommand::SetObjectPropertiesCommand(Object *object, const QString &id, const QString &name, const Object::ObjectProperties &objectProps, const Object::ObjectRepeatRecord &objectRepeat, const QString &textureFile, DataCommand *parent)
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

	if (newObjectProps_.type != oldObjectProps_.type)
	{
		object_->addObjectChanges(Object::CEL_TypeChange);
	}
	else
	{
		object_->addObjectChanges(Object::CEL_ParameterChange);
	}

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

	if (newObjectProps_.type != oldObjectProps_.type)
	{
		object_->addObjectChanges(Object::CEL_TypeChange);
	}
	else
	{
		object_->addObjectChanges(Object::CEL_ParameterChange);
	}

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
        setText(QObject::tr("AddBridge/AddTunnel"));
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
 //       delete bridge_;
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

//#########################//
// SetTunnelPropertiesCommand //
//#########################//
SetTunnelPropertiesCommand::SetTunnelPropertiesCommand(Tunnel *tunnel, const QString &id, const QString &file, const QString &name, int type, double length, double lighting, double daylight, DataCommand *parent)
	: SetBridgePropertiesCommand(tunnel, id, file, name, type, length, parent)
    , tunnel_(tunnel)
	, newLighting_(lighting)
	, newDaylight_(daylight)
{
    // Check for validity //
    //
    if (!tunnel)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("SetTunnelPropertiesCommand: Internal error! No tunnel specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("SetProperties"));
    }

    oldLighting_ = tunnel_->getLighting();
	oldDaylight_ = tunnel_->getDaylight();
}

/*! \brief .
*
*/
SetTunnelPropertiesCommand::~SetTunnelPropertiesCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
    }
    else
    {
        // nothing to be done (tunnel is now owned by the road)
    }
}

/*! \brief .
*
*/
void
SetTunnelPropertiesCommand::redo()
{
	SetBridgePropertiesCommand::redo();	// pass to baseclass /

	tunnel_->setLighting(newLighting_);
	tunnel_->setDaylight(newDaylight_);

    tunnel_->addTunnelChanges(Tunnel::CEL_ParameterChange);

    setRedone();
}

/*! \brief
*
*/
void
SetTunnelPropertiesCommand::undo()
{
	SetBridgePropertiesCommand::undo();

    tunnel_->setLighting(oldLighting_);
	tunnel_->setDaylight(oldDaylight_);

    tunnel_->addTunnelChanges(Tunnel::CEL_ParameterChange);

    setUndone();
}

