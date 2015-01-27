/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   14.06.2010
**
**************************************************************************/

#include "scenerycommands.hpp"

#include "src/data/scenerysystem/scenerysystem.hpp"
#include "src/data/scenerysystem/scenerymap.hpp"
#include "src/data/scenerysystem/heightmap.hpp"

//###############//
// AddMapCommand //
//###############//

AddMapCommand::AddMapCommand(ScenerySystem *scenerySystem, SceneryMap *map, DataCommand *parent)
    : DataCommand(parent)
    , scenerySystem_(scenerySystem)
    , map_(map)
{
    // Check for validity //
    //
    if (!map || !scenerySystem)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("New Map: invalid parameters! No map or ScenerySystem given."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("New Map"));
    }
}

/*! \brief .
*
*/
AddMapCommand::~AddMapCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        delete map_;
    }
    else
    {
        // nothing to be done (map is now owned by the ScenerySystem)
    }
}

/*! \brief .
*
*/
void
AddMapCommand::redo()
{
    scenerySystem_->addSceneryMap(map_);

    setRedone();
}

/*! \brief
*
*/
void
AddMapCommand::undo()
{
    scenerySystem_->delSceneryMap(map_);

    setUndone();
}

//###############//
// DelMapCommand //
//###############//

DelMapCommand::DelMapCommand(SceneryMap *map, DataCommand *parent)
    : DataCommand(parent)
    , scenerySystem_(NULL)
    , map_(map)

{
    // Check for validity //
    //
    if (!map)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Delete Map: invalid parameters! No map given."));
        return;
    }
    else
    {
        scenerySystem_ = map->getParentScenerySystem();

        setValid();
        setText(QObject::tr("Delete Map"));
    }
}

/*! \brief .
*
*/
DelMapCommand::~DelMapCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        // nothing to be done (map is still owned by the ScenerySystem)
    }
    else
    {
        delete map_;
    }
}

/*! \brief .
*
*/
void
DelMapCommand::redo()
{
    scenerySystem_->delSceneryMap(map_);

    setRedone();
}

/*! \brief
*
*/
void
DelMapCommand::undo()
{
    scenerySystem_->addSceneryMap(map_);

    setUndone();
}

//###############//
// SetMapPositionCommand //
//###############//

SetMapPositionCommand::SetMapPositionCommand(SceneryMap *map, double x, double y, DataCommand *parent)
    : DataCommand(parent)
    , map_(map)
    , newX_(x)
    , newY_(y)
{
    // Old x/y //
    //
    oldX_ = map_->getX();
    oldY_ = map_->getY();

    // Check for validity //
    //
    if (!map || ((newX_ == oldX_) && (newY_ == oldY_)))
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Move Map: invalid parameters! No map given or no change necessary."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Move Map"));
    }
}

/*! \brief .
*
*/
SetMapPositionCommand::~SetMapPositionCommand()
{
}

/*! \brief .
*
*/
void
SetMapPositionCommand::redo()
{
    map_->setX(newX_);
    map_->setY(newY_);

    setRedone();
}

/*! \brief
*
*/
void
SetMapPositionCommand::undo()
{
    map_->setX(oldX_);
    map_->setY(oldY_);

    setUndone();
}

/*! \brief Attempts to merge this command with other. Returns true on success; otherwise returns false.
*
*/
bool
SetMapPositionCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids //
    //
    if (other->id() != id())
    {
        return false;
    }

    const SetMapPositionCommand *command = static_cast<const SetMapPositionCommand *>(other);

    // Check parameters //
    //
    if (map_ != command->map_)
    {
        return false;
    }

    // Success //
    //
    newX_ = command->newX_; // adjust to new point, then let the undostack kill the new command
    newY_ = command->newY_;

    return true;
}

//###############//
// SetMapSizeCommand //
//###############//

SetMapSizeCommand::SetMapSizeCommand(SceneryMap *map, double width, double height, DataCommand *parent)
    : DataCommand(parent)
    , map_(map)
    , newWidth_(width)
    , newHeight_(height)
{
    // Value //
    //
    oldWidth_ = map_->getWidth();
    oldHeight_ = map_->getHeight();

    // Check for validity //
    //
    if (!map || (newWidth_ == oldWidth_ && newHeight_ == oldHeight_))
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Set Size: invalid parameters! No map given or no change necessary."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Set Size"));
    }
}

/*! \brief .
*
*/
SetMapSizeCommand::~SetMapSizeCommand()
{
}

/*! \brief .
*
*/
void
SetMapSizeCommand::redo()
{
    map_->setWidth(newWidth_);
    map_->setHeight(newHeight_);

    setRedone();
}

/*! \brief
*
*/
void
SetMapSizeCommand::undo()
{
    map_->setWidth(oldWidth_);
    map_->setHeight(oldHeight_);

    setUndone();
}

//###############//
// SetMapOpacityCommand //
//###############//

SetMapOpacityCommand::SetMapOpacityCommand(SceneryMap *map, double opacity, DataCommand *parent)
    : DataCommand(parent)
    , map_(map)
    , newOpacity_(opacity)
{
    // Opacity //
    //
    oldOpacity_ = map_->getOpacity();

    // Check for validity //
    //
    if (!map || (newOpacity_ == oldOpacity_))
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Set Opacity: invalid parameters! No map given or no change necessary."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Set Opacity"));
    }
}

/*! \brief .
*
*/
SetMapOpacityCommand::~SetMapOpacityCommand()
{
}

/*! \brief .
*
*/
void
SetMapOpacityCommand::redo()
{
    map_->setOpacity(newOpacity_);

    setRedone();
}

/*! \brief
*
*/
void
SetMapOpacityCommand::undo()
{
    map_->setOpacity(oldOpacity_);

    setUndone();
}

/*! \brief Attempts to merge this command with other. Returns true on success; otherwise returns false.
*
*/
bool
SetMapOpacityCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids //
    //
    if (other->id() != id())
    {
        return false;
    }

    const SetMapOpacityCommand *command = static_cast<const SetMapOpacityCommand *>(other);

    // Check parameters //
    //
    if (map_ != command->map_)
    {
        return false;
    }

    // Success //
    //
    newOpacity_ = command->newOpacity_; // adjust to new point, then let the undostack kill the new command

    return true;
}

//###############//
// SetMapFilenameCommand //
//###############//

SetMapFilenameCommand::SetMapFilenameCommand(SceneryMap *map, const QString &filename, DataCommand *parent)
    : DataCommand(parent)
    , map_(map)
    , newFilename_(filename)
{
    // Opacity //
    //
    oldFilename_ = map_->getFilename();

    // Check for validity //
    //
    if (!map || (filename.isEmpty()))
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Set Filename: invalid parameters! No map or no filename given."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Set Filename"));
    }
}

/*! \brief .
*
*/
SetMapFilenameCommand::~SetMapFilenameCommand()
{
}

/*! \brief .
*
*/
void
SetMapFilenameCommand::redo()
{
    map_->setFilename(newFilename_);

    setRedone();
}

/*! \brief
*
*/
void
SetMapFilenameCommand::undo()
{
    map_->setFilename(oldFilename_);

    setUndone();
}

//###############//
// SetHeightmapDataFilenameCommand //
//###############//

SetHeightmapDataFilenameCommand::SetHeightmapDataFilenameCommand(Heightmap *map, const QString &filename, DataCommand *parent)
    : DataCommand(parent)
    , map_(map)
    , newFilename_(filename)
{
    // Opacity //
    //
    oldFilename_ = map_->getFilename();

    // Check for validity //
    //
    if (!map || (filename.isEmpty()) || filename == map_->getHeightmapDataFilename())
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Set Filename: invalid parameters! No map or no filename given."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Set Filename"));
    }
}

SetHeightmapDataFilenameCommand::~SetHeightmapDataFilenameCommand()
{
}

void
SetHeightmapDataFilenameCommand::redo()
{
    map_->setHeightmapDataFilename(newFilename_);

    setRedone();
}

void
SetHeightmapDataFilenameCommand::undo()
{
    map_->setHeightmapDataFilename(oldFilename_);

    setUndone();
}

//###############//
// AddHeightmapCommand //
//###############//

AddHeightmapCommand::AddHeightmapCommand(ScenerySystem *scenerySystem, Heightmap *map, DataCommand *parent)
    : DataCommand(parent)
    , scenerySystem_(scenerySystem)
    , map_(map)
{
    // Check for validity //
    //
    if (!map || !scenerySystem)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("New Map: invalid parameters! No map or ScenerySystem given."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("New Map"));
    }
}

AddHeightmapCommand::~AddHeightmapCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        delete map_;
    }
    else
    {
        // nothing to be done (map is now owned by the ScenerySystem)
    }
}

void
AddHeightmapCommand::redo()
{
    scenerySystem_->addHeightmap(map_);

    setRedone();
}

void
AddHeightmapCommand::undo()
{
    scenerySystem_->delHeightmap(map_);

    setUndone();
}

//###############//
// RemoveHeightmapCommand //
//###############//

RemoveHeightmapCommand::RemoveHeightmapCommand(Heightmap *map, DataCommand *parent)
    : DataCommand(parent)
    , scenerySystem_(NULL)
    , map_(map)

{
    // Check for validity //
    //
    if (!map)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Delete Map: invalid parameters! No map given."));
        return;
    }
    else
    {
        scenerySystem_ = map->getParentScenerySystem();

        setValid();
        setText(QObject::tr("Delete Map"));
    }
}

RemoveHeightmapCommand::~RemoveHeightmapCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        // nothing to be done (map is still owned by the ScenerySystem)
    }
    else
    {
        delete map_;
    }
}

void
RemoveHeightmapCommand::redo()
{
    scenerySystem_->delHeightmap(map_);

    setRedone();
}

void
RemoveHeightmapCommand::undo()
{
    scenerySystem_->addHeightmap(map_);

    setUndone();
}
