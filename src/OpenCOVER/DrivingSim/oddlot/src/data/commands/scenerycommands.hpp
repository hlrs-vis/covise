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

#ifndef SCENERYCOMMANDS_HPP
#define SCENERYCOMMANDS_HPP

// 200

#include "datacommand.hpp"

class ScenerySystem;
class SceneryMap;
class Heightmap;

//###############//
// AddMapCommand //
//###############//

class AddMapCommand : public DataCommand
{
public:
    explicit AddMapCommand(ScenerySystem *scenerySystem, SceneryMap *map, DataCommand *parent = NULL);
    virtual ~AddMapCommand();

    virtual int id() const
    {
        return 0x201;
    }

    virtual void undo();
    virtual void redo();

private:
    AddMapCommand(); /* not allowed */
    AddMapCommand(const AddMapCommand &); /* not allowed */
    AddMapCommand &operator=(const AddMapCommand &); /* not allowed */

private:
    ScenerySystem *scenerySystem_;
    SceneryMap *map_;
};

//###############//
// DelMapCommand //
//###############//

class DelMapCommand : public DataCommand
{
public:
    explicit DelMapCommand(SceneryMap *map, DataCommand *parent = NULL);
    virtual ~DelMapCommand();

    virtual int id() const
    {
        return 0x202;
    }

    virtual void undo();
    virtual void redo();

private:
    DelMapCommand(); /* not allowed */
    DelMapCommand(const DelMapCommand &); /* not allowed */
    DelMapCommand &operator=(const DelMapCommand &); /* not allowed */

private:
    ScenerySystem *scenerySystem_;
    SceneryMap *map_;
};

//###############//
// SetMapPositionCommand //
//###############//

class SetMapPositionCommand : public DataCommand
{
public:
    explicit SetMapPositionCommand(SceneryMap *map, double x, double y, DataCommand *parent = NULL);
    virtual ~SetMapPositionCommand();

    virtual int id() const
    {
        return 0x204;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    SetMapPositionCommand(); /* not allowed */
    SetMapPositionCommand(const SetMapPositionCommand &); /* not allowed */
    SetMapPositionCommand &operator=(const SetMapPositionCommand &); /* not allowed */

private:
    SceneryMap *map_;
    double newX_;
    double newY_;
    double oldX_;
    double oldY_;
};

//###############//
// SetMapSizeCommand //
//###############//

class SetMapSizeCommand : public DataCommand
{
public:
    explicit SetMapSizeCommand(SceneryMap *map, double width, double height, DataCommand *parent = NULL);
    virtual ~SetMapSizeCommand();

    virtual int id() const
    {
        return 0x208;
    }

    virtual void undo();
    virtual void redo();

private:
    SetMapSizeCommand(); /* not allowed */
    SetMapSizeCommand(const SetMapSizeCommand &); /* not allowed */
    SetMapSizeCommand &operator=(const SetMapSizeCommand &); /* not allowed */

private:
    SceneryMap *map_;

    double newWidth_;
    double oldWidth_;

    double newHeight_;
    double oldHeight_;
};

//###############//
// SetMapOpacityCommand //
//###############//

class SetMapOpacityCommand : public DataCommand
{
public:
    explicit SetMapOpacityCommand(SceneryMap *map, double opacity, DataCommand *parent = NULL);
    virtual ~SetMapOpacityCommand();

    virtual int id() const
    {
        return 0x210;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    SetMapOpacityCommand(); /* not allowed */
    SetMapOpacityCommand(const SetMapOpacityCommand &); /* not allowed */
    SetMapOpacityCommand &operator=(const SetMapOpacityCommand &); /* not allowed */

private:
    SceneryMap *map_;
    double newOpacity_;
    double oldOpacity_;
};

//###############//
// SetMapFilenameCommand //
//###############//

class SetMapFilenameCommand : public DataCommand
{
public:
    explicit SetMapFilenameCommand(SceneryMap *map, const QString &filename, DataCommand *parent = NULL);
    virtual ~SetMapFilenameCommand();

    virtual int id() const
    {
        return 0x220;
    }

    virtual void undo();
    virtual void redo();

private:
    SetMapFilenameCommand(); /* not allowed */
    SetMapFilenameCommand(const SetMapFilenameCommand &); /* not allowed */
    SetMapFilenameCommand &operator=(const SetMapFilenameCommand &); /* not allowed */

private:
    SceneryMap *map_;
    QString newFilename_;
    QString oldFilename_;
};

//###############//
// SetMapFilenameCommand //
//###############//

class SetHeightmapDataFilenameCommand : public DataCommand
{
public:
    explicit SetHeightmapDataFilenameCommand(Heightmap *map, const QString &filename, DataCommand *parent = NULL);
    virtual ~SetHeightmapDataFilenameCommand();

    virtual int id() const
    {
        return 0x220;
    }

    virtual void undo();
    virtual void redo();

private:
    SetHeightmapDataFilenameCommand(); /* not allowed */
    SetHeightmapDataFilenameCommand(const SetHeightmapDataFilenameCommand &); /* not allowed */
    SetHeightmapDataFilenameCommand &operator=(const SetHeightmapDataFilenameCommand &); /* not allowed */

private:
    Heightmap *map_;
    QString newFilename_;
    QString oldFilename_;
};

//###############//
// AddHeightmapCommand //
//###############//

class AddHeightmapCommand : public DataCommand
{
public:
    explicit AddHeightmapCommand(ScenerySystem *scenerySystem, Heightmap *map, DataCommand *parent = NULL);
    virtual ~AddHeightmapCommand();

    virtual int id() const
    {
        return 0x251;
    }

    virtual void undo();
    virtual void redo();

private:
    AddHeightmapCommand(); /* not allowed */
    AddHeightmapCommand(const AddHeightmapCommand &); /* not allowed */
    AddHeightmapCommand &operator=(const AddHeightmapCommand &); /* not allowed */

private:
    ScenerySystem *scenerySystem_;
    Heightmap *map_;
};

//###############//
// RemoveHeightmapCommand //
//###############//

class RemoveHeightmapCommand : public DataCommand
{
public:
    explicit RemoveHeightmapCommand(Heightmap *map, DataCommand *parent = NULL);
    virtual ~RemoveHeightmapCommand();

    virtual int id() const
    {
        return 0x252;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveHeightmapCommand(); /* not allowed */
    RemoveHeightmapCommand(const RemoveHeightmapCommand &); /* not allowed */
    RemoveHeightmapCommand &operator=(const RemoveHeightmapCommand &); /* not allowed */

private:
    ScenerySystem *scenerySystem_;
    Heightmap *map_;
};

#endif // SCENERYCOMMANDS_HPP
