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

#include "tilecommands.hpp"

#include "src/data/tilesystem/tilesystem.hpp"

#include "src/data/roadsystem/roadsystem.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/rsystemelementcontroller.hpp"
#include "src/data/roadsystem/rsystemelementjunction.hpp"
#include "src/data/roadsystem/rsystemelementfiddleyard.hpp"
#include "src/data/roadsystem/rsystemelementpedfiddleyard.hpp"

#include "src/data/roadsystem/junctionconnection.hpp"

//#########################//
// NewTileCommand //
//#########################//

NewTileCommand::NewTileCommand(Tile *newTile, TileSystem *tileSystem, DataCommand *parent)
    : DataCommand(parent)
    , newTile_(newTile)
    , tileSystem_(tileSystem)
{
    // Check for validity //
    //
    if (!newTile || !tileSystem_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("NewTileCommand: Internal error! No new tile specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("New Tile"));
    }
}

/*! \brief .
*
*/
NewTileCommand::~NewTileCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        delete newTile_;
    }
    else
    {
        // nothing to be done (tile is now owned by the tilesystem)
    }
}

/*! \brief .
*
*/
void
NewTileCommand::redo()
{
    tileSystem_->addTile(newTile_);

    setRedone();
}

/*! \brief
*
*/
void
NewTileCommand::undo()
{
    tileSystem_->delTile(newTile_);

    setUndone();
}

//#########################//
// RemoveTileCommand //
//#########################//

RemoveTileCommand::RemoveTileCommand(Tile *tile, RoadSystem *roadSystem, DataCommand *parent)
    : DataCommand(parent)
    , tile_(tile)
    , roadSystem_(roadSystem)
{
    // Check for validity //
    //
    if (!tile_ || !tile_->getTileSystem())
    {
        setInvalid(); // Invalid
        setText(QObject::tr("RemoveTileCommand: Internal error! No tile specified."));
        return;
    }

    tileSystem_ = tile->getTileSystem();
    tileRoads_ = roadSystem_->getTileRoads(tile->getID());
    tileControllers_ = roadSystem_->getTileControllers(tile->getID());
    tileJunctions_ = roadSystem_->getTileJunctions(tile->getID());

    for (int i = 0; i < tileJunctions_.size(); i++)
    {
        junctionConnections_.insert(tileJunctions_.at(i), tileJunctions_.at(i)->getConnections().values());
    }

    tileFiddleyards_ = roadSystem_->getTileFiddleyards(tile->getID());
    tilePedFiddleyards_ = roadSystem_->getTilePedFiddleyards(tile->getID());

    setValid();
    setText(QObject::tr("Remove Tile"));
}

/*! \brief .
*
*/
RemoveTileCommand::~RemoveTileCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        // nothing to be done (tile is still owned by the tilesystem)
    }
    else
    {
        tileRoads_.clear();
        tileControllers_.clear();
        tileJunctions_.clear();
        tileFiddleyards_.clear();
        tilePedFiddleyards_.clear();
    }
}

/*! \brief .
*
*/
void
RemoveTileCommand::redo()
{
    for (int i = 0; i < tileRoads_.size(); i++)
    {
        roadSystem_->delRoad(tileRoads_.at(i));
    }

    for (int i = 0; i < tileJunctions_.size(); i++)
    {
        roadSystem_->delJunction(tileJunctions_.at(i));
    }

    for (int i = 0; i < tileControllers_.size(); i++)
    {
        roadSystem_->delController(tileControllers_.at(i));
    }

    for (int i = 0; i < tileFiddleyards_.size(); i++)
    {
        roadSystem_->delFiddleyard(tileFiddleyards_.at(i));
    }

    for (int i = 0; i < tilePedFiddleyards_.size(); i++)
    {
        roadSystem_->delPedFiddleyard(tilePedFiddleyards_.at(i));
    }

    tileSystem_->delTile(tile_);

    setRedone();
}

/*! \brief
*
*/
void
RemoveTileCommand::undo()
{
    if (!tileSystem_->getTile(tile_->getID()))
    {
        tileSystem_->addTile(tile_);
    }

    for (int i = 0; i < tileRoads_.size(); i++)
    {
        roadSystem_->addRoad(tileRoads_.at(i));
    }

    for (int i = 0; i < tileJunctions_.size(); i++)
    {
        RSystemElementJunction *junction = tileJunctions_.at(i);
        QList<JunctionConnection *> connectionList = junctionConnections_.value(junction);
        for (int j = 0; j < connectionList.size(); j++)
        {
            junction->addConnection(connectionList.at(j));
        }

        roadSystem_->addJunction(junction);
    }

    for (int i = 0; i < tileControllers_.size(); i++)
    {
        roadSystem_->addController(tileControllers_.at(i));
    }

    for (int i = 0; i < tileFiddleyards_.size(); i++)
    {
        roadSystem_->addFiddleyard(tileFiddleyards_.at(i));
    }

    for (int i = 0; i < tilePedFiddleyards_.size(); i++)
    {
        roadSystem_->addPedFiddleyard(tilePedFiddleyards_.at(i));
    }

    setUndone();
}
