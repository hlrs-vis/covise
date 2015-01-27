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

#ifndef TILECOMMANDS_HPP
#define TILECOMMANDS_HPP

// 1000

#include "datacommand.hpp"

#include <QMap>
#include <QPointF>

class TileSystem;
class Tile;
class RoadSystem;
class RSystemElementRoad;
class RSystemElementJunction;
class RSystemElementController;
class RSystemElementFiddleyard;
class RSystemElementPedFiddleyard;
class JunctionConnection;

//#########################//
// NewTileCommand //
//#########################//

class NewTileCommand : public DataCommand
{
public:
    explicit NewTileCommand(Tile *newTile, TileSystem *tileSystem, DataCommand *parent = NULL);
    virtual ~NewTileCommand();

    virtual int id() const
    {
        return 0x1004;
    }

    virtual void undo();
    virtual void redo();

private:
    NewTileCommand(); /* not allowed */
    NewTileCommand(const NewTileCommand &); /* not allowed */
    NewTileCommand &operator=(const NewTileCommand &); /* not allowed */

private:
    Tile *newTile_;
    TileSystem *tileSystem_;
};

//#########################//
// RemoveTileCommand //
//#########################//

class RemoveTileCommand : public DataCommand
{
public:
    explicit RemoveTileCommand(Tile *tile, RoadSystem *roadSystem, DataCommand *parent = NULL);
    virtual ~RemoveTileCommand();

    virtual int id() const
    {
        return 0x1008;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveTileCommand(); /* not allowed */
    RemoveTileCommand(const RemoveTileCommand &); /* not allowed */
    RemoveTileCommand &operator=(const RemoveTileCommand &); /* not allowed */

private:
    Tile *tile_;
    TileSystem *tileSystem_;

    RoadSystem *roadSystem_;
    QList<RSystemElementRoad *> tileRoads_;
    QList<RSystemElementJunction *> tileJunctions_;
    QList<RSystemElementFiddleyard *> tileFiddleyards_;
    QList<RSystemElementPedFiddleyard *> tilePedFiddleyards_;
    QList<RSystemElementController *> tileControllers_;

    QMap<RSystemElementJunction *, QList<JunctionConnection *> > junctionConnections_;
};

#endif // TILECOMMANDS_HPP
