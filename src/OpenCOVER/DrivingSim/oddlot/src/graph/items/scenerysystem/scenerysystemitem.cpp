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

#include "scenerysystemitem.hpp"

// Data //
//
#include "src/data/scenerysystem/scenerysystem.hpp"
#include "src/data/scenerysystem/scenerymap.hpp"
#include "src/data/scenerysystem/heightmap.hpp"
#include "src/data/commands/scenerycommands.hpp"

// Graph //
//
#include "src/graph/projectgraph.hpp"

// Items //
//
#include "scenerymapitem.hpp"
#include "heightmapitem.hpp"

ScenerySystemItem::ScenerySystemItem(TopviewGraph *topviewGraph, ScenerySystem *scenerySystem)
    : GraphElement(NULL, scenerySystem)
    , topviewGraph_(topviewGraph)
    , scenerySystem_(scenerySystem)
{
    init();
}

ScenerySystemItem::~ScenerySystemItem()
{
}

void
ScenerySystemItem::init()
{
    // Selection/Highlighting //
    //
    setOpacitySettings(1.0, 1.0); // always highlighted
}

//##################//
// SceneryMapItems  //
//##################//

void
ScenerySystemItem::loadMap(const QString &filename, const QPointF &pos)
{
    // Check file and get size //
    //
    QPixmap pixmap(filename); // this pixmap is only temporary
    if (pixmap.isNull())
    {
        qDebug("ERROR 1006151345! Pixmap could not be loaded!");
        return;
    }

    SceneryMap *map = new SceneryMap("map0", filename, pixmap.width(), pixmap.height(), SceneryMap::DMT_Aerial);
    map->setX(pos.x());
    map->setY(pos.y() - pixmap.height());

    AddMapCommand *command = new AddMapCommand(scenerySystem_, map);
    getProjectGraph()->executeCommand(command);
}

void
ScenerySystemItem::deleteMap()
{
    foreach (SceneryMapItem *mapItem, mapItems_)
    {
        if (mapItem->isSelected())
        {
            SceneryMap *map = mapItem->getMap();
            DelMapCommand *command = new DelMapCommand(map);
            getProjectGraph()->executeCommand(command);
        }
    }
}

void
ScenerySystemItem::lockMaps(bool locked)
{
    foreach (SceneryMapItem *mapItem, mapItems_)
    {
        mapItem->setLocked(locked);
    }
}

void
ScenerySystemItem::setMapOpacity(double opacity)
{

    foreach (SceneryMapItem *mapItem, mapItems_)
    {
        if (mapItem->isSelected())
        {
            SceneryMap *map = mapItem->getMap();
            SetMapOpacityCommand *command = new SetMapOpacityCommand(map, opacity);
            getProjectGraph()->executeCommand(command);
        }
    }
}

void
ScenerySystemItem::setMapX(double x)
{

    foreach (SceneryMapItem *mapItem, mapItems_)
    {
        if (mapItem->isSelected())
        {
            mapItem->setMapX(x);
        }
    }
}

void
ScenerySystemItem::setMapY(double y)
{
    foreach (SceneryMapItem *mapItem, mapItems_)
    {
        if (mapItem->isSelected())
        {
            mapItem->setMapY(y);
        }
    }
}

void
ScenerySystemItem::setMapWith(double width, bool keepRatio)
{

    foreach (SceneryMapItem *mapItem, mapItems_)
    {
        if (mapItem->isSelected())
        {
            mapItem->setMapWidth(width, keepRatio);
        }
    }
}

void
ScenerySystemItem::setMapHeight(double height, bool keepRatio)
{
    foreach (SceneryMapItem *mapItem, mapItems_)
    {
        if (mapItem->isSelected())
        {
            mapItem->setMapHeight(height, keepRatio);
        }
    }
}

//##################//
// Observer Pattern //
//##################//

void
ScenerySystemItem::updateObserver()
{
    // Parent //
    //
    GraphElement::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Get change flags //
    //
    int changes = scenerySystem_->getScenerySystemChanges();

    // Road //
    //
    if ((changes & ScenerySystem::CSC_MapChanged)
        || (changes & ScenerySystem::CSC_HeightmapChanged))
    {
        // A map has been added.
        //
        foreach (SceneryMap *map, scenerySystem_->getSceneryMaps())
        {
            if (!mapItems_.contains(map->getId()))
            {
                // New Item //
                //
                SceneryMapItem *mapItem = new SceneryMapItem(this, map);
                mapItems_.insert(mapItem->getMap()->getId(), mapItem);
            }
        }
        foreach (Heightmap *map, scenerySystem_->getHeightmaps())
        {
            if (!mapItems_.contains(map->getId()))
            {
                HeightmapItem *mapItem = new HeightmapItem(this, map);
                mapItems_.insert(mapItem->getMap()->getId(), mapItem);
            }
        }

        // A map has been deleted.
        //
        foreach (SceneryMapItem *mapItem, mapItems_)
        {
            if (mapItem->getMap()->getDataElementChanges() & DataElement::CDE_DataElementDeleted
                || mapItem->getMap()->getDataElementChanges() & DataElement::CDE_DataElementRemoved)
            {
                mapItems_.remove(mapItem->getMap()->getId());
                //				getProjectGraph()->addToGarbage(mapItem); // done by map itself
            }
        }
    }
}
