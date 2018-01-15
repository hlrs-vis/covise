/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   02.02.2010
**
**************************************************************************/

#include "tilesystem.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "tile.hpp"

/*! \brief CONSTRUCTOR.
*
*/
TileSystem::TileSystem()
    : DataElement()
    , tileSystemChanges_(0x0)
    , currentTile_(NULL)
{
}

TileSystem::~TileSystem()
{
    // Delete child nodes //
    //
    foreach (Tile *child, tiles_)
        delete child;
}

//##################//
// RSystemElements  //
//##################//

Tile *
TileSystem::getTile(const QString &id) const
{
    return tiles_.value(id, NULL);
}

void
TileSystem::addTile(Tile *tile)
{
    if (getProjectData())
    {
        // Id //
        //
        QString name = tile->getName();
        QString id = getUniqueId(tile->getID(), name);
        if (id != tile->getID())
        {
            tile->setID(id);
            if (name != tile->getName())
            {
                tile->setName(name);
            }
        }
    }

    // Insert //
    //
    tile->setTileSystem(this);

    tiles_.insert(tile->getID(), tile);
    addTileSystemChanges(TileSystem::CTS_TileChange);

    setCurrentTile(tile);
}

bool
TileSystem::delTile(Tile *tile)
{
    if (tiles_.size() < 2)
    {
        qDebug("WARNING 1005311351! Tile not deleted because it is the only one!");
        return false;
    }

    if (tiles_.remove(tile->getID()) && tileIds_.removeOne(tile->getID()))
    {
        addTileSystemChanges(TileSystem::CTS_TileChange);

        tile->setTileSystem(NULL);

        if (!tiles_.empty())
        {
            currentTile_ = tiles_.begin().value();
        }
        else
        {
            currentTile_ = NULL;
        }

        return true;
    }
    else
    {
        qDebug("WARNING 1005311350! Delete tile not successful!");
        return false;
    }
}

void
TileSystem::setCurrentTile(Tile *tile)
{
	if (currentTile_)
	{
		currentTile_->setElementSelected(false);
	}
	if (tile !=NULL)
	{
		currentTile_ = tile;
		tile->setElementSelected(true);
	}
}

//##################//
// IDs              //
//##################//

const QString
TileSystem::getUniqueId(const QString &suggestion, QString &name)
{
    // Try suggestion //
    //
    if (!suggestion.isNull() && !suggestion.isEmpty())
    {
        if (!tileIds_.contains(suggestion))
        {
            tileIds_.append(suggestion);
            return suggestion;
        }
    }

    // Create new one //
    //

    int index = 0;
    while ((index < tileIds_.size()) && tileIds_.contains(QString("%1").arg(index)))
    {
        index++;
    }

    QString id = QString("%1").arg(index);
    name = "Tile" + id;
    tileIds_.append(id);
    return id;
}

//##################//
// ProjectData      //
//##################//

void
TileSystem::setParentProjectData(ProjectData *projectData)
{
    parentProjectData_ = projectData;
    setParentElement(projectData);
    addTileSystemChanges(TileSystem::CTS_ProjectDataChanged);
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
TileSystem::notificationDone()
{
    tileSystemChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
TileSystem::addTileSystemChanges(int changes)
{
    if (changes)
    {
        tileSystemChanges_ |= changes;
        notifyObservers();
    }
}

//##################//
// Visitor Pattern  //
//##################//

/*! \brief Accepts a visitor.
*
* With autotraverse: visitor will be send to roads, fiddleyards, etc.
* Without: accepts visitor as 'this'.
*/
void
TileSystem::accept(Visitor *visitor)
{
    visitor->visit(this);
}

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
TileSystem::acceptForChildNodes(Visitor *visitor)
{
    foreach (Tile *child, tiles_)
    {
        child->accept(visitor);
    }
}

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
TileSystem::acceptForTiles(Visitor *visitor)
{
    foreach (Tile *child, tiles_)
    {
        child->accept(visitor);
    }
}
