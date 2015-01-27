/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/7/2010
**
**************************************************************************/

#include "scenerysystemtreeitem.hpp"

// Data //
//
#include "src/data/scenerysystem/scenerysystem.hpp"
#include "src/data/scenerysystem/scenerymap.hpp"
#include "src/data/scenerysystem/heightmap.hpp"

// Tree //
//
#include "scenerymaptreeitem.hpp"

ScenerySystemTreeItem::ScenerySystemTreeItem(ProjectTree *projectTree, ScenerySystem *scenerySystem, QTreeWidgetItem *rootItem)
    : ProjectTreeItem(NULL, scenerySystem, rootItem)
    , // no direct parent, rootItem is foster parent
    projectTree_(projectTree)
    , scenerySystem_(scenerySystem)
    , rootItem_(rootItem)
{
    init();
}

ScenerySystemTreeItem::~ScenerySystemTreeItem()
{
}

void
ScenerySystemTreeItem::init()
{
    // Text //
    //
    setText(0, tr("ScenerySystem"));

    // Maps //
    //
    aerialMapsItem_ = new QTreeWidgetItem(this);
    aerialMapsItem_->setText(0, tr("Aerial Maps"));

    heightMapsItem_ = new QTreeWidgetItem(this);
    heightMapsItem_->setText(0, tr("Height Maps"));

    foreach (SceneryMap *map, scenerySystem_->getSceneryMaps())
    {
        new SceneryMapTreeItem(this, map, aerialMapsItem_);
    }
    foreach (Heightmap *map, scenerySystem_->getHeightmaps())
    {
        new SceneryMapTreeItem(this, map, heightMapsItem_);
    }
}

//##################//
// Observer Pattern //
//##################//

void
ScenerySystemTreeItem::updateObserver()
{
    // Parent //
    //
    ProjectTreeItem::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // ScenerySystem //
    //
    int changes = getScenerySystem()->getScenerySystemChanges();

    if (changes & ScenerySystem::CSC_MapChanged)
    {
        // A SceneryMap has been added (or deleted - but that will be handled by the item itself).
        //
        foreach (SceneryMap *map, scenerySystem_->getSceneryMaps())
        {
            if ((map->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (map->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new SceneryMapTreeItem(this, map, aerialMapsItem_);
            }
        }
        foreach (Heightmap *map, scenerySystem_->getHeightmaps())
        {
            if ((map->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (map->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new SceneryMapTreeItem(this, map, heightMapsItem_);
            }
        }
    }
}
