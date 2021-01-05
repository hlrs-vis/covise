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

#include "scenerymaptreeitem.hpp"

// Data //
//
#include "src/data/scenerysystem/scenerymap.hpp"

// Tree //
//
#include "scenerysystemtreeitem.hpp"

SceneryMapTreeItem::SceneryMapTreeItem(ScenerySystemTreeItem *parent, SceneryMap *map, QTreeWidgetItem *fosterParent)
    : ProjectTreeItem(parent, map, fosterParent)
    , scenerySystemTreeItem_(parent)
    , map_(map)
{
    init();
}

SceneryMapTreeItem::~SceneryMapTreeItem()
{
}

void
SceneryMapTreeItem::init()
{
    updateText();
}

void
SceneryMapTreeItem::updateText()
{
    // Text //
    //
    QString text = map_->getId();
    if (!map_->getFilename().isEmpty())
    {
        text.append(" (");
        text.append(map_->getFilename());
        text.append(")");
    }
    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
SceneryMapTreeItem::updateObserver()
{
    // Parent //
    //
    ProjectTreeItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Get change flags //
    //
    int changes = map_->getSceneryMapChanges();
    if ((changes & SceneryMap::CSM_Id)
        || (changes & SceneryMap::CSM_Filename))
    {
        updateText();
    }
}
