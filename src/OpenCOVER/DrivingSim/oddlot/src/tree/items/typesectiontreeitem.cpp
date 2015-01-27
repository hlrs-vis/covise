/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/13/2010
**
**************************************************************************/

#include "typesectiontreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/sections/typesection.hpp"

// Tree //
//
#include "roadtreeitem.hpp"

TypeSectionTreeItem::TypeSectionTreeItem(RoadTreeItem *parent, TypeSection *section, QTreeWidgetItem *fosterParent)
    : SectionTreeItem(parent, section, fosterParent)
    , typeSection_(section)
{
    init();
}

TypeSectionTreeItem::~TypeSectionTreeItem()
{
}

void
TypeSectionTreeItem::init()
{
    updateName();
}

void
TypeSectionTreeItem::updateName()
{
    QString text = TypeSection::parseRoadTypeBack(typeSection_->getRoadType());
    text.append(QString(" %1").arg(typeSection_->getSStart()));
    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
TypeSectionTreeItem::updateObserver()
{

    // Parent //
    //
    SectionTreeItem::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // TypeSection //
    //
    int changes = typeSection_->getTypeSectionChanges();

    if (changes & TypeSection::CTS_TypeChange)
    {
        updateName();
    }
}
