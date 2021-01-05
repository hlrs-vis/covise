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

#include "shapetreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/sections/shapesection.hpp"

// Tree //
//
#include "roadtreeitem.hpp"

ShapeSectionTreeItem::ShapeSectionTreeItem(RoadTreeItem *parent, ShapeSection *section, QTreeWidgetItem *fosterParent)
    : SectionTreeItem(parent, section, fosterParent)
    , shapeSection_(section)
{
    init();
}

ShapeSectionTreeItem::~ShapeSectionTreeItem()
{
}

void
ShapeSectionTreeItem::init()
{
    updateName();
}

void
ShapeSectionTreeItem::updateName()
{
    QString text(tr("ShapeSection (%1)").arg(shapeSection_->getSStart()));
    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
ShapeSectionTreeItem::updateObserver()
{

    // Parent //
    //
    SectionTreeItem::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // ShapeSection //
    //
    int changes = shapeSection_->getShapeSectionChanges();

    if (changes & ShapeSection::CSS_ParameterChange)
    {
        updateName();
    }

}
