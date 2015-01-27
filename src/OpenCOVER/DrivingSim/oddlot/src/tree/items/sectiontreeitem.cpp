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

#include "sectiontreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/sections/roadsection.hpp"

// Tree //
//
#include "roadtreeitem.hpp"

SectionTreeItem::SectionTreeItem(RoadTreeItem *parent, RoadSection *roadSection, QTreeWidgetItem *fosterParent)
    : ProjectTreeItem(parent, roadSection, fosterParent)
    , roadTreeItem_(parent)
    , roadSection_(roadSection)
{
    init();
}

SectionTreeItem::~SectionTreeItem()
{
}

void
SectionTreeItem::init()
{
    //	updateName();
}

//##################//
// Observer Pattern //
//##################//

void
SectionTreeItem::updateObserver()
{

    // Parent //
    //
    ProjectTreeItem::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // RoadSection //
    //
    int changes = roadSection_->getRoadSectionChanges();
    if (changes & RoadSection::CRS_SChange)
    {
        // Change of the road coordinate s //
        //
        updateName();
    }
    //	if(changes & RoadSection::CRS_LengthChange)
    //	{
    //		// Change of the length of the section //
    //		//
    //	}
}
