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

#include "crossfalltreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/sections/crossfallsection.hpp"

// Tree //
//
#include "roadtreeitem.hpp"

CrossfallSectionTreeItem::CrossfallSectionTreeItem(RoadTreeItem *parent, CrossfallSection *section, QTreeWidgetItem *fosterParent)
    : SectionTreeItem(parent, section, fosterParent)
    , crossfallSection_(section)
{
    init();
}

CrossfallSectionTreeItem::~CrossfallSectionTreeItem()
{
}

void
CrossfallSectionTreeItem::init()
{
    updateName();
}

void
CrossfallSectionTreeItem::updateName()
{
    QString text(tr("CrossfallSection (%1)").arg(crossfallSection_->getDegree()));
    text.append(QString(" %1").arg(crossfallSection_->getSStart()));
    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
CrossfallSectionTreeItem::updateObserver()
{

    // Parent //
    //
    SectionTreeItem::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // CrossfallSection //
    //
    int changes = crossfallSection_->getCrossfallSectionChanges();

    if (changes & CrossfallSection::CCF_ParameterChange)
    {
        updateName();
    }

    if (changes & CrossfallSection::CCF_SideChange)
    {
        updateName();
    }
}
