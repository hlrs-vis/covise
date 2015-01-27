/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/17/2010
**
**************************************************************************/

#include "pedestriantreeitem.hpp"

// Data //
//
#include "src/data/pedestriansystem/pedestrian.hpp"

// Tree //
//
#include "pedestriangrouptreeitem.hpp"

PedestrianTreeItem::PedestrianTreeItem(PedestrianGroupTreeItem *parent, Pedestrian *pedestrian, QTreeWidgetItem *fosterParent)
    : ProjectTreeItem(parent, pedestrian, fosterParent)
    , pedestrianGroupTreeItem_(parent)
    , pedestrian_(pedestrian)
{
    init();
}

PedestrianTreeItem::~PedestrianTreeItem()
{
}

void
PedestrianTreeItem::init()
{
    updateText();
}

void
PedestrianTreeItem::updateText()
{
    // Text //
    //
    QString text;
    if (pedestrian_->isDefault())
    {
        text = "<Default>";
    }
    else if (pedestrian_->isTemplate())
    {
        text = "<Template: ";
        text.append(pedestrian_->getId());
        text.append(">");
    }
    else
    {
        text = pedestrian_->getId();
        text.append(": ");
        text.append(pedestrian_->getName());
    }
    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
PedestrianTreeItem::updateObserver()
{
    // Parent //
    //
    ProjectTreeItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // TODO: Pedestrian SUBJECT stuff

    // Get change flags //
    //
    //	int changes = pedestrian_->get();
    //	if((changes & Pedestrian::CSM_Id)
    //		|| (changes & Pedestrian::CSM_Filename)
    //	)
    //	{
    //		updateText();
    //	}
}
