/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/11/2010
**
**************************************************************************/

#include "controllertreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementcontroller.hpp"

// Tree //
//
#include "roadsystemtreeitem.hpp"

ControllerTreeItem::ControllerTreeItem(RoadSystemTreeItem *parent, RSystemElementController *controller, QTreeWidgetItem *fosterParent)
    : ProjectTreeItem(parent, controller, fosterParent)
    , roadSystemTreeItem_(parent)
    , controller_(controller)
{
    init();
}

ControllerTreeItem::~ControllerTreeItem()
{
}

void
ControllerTreeItem::init()
{
    // Text //
    //
    updateName();

}

void
ControllerTreeItem::updateName()
{
    QString text = controller_->getID().speakingName();
    if (!controller_->getName().isEmpty())
    {
        text.append(" (");
        text.append(controller_->getName());
        text.append(")");
    }
    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
ControllerTreeItem::updateObserver()
{
    // Parent //
    //
    ProjectTreeItem::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // RSystemElement //
    //
    int changes = controller_->getRSystemElementChanges();

    if ((changes & RSystemElement::CRE_NameChange)
        || (changes & RSystemElement::CRE_IdChange))
    {
        updateName();
    }
}
