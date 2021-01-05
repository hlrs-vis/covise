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

#include "junctiontreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementjunction.hpp"
#include "src/data/roadsystem/junctionconnection.hpp"

// Tree //
//
#include "roadsystemtreeitem.hpp"
#include "connectiontreeitem.hpp"

JunctionTreeItem::JunctionTreeItem(RoadSystemTreeItem *parent, RSystemElementJunction *junction, QTreeWidgetItem *fosterParent)
    : ProjectTreeItem(parent, junction, fosterParent)
    , roadSystemTreeItem_(parent)
    , junction_(junction)
{
    init();
}

JunctionTreeItem::~JunctionTreeItem()
{
}

void
JunctionTreeItem::init()
{
    // Text //
    //
    updateName();

    // Connections //
    //
    foreach (JunctionConnection *element, junction_->getConnections())
    {
        new ConnectionTreeItem(this, element, NULL);
    }
}

void
JunctionTreeItem::updateName()
{
    QString text = junction_->getID().speakingName();
    if (!junction_->getName().isEmpty())
    {
        text.append(" (");
        text.append(junction_->getName());
        text.append(")");
    }
    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
JunctionTreeItem::updateObserver()
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
    int changes = junction_->getRSystemElementChanges();

    if ((changes & RSystemElement::CRE_NameChange)
        || (changes & RSystemElement::CRE_IdChange))
    {
        updateName();
    }

    // RSystemElementJunction //
    //
    changes = junction_->getJunctionChanges();

    if (changes & RSystemElementJunction::CJN_ConnectionChanged)
    {
        foreach (JunctionConnection *element, junction_->getConnections())
        {
            if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new ConnectionTreeItem(this, element, NULL);
            }
        }
    }
}
