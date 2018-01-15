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

#include "connectiontreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/junctionconnection.hpp"

// Tree //
//
#include "junctiontreeitem.hpp"

ConnectionTreeItem::ConnectionTreeItem(JunctionTreeItem *parent, JunctionConnection *connection, QTreeWidgetItem *fosterParent)
    : ProjectTreeItem(parent, connection, fosterParent)
    , junctionTreeItem_(parent)
    , connection_(connection)
{
    init();
}

ConnectionTreeItem::~ConnectionTreeItem()
{
}

void
ConnectionTreeItem::init()
{
    // Text //
    //
    updateName();

    // LaneLinks //
    //
    //	typesItem_ = new QTreeWidgetItem(this);
    //	typesItem_->setText(0, tr("types"));

    //	foreach(TypeSection * element, connection_->getTypeSections())
    //	{
    //		new TypeSectionTreeItem(this, element, typesItem_);
    //	}
}

void
ConnectionTreeItem::updateName()
{
    QString text = connection_->getId();
    text.append(QString(" (%1").arg(connection_->getIncomingRoad().speakingName()));
    text.append(QString(" > %1").arg(connection_->getConnectingRoad().speakingName()));
    text.append(JunctionConnection::parseContactPointBack(connection_->getContactPoint()));
    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
ConnectionTreeItem::updateObserver()
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
    int changes = connection_->getJunctionConnectionChanges();

    if ((changes))
    {
        updateName();
    }

    // LaneLinks //
    //
    //	if(changes & JunctionConnection::CJC_LaneLinkChanged)
    //	{
    //	}
}
