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

#include "signaltreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/sections/signalobject.hpp"

// Tree //
//
#include "roadtreeitem.hpp"

SignalTreeItem::SignalTreeItem(RoadTreeItem *parent, Signal *signal, QTreeWidgetItem *fosterParent)
    : SectionTreeItem(parent, signal, fosterParent)
    , signal_(signal)
{
    init();
}

SignalTreeItem::~SignalTreeItem()
{
}

void
SignalTreeItem::init()
{
    updateName();
}

void
SignalTreeItem::updateName()
{
    QString text = signal_->getName();
    text.append(QString(" (%1").arg(signal_->getType()));
    text.append(QString(") %1").arg(signal_->getSStart()));
    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
SignalTreeItem::updateObserver()
{

    // Parent //
    //
    SectionTreeItem::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Signal //
    //
    int changes = signal_->getSignalChanges();

    if (changes & Signal::CEL_ParameterChange)
    {
        updateName();
    }
}
