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

#ifndef BRIDGETREEITEM_HPP
#define BRIDGETREEITEM_HPP

#include "sectiontreeitem.hpp"

class RoadTreeItem;
class Bridge;

class BridgeTreeItem : public SectionTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit BridgeTreeItem(RoadTreeItem *parent, Bridge *section, QTreeWidgetItem *fosterParent);
    virtual ~BridgeTreeItem();

    // BridgeBridge //
    //
    Bridge *getBridge() const
    {
        return bridge_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
    virtual void updateName();

private:
    BridgeTreeItem(); /* not allowed */
    BridgeTreeItem(const BridgeTreeItem &); /* not allowed */
    BridgeTreeItem &operator=(const BridgeTreeItem &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // PROPERTIES     //
    //################//

private:
    // Road //
    //
    Bridge *bridge_;
};

#endif // OBJECTTREEITEM_HPP
