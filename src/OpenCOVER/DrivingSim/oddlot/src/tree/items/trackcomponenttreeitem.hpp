/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/26/2010
**
**************************************************************************/

#ifndef TRACKCOMPONENTTREEITEM_HPP
#define TRACKCOMPONENTTREEITEM_HPP

#include "projecttreeitem.hpp"

class RoadTreeItem;
class TrackComponent;

class TrackComponentTreeItem : public ProjectTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackComponentTreeItem(RoadTreeItem *parent, TrackComponent *component, QTreeWidgetItem *fosterParent);
    explicit TrackComponentTreeItem(TrackComponentTreeItem *parent, TrackComponent *component, QTreeWidgetItem *fosterParent);
    virtual ~TrackComponentTreeItem();

    // TrackComponent //
    //
    TrackComponent *getTrackComponent() const
    {
        return trackComponent_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
    //	virtual void			updateName() = 0;

private:
    TrackComponentTreeItem(); /* not allowed */
    TrackComponentTreeItem(const TrackComponentTreeItem &); /* not allowed */
    TrackComponentTreeItem &operator=(const TrackComponentTreeItem &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // PROPERTIES     //
    //################//

private:
    // Parent //
    //
    RoadTreeItem *parentRoadTreeItem_;
    TrackComponentTreeItem *parentTrackComponentTreeItem_;

    // TrackComponent //
    //
    TrackComponent *trackComponent_;
};

#endif // TRACKCOMPONENTTREEITEM_HPP
