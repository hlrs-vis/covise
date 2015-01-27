/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   22.04.2010
**
**************************************************************************/

#ifndef TRACKCOMPONENTITEM_HPP
#define TRACKCOMPONENTITEM_HPP

#include "src/graph/items/graphelement.hpp"

class TrackRoadItem;
class TrackComponentItem;

class TrackComponent;

class TrackEditor;

class TrackComponentItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackComponentItem(TrackRoadItem *parentTrackRoadItem, TrackComponent *trackComponent);
    explicit TrackComponentItem(TrackComponentItem *parentTrackComponentItem, TrackComponent *trackComponent);
    virtual ~TrackComponentItem();

    TrackRoadItem *getParentTrackRoadItem() const
    {
        return parentTrackRoadItem_;
    }
    TrackComponentItem *getParentTrackComponentItem() const
    {
        return parentTrackComponentItem_;
    }

    // TrackEditor //
    //
    TrackEditor *getTrackEditor() const
    {
        return trackEditor_;
    }

    // Graphics //
    //
    virtual void updateColor() = 0;
    virtual void createPath() = 0;

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest();

private:
    TrackComponentItem(); /* not allowed */
    TrackComponentItem(const TrackComponentItem &); /* not allowed */
    TrackComponentItem &operator=(const TrackComponentItem &); /* not allowed */

    void init();

    //################//
    // SLOTS          //
    //################//

public slots:
    virtual void ungroupComposite();

    void hideParentRoad();
    void hideParentTrackComponent();
    bool removeSection();
    bool removeParentRoad();
    void morphIntoPoly3();
    void addToCurrentTile();

    //################//
    // EVENTS         //
    //################//

protected:
    //################//
    // PROPERTIES     //
    //################//

private:
    // TrackEditor //
    //
    TrackEditor *trackEditor_;

    // Parent //
    //
    TrackRoadItem *parentTrackRoadItem_;
    TrackComponentItem *parentTrackComponentItem_;

    // Track //
    //
    TrackComponent *trackComponent_;
};

#endif // TRACKCOMPONENTITEM_HPP
