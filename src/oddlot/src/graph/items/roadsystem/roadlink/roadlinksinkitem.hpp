/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /**************************************************************************
 ** ODD: OpenDRIVE Designer
 **   Frank Naegele (c) 2010
 **   <mail@f-naegele.de>
 **   11/19/2010
 **
 **************************************************************************/

#ifndef ROADLINKSINKITEM_HPP
#define ROADLINKSINKITEM_HPP

#include "src/graph/items/graphelement.hpp"

class RoadLinkEditor;
class RoadLinkRoadItem;
class RoadLinkHandle;

class RoadLinkSinkHandle;

class RoadLinkSinkItem : public GraphElement
{
    Q_OBJECT

        //################//
        // FUNCTIONS      //
        //################//

public:
    explicit RoadLinkSinkItem(RoadLinkRoadItem *parent, RoadLinkEditor *editor, bool isStart);
    virtual ~RoadLinkSinkItem();

    bool getIsStart() const
    {
        return isStart_;
    }
    RSystemElementRoad *getParentRoad() const
    {
        return parentRoad_;
    }
    RoadLinkSinkHandle *getRoadLinkSinkHandle() const
    {
        return sinkHandle_;
    }

    void setHandlesSelectable(bool selectable);

    // Graphics //
    //
    void updateColor();
    virtual void createPath();

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest()
    {
        return false;
    };

private:
    RoadLinkSinkItem(); /* not allowed */
    RoadLinkSinkItem(const RoadLinkSinkItem &); /* not allowed */
    RoadLinkSinkItem &operator=(const RoadLinkSinkItem &); /* not allowed */

    void init();

    //################//
    // SLOTS          //
    //################//

public slots:

public:
    //################//
    // PROPERTIES     //
    //################//

private:
    bool isStart_;

    RoadLinkEditor *editor_;

    // Road //
    //
    RoadLinkRoadItem *parentRoadItem_;
    RSystemElementRoad *parentRoad_;

    // Handle //
    //
    RoadLinkSinkHandle *sinkHandle_;
};

#endif // ROADLINKSINKITEM_HPP
