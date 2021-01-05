/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   30.03.2010
**
**************************************************************************/

#ifndef ROADSYSTEMITEM_HPP
#define ROADSYSTEMITEM_HPP

#include "src/graph/items/graphelement.hpp"
#include "src/data/roadsystem/odrID.hpp"

class RoadSystem;

class RoadItem;

class SignalEditor;

class RoadSystemItem : public GraphElement
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadSystemItem(TopviewGraph *topviewGraph, RoadSystem *roadSystem);
    explicit RoadSystemItem(ProfileGraph *profileGraph, RoadSystem *roadSystem);
    virtual ~RoadSystemItem();

    // Graph //
    //
    virtual TopviewGraph *getTopviewGraph() const
    {
        return topviewGraph_;
    }
    virtual ProfileGraph *getProfileGraph() const
    {
        return profileGraph_;
    }

    // Roads //
    //
    void appendRoadItem(RoadItem *roadItem);
    bool removeRoadItem(RoadItem *roadItem);
    RoadItem *getRoadItem(const odrID &id) const
    {
        return roadItems_.value(id.getID(), NULL);
    }
    QMap<int32_t, RoadItem *> getRoadItems() const
    {
        return roadItems_;
    }

    // RoadSystem //
    //
    RoadSystem *getRoadSystem() const
    {
        return roadSystem_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest()
    {
        return false;
    };

protected:
private:
    RoadSystemItem(); /* not allowed */
    RoadSystemItem(const RoadSystemItem &); /* not allowed */
    RoadSystemItem &operator=(const RoadSystemItem &); /* not allowed */

    void init();

    //################//
    // PROPERTIES     //
    //################//

private:
    // Graph //
    //
    TopviewGraph *topviewGraph_;
    ProfileGraph *profileGraph_;

    // RoadSystem //
    //
    RoadSystem *roadSystem_;

    // Roads //
    //
    QMap<int32_t, RoadItem *> roadItems_;
};

#endif // ROADSYSTEMITEM_HPP
