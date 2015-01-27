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

#ifndef JUNCTIONCOMPONENTITEM_HPP
#define JUNCTIONCOMPONENTITEM_HPP

#include "src/graph/items/graphelement.hpp"

class JunctionRoadItem;

class TrackComponent;

class JunctionEditor;

class JunctionComponentItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit JunctionComponentItem(JunctionRoadItem *parentJunctionRoadItem, TrackComponent *trackComponent);
    explicit JunctionComponentItem(JunctionComponentItem *parentJunctionComponentItem, TrackComponent *trackComponent);
    virtual ~JunctionComponentItem();

    JunctionRoadItem *getParentJunctionRoadItem() const
    {
        return parentJunctionRoadItem_;
    }
    JunctionComponentItem *getParentJunctionComponentItem() const
    {
        return parentJunctionComponentItem_;
    }

    // JunctionEditor //
    //
    JunctionEditor *getJunctionEditor() const
    {
        return junctionEditor_;
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
    JunctionComponentItem(); /* not allowed */
    JunctionComponentItem(const JunctionComponentItem &); /* not allowed */
    JunctionComponentItem &operator=(const JunctionComponentItem &); /* not allowed */

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

    //################//
    // EVENTS         //
    //################//

protected:
    //################//
    // PROPERTIES     //
    //################//

private:
    // JunctionEditor //
    //
    JunctionEditor *junctionEditor_;

    // Parent //
    //
    JunctionRoadItem *parentJunctionRoadItem_;
    JunctionComponentItem *parentJunctionComponentItem_;

    // Junction //
    //
    TrackComponent *trackComponent_;
};

#endif // JUNCTIONCOMPONENTITEM_HPP
