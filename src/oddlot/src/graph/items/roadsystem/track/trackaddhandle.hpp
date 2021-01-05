/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10.05.2010
**
**************************************************************************/

#ifndef TRACKADDHANDLE_HPP
#define TRACKADDHANDLE_HPP

#include "src/graph/items/handles/linkhandle.hpp"

class TrackEditor;

class RSystemElementRoad;
class TrackComponent;

class TrackAddHandle : public LinkHandle
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackAddHandle(TrackEditor *trackEditor, QGraphicsItem *parentItem, RSystemElementRoad *road, bool isStart);
    virtual ~TrackAddHandle();

    void updateTransformation();
    void updateColor();

    RSystemElementRoad *getRoad() const
    {
        return road_;
    }
    bool isStart() const
    {
        return isStart_;
    }

    // Observer Pattern //
    //
    virtual void updateObserver();

protected:
private:
    TrackAddHandle(); /* not allowed */
    TrackAddHandle(const TrackAddHandle &); /* not allowed */

    //################//
    // EVENTS         //
    //################//

protected:
    virtual QVariant itemChange(GraphicsItemChange change, const QVariant &value);

    virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    TrackEditor *trackEditor_;

    RSystemElementRoad *road_;
    bool isStart_;

    TrackComponent *track_;
};

#endif // TRACKADDHANDLE_HPP
