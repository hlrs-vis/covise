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

#ifndef JUNCTIONADDHANDLE_HPP
#define JUNCTIONHANDLE_HPP

#include "src/graph/items/handles/linkhandle.hpp"

class JunctionEditor;

class RSystemElementRoad;
class TrackComponent;

class JunctionAddHandle : public LinkHandle
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit JunctionAddHandle(JunctionEditor *junctionEditor, QGraphicsItem *parentItem, RSystemElementRoad *road, bool isStart);
    virtual ~JunctionAddHandle();

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
    JunctionAddHandle(); /* not allowed */
    JunctionAddHandle(const JunctionAddHandle &); /* not allowed */

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
    JunctionEditor *junctionEditor_;

    RSystemElementRoad *road_;
    bool isStart_;

    TrackComponent *track_;
};

#endif // TRACKADDHANDLE_HPP
