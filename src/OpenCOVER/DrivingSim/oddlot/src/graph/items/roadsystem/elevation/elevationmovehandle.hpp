/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   25.06.2010
**
**************************************************************************/

#ifndef ELEVATIONMOVEHANDLE_HPP
#define ELEVATIONMOVEHANDLE_HPP

#include "src/graph/items/handles/movehandle.hpp"

class ElevationEditor;
class ElevationSection;

class ElevationMoveHandle : public MoveHandle
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ElevationMoveHandle(ElevationEditor *elevationEditor, QGraphicsItem *parent);
    virtual ~ElevationMoveHandle();

    void registerLowSlot(ElevationSection *elevationSection);
    void registerHighSlot(ElevationSection *elevationSection);

    ElevationSection *getLowSlot() const
    {
        return lowSlot_;
    }
    ElevationSection *getHighSlot() const
    {
        return highSlot_;
    }

    int getPosDOF() const
    {
        return posDOF_;
    }
    void setDOF(int dof);

    // Observer Pattern //
    //
    virtual void updateObserver();

protected:
private:
    ElevationMoveHandle(); /* not allowed */
    ElevationMoveHandle(const ElevationMoveHandle &); /* not allowed */
    ElevationMoveHandle &operator=(const ElevationMoveHandle &); /* not allowed */

    void updateColor();

    //################//
    // SLOTS          //
    //################//

public slots:
    void removeCorner();
    void smoothCorner();

    //################//
    // EVENTS         //
    //################//

protected:
    virtual QVariant itemChange(GraphicsItemChange change, const QVariant &value);

    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
    virtual void contextMenuEvent(QGraphicsSceneContextMenuEvent *event);

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    ElevationEditor *elevationEditor_;

    ElevationSection *lowSlot_;
    ElevationSection *highSlot_;

    int posDOF_;

    // ContextMenu //
    //
    QAction *removeAction_;
    QAction *smoothAction_;
};

#endif // ELEVATIONMOVEHANDLE_HPP
