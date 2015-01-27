/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   14.07.2010
**
**************************************************************************/

#ifndef CROSSFALLMOVEHANDLE_HPP
#define CROSSFALLMOVEHANDLE_HPP

#include "src/graph/items/handles/movehandle.hpp"

class CrossfallEditor;
class CrossfallSection;

class CrossfallMoveHandle : public MoveHandle
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit CrossfallMoveHandle(CrossfallEditor *crossfallEditor, QGraphicsItem *parent);
    virtual ~CrossfallMoveHandle();

    void registerLowSlot(CrossfallSection *crossfallSection);
    void registerHighSlot(CrossfallSection *crossfallSection);

    CrossfallSection *getLowSlot() const
    {
        return lowSlot_;
    }
    CrossfallSection *getHighSlot() const
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

private:
    CrossfallMoveHandle(); /* not allowed */
    CrossfallMoveHandle(const CrossfallMoveHandle &); /* not allowed */
    CrossfallMoveHandle &operator=(const CrossfallMoveHandle &); /* not allowed */

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

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    CrossfallEditor *crossfallEditor_;

    CrossfallSection *lowSlot_;
    CrossfallSection *highSlot_;

    int posDOF_;

    QAction *removeAction_;
    QAction *smoothAction_;
};

#endif // CROSSFALLMOVEHANDLE_HPP
