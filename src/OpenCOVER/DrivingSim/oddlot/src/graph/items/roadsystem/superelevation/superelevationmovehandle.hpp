/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   16.07.2010
**
**************************************************************************/

#ifndef SUPERELEVATIONMOVEHANDLE_HPP
#define SUPERELEVATIONMOVEHANDLE_HPP

#include "src/graph/items/handles/movehandle.hpp"

class SuperelevationEditor;
class SuperelevationSection;

class SuperelevationMoveHandle : public MoveHandle
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SuperelevationMoveHandle(SuperelevationEditor *superelevationEditor, QGraphicsItem *parent);
    virtual ~SuperelevationMoveHandle();

    void registerLowSlot(SuperelevationSection *superelevationSection);
    void registerHighSlot(SuperelevationSection *superelevationSection);

    SuperelevationSection *getLowSlot() const
    {
        return lowSlot_;
    }
    SuperelevationSection *getHighSlot() const
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
    SuperelevationMoveHandle(); /* not allowed */
    SuperelevationMoveHandle(const SuperelevationMoveHandle &); /* not allowed */
    SuperelevationMoveHandle &operator=(const SuperelevationMoveHandle &); /* not allowed */

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
    SuperelevationEditor *superelevationEditor_;

    SuperelevationSection *lowSlot_;
    SuperelevationSection *highSlot_;

    int posDOF_;

    QAction *removeAction_;
    QAction *smoothAction_;
};

#endif // SUPERELEVATIONMOVEHANDLE_HPP
