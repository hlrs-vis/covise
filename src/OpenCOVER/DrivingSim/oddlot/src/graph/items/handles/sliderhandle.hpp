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

#ifndef SLIDERHANDLE_HPP
#define SLIDERHANDLE_HPP

#include "handle.hpp"

class SliderMoveHandle;

class SliderHandle : public Handle
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SliderHandle(QGraphicsItem *parent);
    virtual ~SliderHandle();

protected:
    // Move Handle //
    //
    SliderMoveHandle *getMoveHandle() const
    {
        return moveHandle_;
    }

private:
    SliderHandle(); /* not allowed */
    SliderHandle(const SliderHandle &); /* not allowed */
    SliderHandle &operator=(const SliderHandle &); /* not allowed */

    //################//
    // EVENTS         //
    //################//

protected:
    //################//
    // SLOTS          //
    //################//

public slots:
    //	virtual void			moveHandlePositionChange(const QPointF & pos);

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    //################//
    // STATIC         //
    //################//

private:
    // Path Template //
    //
    // this path will be shared by all handles
    static void createPath();
    static QPainterPath *pathTemplate_;

    // Move Handle //
    //
    SliderMoveHandle *moveHandle_;
};

#endif // SLIDERHANDLE_HPP
