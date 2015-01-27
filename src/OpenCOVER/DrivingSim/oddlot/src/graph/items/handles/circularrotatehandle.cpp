/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   09.07.2010
**
**************************************************************************/

#include "circularrotatehandle.hpp"

//################//
// CONSTRUCTOR    //
//################//

CircularRotateHandle::CircularRotateHandle(QGraphicsItem *parent)
    : Handle(parent)
    , angle_(0.0)
{
    // Pen //
    //
    QPen pen;
    pen.setWidth(1);
    pen.setCosmetic(true); // constant size independent of scaling
    pen.setColor(ODD::instance()->colors()->darkGreen());
    setPen(pen);

    // Brush //
    //
    setBrush(QBrush(ODD::instance()->colors()->brightGreen()));
}

CircularRotateHandle::~CircularRotateHandle()
{
}

/*! \brief Recalculates the path.
*
*/
void
CircularRotateHandle::setMousePos(const QPointF &mousePoint)
{
    // Size //
    //
    double radius = 16.0;

    // Calculations //
    //
    QPointF deltaPos = mousePoint - pos();

    double angleRad = atan2(deltaPos.y(), deltaPos.x());
    angle_ = angleRad * 360.0 / (2.0 * M_PI);

    if (angle_ < -180.0)
    {
        angle_ += 360.0;
    }

    // Path //
    //
    QPainterPath path;
    //path.moveTo(0.0, 0.0);
    path.moveTo(-2.0 * radius, 0.0);
    path.lineTo(2.0 * radius, 0.0);

    //path.moveTo(0.0, 0.0);
    path.moveTo(-2.0 * cos(angleRad) * radius, -2.0 * sin(angleRad) * radius);
    path.lineTo(2.0 * cos(angleRad) * radius, 2.0 * sin(angleRad) * radius);

    path.moveTo(0.0, 0.0);
    path.lineTo(radius, 0.0);
    path.arcTo(-radius, -radius, 2.0 * radius, 2.0 * radius, 0.0, -angle_);
    path.lineTo(0.0, 0.0);

    setPath(path);
}
