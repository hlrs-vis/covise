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

#include "sliderhandle.hpp"

#include "slidermovehandle.hpp"

//################//
// CONSTRUCTOR    //
//################//

SliderHandle::SliderHandle(QGraphicsItem *parent)
    : Handle(parent)
{
    // Path //
    //
    if (!SliderHandle::pathTemplate_)
    {
        createPath();
    }
    setPath(*SliderHandle::pathTemplate_);

    setFlag(QGraphicsItem::ItemIsMovable, false);
    setFlag(QGraphicsItem::ItemIsSelectable, false);
    setFlag(QGraphicsItem::ItemSendsGeometryChanges, false);

    moveHandle_ = new SliderMoveHandle(this);
}

SliderHandle::~SliderHandle()
{
}

//################//
// EVENTS         //
//################//

//################//
// SLOTS          //
//################//

//void
//	SliderHandle
//	::moveHandlePositionChange(const QPointF & pos)
//{
////	qDebug() << pos.x() << " " << pos.y();
////	moveHandle_->setPos(pos.x(), 0.0);

//}

//################//
// STATIC         //
//################//

/*! \brief Initialize the path once.
*
*/
void
SliderHandle::createPath()
{
    static QPainterPath pathTemplate; // Static, so the destructor kills it on application shutdown.

    double size = 30.0;
    pathTemplate.moveTo(0.0, 0.0);
    pathTemplate.lineTo(size, 0.0);

    pathTemplate_ = &pathTemplate;
}

// Initialize to NULL //
//
QPainterPath *SliderHandle::pathTemplate_ = NULL;
