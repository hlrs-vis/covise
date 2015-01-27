/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/19/2010
**
**************************************************************************/

#include "circularhandle.hpp"

//################//
// CONSTRUCTOR    //
//################//

CircularHandle::CircularHandle(QGraphicsItem *parent)
    : Handle(parent) /*,
		passSelectionToParent_(false)*/
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

    // Path //
    //
    QPainterPath path;
    path.addEllipse(QPointF(), 4.0, 4.0);
    setPath(path);
}

CircularHandle::~CircularHandle()
{
}

//void
//	CircularHandle
//	::setPassSelectionToParent(bool passSelectionToParent)
//{
//	setFlag(QGraphicsItem::ItemIsSelectable, true);
//	passSelectionToParent_ = passSelectionToParent;
//}

//################//
// EVENTS         //
//################//

//QVariant
//	CircularHandle
//	::itemChange(GraphicsItemChange change, const QVariant & value)
//{
//	if(change == QGraphicsItem::ItemSelectedChange)
//	{
//		if(passSelectionToParent_)
//		{
//			parentItem()->setSelected(value.toBool());
//		}
//	}

//	return Handle::itemChange(change, value);
//}
