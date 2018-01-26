/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   12.03.2010
**
**************************************************************************/

#include "svgitem.hpp"

// Graph //
//
#include "src/graph/items/oscsystem/oscitem.hpp"

// Qt //
//
#include "QtSvg/qsvgrenderer.h"


SVGItem::SVGItem(OSCItem *oscItem, std::string file)
    : QGraphicsSvgItem(oscItem)
	, parentItem_(oscItem)
	, file_(file)
{

    init();
}

SVGItem::~SVGItem()
{
}



void
SVGItem::init()
{
	
	QSvgRenderer *renderer = new QSvgRenderer(QLatin1String(file_.c_str()));
	setFlag(ItemIsSelectable);
	setFlag(ItemIsFocusable);
	setFlag(ItemIsMovable);
	setSharedRenderer(renderer);
}






//################//
// SLOTS          //
//################//



//################//
// EVENTS         //
//################//



void
SVGItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	QGraphicsItem::mousePressEvent(event);
	parentItem_->mousePressEvent(event);
}

void
SVGItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{	
	QGraphicsItem::mouseMoveEvent(event);
	parentItem_->mouseMoveEvent(event);
}

void
SVGItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
	QGraphicsItem::mouseReleaseEvent(event);
	parentItem_->mouseReleaseEvent(event);
} 

void
SVGItem::keyPressEvent(QKeyEvent *event)
{
	QGraphicsItem::keyPressEvent(event);
	parentItem_->keyPressEvent(event);
}

void
SVGItem::keyReleaseEvent(QKeyEvent *event)
{
	QGraphicsItem::keyReleaseEvent(event);
	parentItem_->keyReleaseEvent(event);
}

