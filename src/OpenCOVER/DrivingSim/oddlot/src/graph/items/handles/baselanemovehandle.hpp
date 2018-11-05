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

#ifndef BASELANEMOVEHANDLE_HPP
#define BASELANEMOVEHANDLE_HPP


#include "src/graph/items/handles/movehandle.hpp"

class LaneEditor;
class LaneWidth;
class TextHandle;



class BaseLaneMoveHandle : public MoveHandle
{
	Q_OBJECT

	//################//
	// FUNCTIONS      //
	//################//

public:
	explicit BaseLaneMoveHandle(LaneEditor *laneEditor, QGraphicsItem *parent);		
	virtual ~BaseLaneMoveHandle();

	// Get low and high slot of derived lanemovehandle //
	//
	virtual LaneWidth *getLowSlot() = 0;
	virtual LaneWidth *getHighSlot() = 0;


private:
	BaseLaneMoveHandle(); /* not allowed */
	BaseLaneMoveHandle(const BaseLaneMoveHandle &); /* not allowed */
	BaseLaneMoveHandle &operator=(const BaseLaneMoveHandle &); /* not allowed */

protected:
	virtual const QString getText() = 0;

	//################//
	// SLOTS          //
	//################//

	public slots :
	virtual void removeCorner() = 0;
	virtual void smoothCorner() = 0;
	virtual void corner() = 0;

	//################//
	// EVENTS         //
	//################//

protected:
	virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
	virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
	virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
	virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
	virtual void contextMenuEvent(QGraphicsSceneContextMenuEvent *event);



	//################//
	// PROPERTIES     //
	//################//

protected:

private:
	LaneEditor *laneEditor_;

	QPointF pressPos_;

	// ContextMenu //
	//
	QAction *removeAction_;
	QAction *smoothAction_;
	QAction *cornerAction_;

	TextHandle *widthTextItem_;

};

#endif // BASELANEMOVEHANDLE_HPP
