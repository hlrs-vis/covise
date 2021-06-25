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

#ifndef LANEBORDERHANDLE_HPP
#define LANEBORDERHANDLE_HPP

#include "src/graph/items/handles/lanemovehandle.hpp"

class LaneEditor;
class LaneBorder;

#include <QGraphicsItem>

class LaneBorderHandle : public LaneMoveHandle<LaneBorder,LaneBorder>
{
	Q_OBJECT

		//################//
		// FUNCTIONS      //
		//################//

public:
	explicit LaneBorderHandle(LaneEditor *laneEditor, QGraphicsItem *parent);
	virtual ~LaneBorderHandle();


	//################//
	// SLOTS          //
	//################//

public slots :
	void removeCorner();
	void smoothCorner();

	//################//
	// EVENTS         //
	//################//

protected:
	virtual QVariant itemChange(GraphicsItemChange change, const QVariant &value);

private:
	LaneEditor *laneEditor_;

	// ContextMenu //
	//
	QAction *removeAction_;
	QAction *smoothAction_;
};


#endif // LANEBORDERHANDLE_HPP
