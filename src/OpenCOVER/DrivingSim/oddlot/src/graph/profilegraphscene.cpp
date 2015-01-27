/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   08.06.2010
**
**************************************************************************/

#include "profilegraphscene.hpp"
#include "qgraphicssceneevent.h"

ProfileGraphScene::ProfileGraphScene(const QRectF &sceneRect, QObject *parent)
    : QGraphicsScene(sceneRect, parent)
{
    init();
}

//################//
// FUNCTIONS      //
//################//

void
ProfileGraphScene::init()
{
    setBackgroundBrush(QBrush(QColor(238, 243, 238)));
    doDeselect_ = true;
}

void ProfileGraphScene::mousePressEvent(QGraphicsSceneMouseEvent *mevent)
{
    if (!doDeselect_)
    {
        //QGraphicsItem* pItemUnderMouse = itemAt(mevent->scenePos().x(), mevent->scenePos().y());
        QTransform transform;
        QGraphicsItem *pItemUnderMouse = itemAt(mevent->scenePos(), transform);

        if (!pItemUnderMouse)
            return;
    }
    QGraphicsScene::mousePressEvent(mevent);
}