/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   27.05.2010
**
**************************************************************************/

#ifndef HANDLE_HPP
#define HANDLE_HPP

#include <QObject>
#include <QGraphicsPathItem>
#include <QPainterPath>
#include "src/data/observer.hpp"

#include <QMenu>
#include <QAction>
#include <QGraphicsSceneContextMenuEvent>
#include <QGraphicsSceneMouseEvent>
#include <QBrush>
#include <QPen>
#include <QCursor>

// Utils //
//
#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"
#include "math.h"

class Handle : public QObject, public QGraphicsPathItem, public Observer
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit Handle(QGraphicsItem *parent, bool flip = true);
    virtual ~Handle();

    // ContextMenu //
    //
    QMenu *getContextMenu()
    {
        return contextMenu_;
    }

private:
    Handle(); /* not allowed */
    Handle(const Handle &); /* not allowed */
    Handle &operator=(const Handle &); /* not allowed */

    //################//
    // EVENTS         //
    //################//

protected:
    virtual void contextMenuEvent(QGraphicsSceneContextMenuEvent *event);

    //################//
    // PROPERTIES     //
    //################//

private:
    // ContextMenu //
    //
    QMenu *contextMenu_;
};

#endif // HANDLE_HPP
