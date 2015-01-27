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

#ifndef RULER_HPP
#define RULER_HPP

#include <QGraphicsPathItem>

class Ruler : public QGraphicsPathItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit Ruler(Qt::Orientation orientation, QGraphicsItem *parent = NULL);
    virtual ~Ruler()
    { /* does nothing */
    }

    void updateRect(const QRectF &rect, double scaleX, double scaleY);

protected:
private:
    Ruler(); /* not allowed */
    Ruler(const Ruler &); /* not allowed */
    Ruler &operator=(const Ruler &); /* not allowed */

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    Qt::Orientation orientation_;

    QList<QGraphicsTextItem *> rulerTextItems_;
};

#endif // RULER_HPP
