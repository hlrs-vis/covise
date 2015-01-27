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

#include "ruler.hpp"

// Utils //
//
#include "math.h"

// Qt //
//
#include <QGraphicsScene>
#include <QPen>
#include <QDebug>

#define BIGSTEPS 100.0
#define MARGIN 10.0

Ruler::Ruler(Qt::Orientation orientation, QGraphicsItem *parent)
    : QGraphicsPathItem(parent)
    , orientation_(orientation)
{
    // Pen //
    //
    QPen pen;
    pen.setWidth(1);
    pen.setCosmetic(true);
    setPen(pen);
}

void
Ruler::updateRect(const QRectF &rect, double scaleX, double scaleY)
{
    // TODO: set min size for scene view widget

    //	qDebug() << rect;

    // Clean up //
    //
    foreach (QGraphicsTextItem *item, rulerTextItems_)
    {
        scene()->removeItem(item);
        delete item;
    }
    rulerTextItems_.clear();

    // Step size //
    //
    double bigStepLength = BIGSTEPS / scaleX;
    if (orientation_ == Qt::Vertical)
    {
        bigStepLength = BIGSTEPS / scaleY;
    }

    double smallStepLength = bigStepLength / 5.0;
    smallStepLength = abs(int(smallStepLength + 0.5));
    //	qDebug() << "smallStepLength: " << smallStepLength;

    if (smallStepLength < 0.0)
    {
        if (smallStepLength > -1.0)
        {
            smallStepLength = -1.0;
        }
    }
    else
    {
        if (smallStepLength < 1.0)
        {
            smallStepLength = 1.0;
        }
    }

    if (fabs(smallStepLength) > 25.0)
    {
        smallStepLength = int((smallStepLength / 5.0) + 0.5) * 5.0; // multiple of 25
    }

    bigStepLength = 5.0 * smallStepLength;
    //	qDebug() << "bigStepLength: " << bigStepLength;
    //	qDebug() << "smallStepLength: " << smallStepLength;

    double marginX = MARGIN / scaleX;
    double marginY = MARGIN / scaleY;

    // Path //
    //
    QPainterPath path;

    if (orientation_ == Qt::Horizontal)
    {
        const double leftEnd = rect.x() + marginX;
        const double rightEnd = rect.x() + rect.width() - marginX;
        const double y = rect.y() + marginY;

        // Big //
        //
        double x = ceil(leftEnd / bigStepLength) * bigStepLength; // first
        while (x <= rightEnd)
        {
            path.moveTo(x, y);
            path.lineTo(x, y + 8.0 / scaleY);

            QGraphicsTextItem *text = new QGraphicsTextItem(QString("%1").arg(x), this);
            text->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
            text->setPos(x - 4.0 / scaleX, y + 6.0 / scaleY);
            rulerTextItems_.append(text);

            x += bigStepLength;
        }

        // Small //
        //
        x = ceil(leftEnd / smallStepLength) * smallStepLength;
        while (x <= rightEnd)
        {
            path.moveTo(x, y);
            path.lineTo(x, y + 4.0 / scaleY);
            x += smallStepLength;
        }
    }
    else
    {
        double upperEnd = rect.y() + marginY;
        double lowerEnd = rect.y() + rect.height() - marginY;
        const double x = rect.x() + marginX;
        //qDebug() << "upperEnd: " << upperEnd;
        //qDebug() << "lowerEnd: " << lowerEnd;

        if (upperEnd > lowerEnd)
        {
            double tmp = upperEnd;
            upperEnd = lowerEnd;
            lowerEnd = tmp;
        }

        //qDebug() << "x: " << x;
        // Big //
        //
        double y = ceil(upperEnd / bigStepLength) * bigStepLength; // first
        //qDebug() << "y: " << y;
        while (y <= lowerEnd)
        {
            path.moveTo(x, y);
            path.lineTo(x + 8.0 / scaleX, y);

            QGraphicsTextItem *text = new QGraphicsTextItem(QString("%1").arg(y), this);
            text->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
            text->setPos(x + 6.0 / scaleX, y - 4.0 / scaleY);
            rulerTextItems_.append(text);

            y += bigStepLength;
        }

        // Small //
        //
        y = ceil(upperEnd / smallStepLength) * smallStepLength;
        while (y <= lowerEnd)
        {
            path.moveTo(x, y);
            path.lineTo(x + 4.0 / scaleX, y);
            y += smallStepLength;
        }
    }

    setPath(path);
}
