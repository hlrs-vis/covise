/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QtGui>

#include "colormapListWidget.h"
#include "qglobal.h"

#include "qapplication.h"
#include "qevent.h"
#include "qfontmetrics.h"
#include "qpainter.h"
#include "qpixmap.h"
#include "qstringlist.h"
#include "qstyle.h"
#include "qstyleoption.h"
#include "qtimer.h"
#include "qvector.h"
#include "qpointer.h"
#ifndef QT_NO_ACCESSIBILITY
#include "qaccessible.h"
#endif

#include <stdlib.h>

/*!
    \class PixmapWidget
    \brief The PixmapWidget class is the base class of all list box items.

    \compat

    This class is an abstract base class used for all list box items.
    If you need to insert customized items into a Q3ListBox you must
    inherit this class and reimplement paint(), height() and width().

    \sa Q3ListBox
*/

/*!
    Constructs an empty list box item in the list box \a listbox.
*/

PixmapWidget::PixmapWidget(const QPixmap &p)
{
    pm = p;
}

/*!
    Constructs an empty list box item in the list box \a listbox and
    inserts it after the item \a after or at the beginning if \a after
    is 0.
*/

PixmapWidget::PixmapWidget(const QPixmap &p, const QString &s)
{
    pm = p;
    name = s;
}

/*!
    Destroys the list box item.
*/

PixmapWidget::~PixmapWidget()
{
}

/*!
    \fn void PixmapWidget::paint(QPainter *p)

    Implement this function to draw your item. The painter, \a p, is
    already open for painting.

    \sa height(), width()
*/

/*!
    \fn int PixmapWidget::width(const Q3ListBox* lb) const

    Reimplement this function to return the width of your item. The \a
    lb parameter is the same as listBox() and is provided for
    convenience and compatibility.

    The default implementation returns
    \l{QApplication::globalStrut()}'s width.

    \sa paint(), height()
*/
int PixmapWidget::width() const
{
    return QApplication::globalStrut().width();
}

/*!
    \fn int PixmapWidget::height(const Q3ListBox* lb) const

    Implement this function to return the height of your item. The \a
    lb parameter is the same as listBox() and is provided for
    convenience and compatibility.

    The default implementation returns
    \l{QApplication::globalStrut()}'s height.

    \sa paint(), width()
*/
int PixmapWidget::height() const
{
    return QApplication::globalStrut().height();
}
