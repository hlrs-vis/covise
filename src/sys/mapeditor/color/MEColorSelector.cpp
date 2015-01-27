/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include <QApplication>
#include <QDesktopWidget>
#include <QMouseEvent>
#include <QCursor>

#include "MEColorSelector.h"
#include "MEColorSelector.h"
#include "handler/MEMainHandler.h"

/*****************************************************************************
 *
 * Class MEColorSelector
 *
 *****************************************************************************/

MEColorSelector::MEColorSelector(QWidget *parent)
    : QPushButton(parent)
{
    setFlat(true);
    setFocusPolicy(Qt::NoFocus);
    setIcon(MEMainHandler::instance()->pm_colorpicker);
    m_colorPicking = false;
}

/*void MEColorSelector::mouseMoveEvent(QMouseEvent *e)
{
   if (m_colorPicking) 54

   {
      return;
    }

    QPushButton::mouseMoveEvent(e);
}*/

void MEColorSelector::mouseReleaseEvent(QMouseEvent *e)
{
    if (m_colorPicking)
    {
        m_colorPicking = false;
        releaseMouse();
        releaseKeyboard();
        QColor color = grabColor(e->globalPos());
        emit pickedColor(color);
    }

    QPushButton::mouseReleaseEvent(e);
}

QColor MEColorSelector::grabColor(const QPoint &p)
{
    QDesktopWidget *desktop = QApplication::desktop();
    QPixmap pm = QPixmap::grabWindow(desktop->winId(), p.x(), p.y(), 1, 1);
    QImage i = pm.toImage();
    return i.pixel(0, 0);
}

void MEColorSelector::selectedColorCB()
{
    m_colorPicking = true;
    grabMouse(Qt::CrossCursor);
    grabKeyboard();
}
