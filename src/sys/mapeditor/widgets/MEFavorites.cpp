/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QMimeData>
#include <QMouseEvent>
#include <QToolBar>
#include <QDrag>

#include "MEFavorites.h"
#include "handler/MEMainHandler.h"
#include "handler/MEFavoriteListHandler.h"
#include "handler/MEHostListHandler.h"
#include "hosts/MEHost.h"

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MEFavorites::MEFavorites(QWidget *parent, QString sname)
    : QToolButton(parent)
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
{
    setFocusPolicy(Qt::StrongFocus);
    setAcceptDrops(true);
    setModuleName(sname);
    setFocusPolicy(Qt::NoFocus);
    setFont(MEMainHandler::s_boldFont);

    setToolTip("<p>Drop onto visual programming area to start,<br>drag back to module browser to remove</p>");

    connect(this, SIGNAL(clicked()), MEMainHandler::instance(), SLOT(initModule()));
    m_action = static_cast<QToolBar *>(parent)->addWidget(this);
}

MEFavorites::~MEFavorites()
{
}

//------------------------------------------------------------------------
// store category & modulename info
//------------------------------------------------------------------------
void MEFavorites::setModuleName(const QString &sname)
{
    m_label = sname.section(":", 0, 0);
    m_category = sname.section(":", 1, 1);
    setText(m_label);
}

//------------------------------------------------------------------------
// return combind name
//------------------------------------------------------------------------
QString MEFavorites::getModuleName()
{
    return (m_label + ":" + m_category);
}

//------------------------------------------------------------------------
// create the text object for dragging
//------------------------------------------------------------------------
QString MEFavorites::getStartText()
{
#ifdef YAC
    MEHost *h = MEHostListHandler::instance()->getFirstHost();
    int hostid = -1;
    if (h)
    {
        hostid = h->getID();
    }
    QString text = QString::number(hostid);
#else
    QString text = MEHostListHandler::instance()->getIPAddress(MEMainHandler::instance()->localHost);
#endif

    text.append(":" + MEMainHandler::instance()->localUser + ":" + m_category + ":" + m_label);

    return text;
}

//------------------------------------------------------------------------
//
//------------------------------------------------------------------------
void MEFavorites::mouseDoubleClickEvent(QMouseEvent *)
{
    MEFavoriteListHandler::instance()->sortFavorites();
}

//------------------------------------------------------------------------
// create the text object for dragging
//------------------------------------------------------------------------
void MEFavorites::mouseMoveEvent(QMouseEvent *e)
{
    QString text = getStartText();

    // create a mime source
    QMimeData *mimeData = new QMimeData;
    mimeData->setText(text);

    // create drag object
    QDrag *drag = new QDrag(this);
    drag->setMimeData(mimeData);
    drag->setPixmap(MEMainHandler::instance()->pm_file);

    drag->start();
    e->accept();
}

//------------------------------------------------------------------------
// is this drag allowed ?
//------------------------------------------------------------------------
void MEFavorites::dragEnterEvent(QDragEnterEvent *event)
{
    // don't allow drops from outside the application
    if (event->source() == NULL)
        return;

    if (event->mimeData()->hasFormat("application/x-qabstractitemmodeldatalist") || event->mimeData()->hasText())
        event->accept();
    else
        event->ignore();
}

//------------------------------------------------------------------------
// is this drag allowed ?
//------------------------------------------------------------------------
void MEFavorites::dragMoveEvent(QDragMoveEvent *event)
{
    // don't allow drops from outside the application
    if (event->source() == NULL)
        return;

    if (event->source() == this || event->source() == this->parentWidget())
        event->ignore();

    else if (event->mimeData()->hasFormat("application/x-qabstractitemmodeldatalist") || event->mimeData()->hasText())
        event->accept();
}

//------------------------------------------------------------------------
// is this drag allowed ?
//------------------------------------------------------------------------
void MEFavorites::dragLeaveEvent(QDragLeaveEvent *event)
{
    event->accept();
}

//------------------------------------------------------------------------
// reread the dragged module information
// append modulename to favorite list
//------------------------------------------------------------------------
void MEFavorites::dropEvent(QDropEvent *event)
{
    // from module tree
    if (event->mimeData()->hasFormat("application/x-qabstractitemmodeldatalist"))
    {
        QByteArray encodedData = event->mimeData()->data("application/x-qabstractitemmodeldatalist");
        QDataStream stream(&encodedData, QIODevice::ReadOnly);

        while (!stream.atEnd())
        {
            QString text;
            stream >> text;
            QString combiname = text.section(':', 3, 3) + ":" + text.section(':', 2, 2);
            QString insertname = m_label + ":" + m_category;

            if (text.section(':', 3, 3) != m_label)
                MEFavoriteListHandler::instance()->insertFavorite(combiname, insertname);
        }
        event->accept();
    }

    else if (event->mimeData()->hasText())
    {
        QString text = event->mimeData()->text();
        QString combiname = text.section(':', 3, 3) + ":" + text.section(':', 2, 2);
        QString insertname = m_label + ":" + m_category;

        if (text.section(':', 3, 3) != m_label)
            MEFavoriteListHandler::instance()->insertFavorite(combiname, insertname);
        event->accept();
    }
    else
        event->ignore();
}
