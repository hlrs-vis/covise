/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <assert.h>
#include <stdio.h>
#include <iostream>

#include <QFrame>
#include <QAbstractScrollArea>
#include <QGridLayout>
#include <QTabWidget>

#include "TUIScrollArea.h"
#include "TUIApplication.h"
#include <net/tokenbuffer.h>

/// Constructor
TUIScrollArea::TUIScrollArea(int id, int type, QWidget *w, int parent, QString name)
    : TUIContainer(id, type, w, parent, name)
{
    label = name;

    QAbstractScrollArea *frame = new QAbstractScrollArea(w);
    frame->setFrameStyle(QFrame::NoFrame);
#ifdef _WIN32_WCE
    frame->setContentsMargins(1, 1, 1, 1);
#else
    frame->setContentsMargins(5, 5, 5, 5);
#endif

    layout = new QGridLayout(frame);
    widget = frame;
}

/// Destructor
TUIScrollArea::~TUIScrollArea()
{
    removeAllChildren();
    delete layout;
    delete widget;
}

void TUIScrollArea::setPos(int x, int y)
{
    xPos = x;
    yPos = y;
    TUIContainer *parent = getParent();
    if (parent)
    {
        parent->addElementToLayout(this);
        if (QTabWidget *tw = qobject_cast<QTabWidget *>(parent->getWidget()))
        {
            tw->setCurrentIndex(tw->indexOf(widget));
        }
        else
            std::cerr << "error: parent is not a QTabWidget" << std::endl;
    }
    else
    {
        TUIMainWindow::getInstance()->addElementToLayout(this);
    }
    widget->setVisible(!hidden);
}

const char *TUIScrollArea::getClassName() const
{
    return "TUIScrollArea";
}

void TUIScrollArea::setValue(TabletValue type, covise::TokenBuffer &tb)
{
    QFrame *frame = (QFrame *)widget;
    if (type == TABLET_TYPE)
    {
        int type;
        tb >> type;
        frame->setFrameShape((QFrame::Shape)type);
    }
    if (type == TABLET_STYLE)
    {
        int style;
        tb >> style;
        frame->setFrameStyle(style);
    }
    TUIElement::setValue(type, tb);
}
