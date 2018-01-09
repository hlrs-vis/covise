/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <assert.h>
#include <stdio.h>
#include <iostream>

#include <QFrame>
#include <QGridLayout>
#include <QTabWidget>
#include <QDir>

#include "TUIFrame.h"
#include "TUIApplication.h"
#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#else
#include <wce_msg.h>
#endif

/// Constructor
TUIFrame::TUIFrame(int id, int type, QWidget *w, int parent, QString name)
    : TUIContainer(id, type, w, parent, name)
{
    label = name;
    QFrame *frame = new QFrame(w);
    if (name.contains("."))
    {
        QPixmap pm(name);
        //           frame->setFrameStyle(QFrame::StyledPanel);
        if (pm.isNull())
        {
            QString covisedirname = QString(getenv("COVISEDIR")) + "/" + name;
            QPixmap pm(covisedirname);
            if (!pm.isNull())
            {
                QString filename = "background-image: url(" + QDir(covisedirname).path() + ");";
                frame->setStyleSheet(filename);
            }
        }
        else
        {
            frame->setStyleSheet("background-image: url(" + name + ");");
        }
    }

    frame->setFrameStyle(QFrame::NoFrame);
#ifdef _WIN32_WCE
    frame->setContentsMargins(1, 1, 1, 1);
#else
    frame->setContentsMargins(5, 5, 5, 5);
#endif
    frame->setSizePolicy(QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding));

    layout = new QGridLayout(frame);
    widget = frame;
}

/// Destructor
TUIFrame::~TUIFrame()
{
    removeAllChildren();
    delete layout;
    delete widget;
}

void TUIFrame::setPos(int x, int y)
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
        //else
        //std::cerr << "error: parent is not a QTabWidget" << std::endl;
    }
    else
    {
        TUIMainWindow::getInstance()->addElementToLayout(this);
    }
    widget->setVisible(!hidden);
}

const char *TUIFrame::getClassName() const
{
    return "TUIFrame";
}

void TUIFrame::setValue(int type, covise::TokenBuffer &tb)
{
    QFrame *frame = (QFrame *)widget;
    if (type == TABLET_SHAPE)
    {
        int shape;
        tb >> shape;
        frame->setFrameShape((QFrame::Shape)shape);
    }
    if (type == TABLET_STYLE)
    {
        int style;
        tb >> style;
        frame->setFrameStyle((QFrame::Shadow)style);
    }
    TUIElement::setValue(type, tb);
}
