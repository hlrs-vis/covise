/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <assert.h>
#include <stdio.h>
#include <iostream>

#include <QFrame>
#include <QGroupBox>
#include <QGridLayout>
#include <QTabWidget>
#include <QDir>

#include "TUIGroupBox.h"
#include "TUIApplication.h"
#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#else
#include <wce_msg.h>
#endif

/// Constructor
TUIGroupBox::TUIGroupBox(int id, int type, QWidget *w, int parent, QString name)
    : TUIContainer(id, type, w, parent, name)
{
    label = name;
    auto gb = new QGroupBox(w);

#ifdef _WIN32_WCE
    gb->setContentsMargins(1, 1, 1, 1);
#else
    gb->setContentsMargins(5, 5, 5, 5);
#endif
    gb->setSizePolicy(QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding));

    layout = new QGridLayout(gb);
    widget = gb;
}

/// Destructor
TUIGroupBox::~TUIGroupBox()
{
    removeAllChildren();
    delete layout;
    delete widget;
}

void TUIGroupBox::setPos(int x, int y)
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

const char *TUIGroupBox::getClassName() const
{
    return "TUIGroupBox";
}

void TUIGroupBox::setLabel(QString textl)
{
    TUIContainer::setLabel(textl);
    auto gb = static_cast<QGroupBox *>(widget);
    gb->setTitle(textl);
}

void TUIGroupBox::setValue(int type, covise::TokenBuffer &tb)
{
    if (type == TABLET_SHAPE)
    {
        int shape;
        tb >> shape;
    }
    if (type == TABLET_STYLE)
    {
        int style;
        tb >> style;
    }
    TUIElement::setValue(type, tb);
}
