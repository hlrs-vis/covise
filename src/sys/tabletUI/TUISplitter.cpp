/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include <stdio.h>
#include <iostream>

#include <QSplitter>
#include <QTabWidget>
#include <QGridLayout>

#include "TUISplitter.h"
#include "TUIApplication.h"
#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#else
#include <wce_msg.h>
#endif

/// Constructor
TUISplitter::TUISplitter(int id, int type, QWidget *w, int parent, QString name)
    : TUIContainer(id, type, w, parent, name)
{

    label = name;

    QSplitter *split = new QSplitter(Qt::Horizontal, w);
    split->setFrameStyle(QFrame::NoFrame);
    //   split->setContentsMargins(5,5,5,5);
    hBoxLayout = new QHBoxLayout(split);
    vBoxLayout = NULL;
    widget = split;
}

/// Destructor
TUISplitter::~TUISplitter()
{
    removeAllChildren();
    delete layout;
    delete widget;
}

void TUISplitter::setPos(int x, int y)
{
    xPos = x;
    yPos = y;
    TUIContainer *parent;
    if ((parent = getParent()))
    {
        parent->addElementToLayout(this);
        if (QTabWidget *tw = qobject_cast<QTabWidget *>(parent->getWidget()))
            tw->setCurrentIndex(tw->indexOf(widget));
        else
            std::cerr << "error: parent is not a QTabWidget" << std::endl;
    }
    else
    {
        TUIMainWindow::getInstance()->addElementToLayout(this);
    }
    widget->setVisible(!hidden);
}

char *TUISplitter::getClassName()
{
    return (char *)"TUISplitter";
}

bool TUISplitter::isOfClassName(char *classname)
{
    // paranoia makes us mistrust the string library and check for NULL.
    if (classname && getClassName())
    {
        // check for identity
        if (!strcmp(classname, getClassName()))
        { // we are the one
            return true;
        }
        else
        { // we are not the wanted one. Branch up to parent class
            return TUIElement::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}

void TUISplitter::setValue(int type, covise::TokenBuffer &tb)
{
    QSplitter *split = (QSplitter *)widget;
    if (type == TABLET_SHAPE)
    {
        int shape;
        tb >> shape;
        split->setFrameShape((QFrame::Shape)shape);
    }
    if (type == TABLET_STYLE)
    {
        int style;
        tb >> style;
        split->setFrameStyle((QFrame::Shadow)style);
    }
    if (type == TABLET_ORIENTATION)
    {
        int orientation;
        tb >> orientation;
        if ((orientation == Qt::Vertical) && !vBoxLayout)
        {
            delete hBoxLayout;
            vBoxLayout = new QVBoxLayout(widget);
        }
        else if ((orientation == Qt::Horizontal) && !hBoxLayout)
        {
            delete vBoxLayout;
            hBoxLayout = new QHBoxLayout(widget);
        }

        split->setOrientation((Qt::Orientation)orientation);
    }
    TUIElement::setValue(type, tb);
}
