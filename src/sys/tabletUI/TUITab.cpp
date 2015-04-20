/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include <stdio.h>

#include <QFrame>
#include <QTabWidget>
#include <QGridLayout>

#include "TUITab.h"
#include "TUIApplication.h"
#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#else
#include <wce_msg.h>
#endif

#include <iostream>

/// Constructor
TUITab::TUITab(int id, int type, QWidget *w, int parent, QString name)
    : TUIContainer(id, type, w, parent, name)
{
    label = name;

    QFrame *frame = new QFrame(w);
    frame->setFrameStyle(QFrame::NoFrame);
#ifdef _WIN32_WCE
    frame->setContentsMargins(1, 1, 1, 1);
#else
    frame->setContentsMargins(5, 5, 5, 5);
#endif

    layout = new QGridLayout(frame);
    widget = frame;
    firstTime = true;
}

/// Destructor
TUITab::~TUITab()
{
    removeAllChildren();
    delete layout;
    delete widget;
}

void TUITab::activated()
{

    if (!firstTime)
    {
        covise::TokenBuffer tb;
        tb << ID;
        tb << TABLET_ACTIVATED;
        TUIMainWindow::getInstance()->send(tb);
    }
    firstTime = false;
}

void TUITab::setLabel(QString textl)
{
    TUIContainer::setLabel(textl);
    if (TUIContainer* p = getParent())
    {
        p->addElementToLayout(this);
    }
}

void TUITab::deActivate(TUITab *activedTab)
{

    if (activedTab != this)
    {
        covise::TokenBuffer tb;
        tb << ID;
        tb << TABLET_DISACTIVATED;
        TUIMainWindow::getInstance()->send(tb);
    }
}

void TUITab::setPos(int x, int y)
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

void TUITab::setValue(int type, covise::TokenBuffer &tb)
{
    if (type == TABLET_BOOL)
    {
        TUIContainer *parent;
        if ((parent = getParent()))
        {
            if (QTabWidget *tw = qobject_cast<QTabWidget *>(parent->getWidget()))
                tw->setCurrentIndex(tw->indexOf(widget));
            else
                std::cerr << "error: parent is not a QTabWidget" << std::endl;
        }
    }
    TUIElement::setValue(type, tb);
}

void TUITab::setHidden(bool hide)
{
    TUIContainer::setHidden(hide);
    if (TUIContainer *parent = getParent())
    {
        if (QTabWidget *tw = qobject_cast<QTabWidget *>(parent->getWidget()))
        {
            int index = tw->indexOf(widget);
            if (hidden)
            {
                if (index >= 0)
                    tw->removeTab(index);
            }
            else
            {
                if (index < 0)
                    tw->addTab(widget, label);
            }
        }
    }
}

char *TUITab::getClassName()
{
    return (char *)"TUITab";
}

bool TUITab::isOfClassName(char *classname)
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
