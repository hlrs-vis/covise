/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include <stdio.h>

#include <QFrame>
#include <QTabWidget>
#include <QGridLayout>
#include <QScrollArea>

#include "TUITab.h"
#include "TUITabFolder.h"
#include "TUIMain.h"
#include <net/tokenbuffer.h>

#include <iostream>

void initScrollable(QScrollArea *scroll, QFrame *frame)
{
        scroll->setMinimumWidth(300);
        scroll->setMinimumHeight(300);
        scroll->setSizePolicy(QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding));
        scroll->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
        scroll->setFrameStyle(QFrame::NoFrame | QFrame::Plain);
        scroll->setWidget(frame);
        scroll->setWidgetResizable(true);
}

/// Constructor
TUITab::TUITab(int id, int type, QWidget *parentWidget, int parent, QString name)
    : TUIContainer(id, type, parentWidget, parent, name)
{
    label = name;

    QFrame *frame = new QFrame(parentWidget);
    frame->setFrameStyle(QFrame::NoFrame | QFrame::Plain);
    frame->setSizePolicy(QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding));
    frame->setContentsMargins(0, 0, 0, 0);
    createLayout(frame);
    frame->setLayout(getLayout());
    
    bool inMainFolder = parent==3;
    if (inMainFolder)
    {
        auto scroll = createWidget<QScrollArea>(parentWidget);
        initScrollable(scroll, frame);
    }
    else{
        setWidget(frame, parentWidget);
    }
}

/// Destructor
TUITab::~TUITab()
{
    removeAllChildren();
}

void TUITab::activated()
{

    if (!firstTime)
    {
        covise::TokenBuffer tb;
        tb << ID;
        tb << TABLET_ACTIVATED;
        TUIMain::getInstance()->send(tb);
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
        TUIMain::getInstance()->send(tb);
    }
}

void TUITab::setPos(int x, int y)
{
    xPos = x;
    yPos = y;
    TUIContainer *parent = getParent();
    if (parent)
    {
        parent->addElementToLayout(this);
        if (auto folder = dynamic_cast<TUITabFolder *>(parent))
        {
            int index = folder->indexOf(widget());
            if (index < 0)
            {
                widget()->show();
                folder->addTab(widget(), label);
                index = folder->indexOf(widget());
            }
            folder->setCurrentIndex(folder->indexOf(widget()));
        }
    }
    else
    {
        TUIMain::getInstance()->addElementToLayout(this);
    }
    setHidden(hidden);
}

void TUITab::setValue(TabletValue type, covise::TokenBuffer &tb)
{
    if (type == TABLET_BOOL)
    {
        if (auto parent = getParent())
        {
            if (auto folder = dynamic_cast<TUITabFolder *>(parent))
                folder->setCurrentIndex(folder->indexOf(widget()));
            else
                std::cerr << "error: parent is not a TUITabFolder" << std::endl;
        }
    }
    TUIContainer::setValue(type, tb);
}

void TUITab::setHidden(bool hide)
{
    //std::cerr << "TUITab::setHidden(hide=" << hide << "), tab=" << getID() << "/" << getName().toStdString() << std::endl;
    TUIContainer::setHidden(hide);
    if (TUIContainer *parent = getParent())
    {
        if (auto folder = dynamic_cast<TUITabFolder *>(parent))
        {
            int index = folder->indexOf(widget());
            if (hidden)
            {
                if (index >= 0)
                    folder->removeTab(index);
            }
            else
            {
                if (index < 0)
                {
                    widget()->show();
                    folder->addTab(widget(), label);
                }
            }
        }
        else
        {
            std::cerr << "TUITab::setHidden(): parent of " << getID() << "/" << getName().toStdString() << " is not a TUITabFolder but a " << parent->getClassName() << std::endl;
        }
    }
    else
    {
        //std::cerr << "TUITab::setHidden(): no parent for " << getID() << "/" << getName().toStdString() << std::endl;
    }
}

const char *TUITab::getClassName() const
{
    return "TUITab";
}
