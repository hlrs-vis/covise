/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include "TUIProgressBar.h"
#include "TUIApplication.h"
#include <stdio.h>
#include <QLabel>
#include <QProgressBar>
#include <QString>
#include <QLineEdit>
#include "TUIContainer.h"
#include <net/tokenbuffer.h>

/// Constructor
TUIProgressBar::TUIProgressBar(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
    //int row  = 0;

    pb = new QProgressBar(w);
    pb->setRange(0, 100);
    pb->setValue(0);
    widget = pb;
}

/// Destructor
TUIProgressBar::~TUIProgressBar()
{
    delete widget;
}

void TUIProgressBar::setPos(int x, int y)
{
    xPos = x;
    yPos = y;
    TUIContainer *parent;
    if ((parent = getParent()))
    {
        parent->addElementToLayout(this);
    }
    else
    {
        TUIMainWindow::getInstance()->addElementToLayout(this);
    }
    pb->setVisible(!hidden);
}

const char *TUIProgressBar::getClassName() const
{
    return "TUIProgressBar";
}

void TUIProgressBar::setValue(TabletValue type, covise::TokenBuffer &tb)
{
    //cerr << "TUIProgressBar::setValue info: type = " << type << endl;

    if (type == TABLET_MAX)
    {
        tb >> max;
        if (value > max)
            value = max;
        pb->setRange(0, max);
        pb->setValue(value);
    }
    else if (type == TABLET_INT)
    {
        tb >> value;
        if (value > max)
            value = max;
        pb->setValue(value);
    }
    TUIElement::setValue(type, tb);
}
