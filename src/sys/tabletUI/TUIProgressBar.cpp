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
#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#else
#include <wce_msg.h>
#endif

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

/** Set activation state of this container and all its children.
  @param en true = elements enabled
*/
void TUIProgressBar::setEnabled(bool en)
{
    TUIElement::setEnabled(en);
}

/** Set highlight state of this container and all its children.
  @param hl true = element highlighted
*/
void TUIProgressBar::setHighlighted(bool hl)
{
    TUIElement::setHighlighted(hl);
}

const char *TUIProgressBar::getClassName() const
{
    return "TUIProgressBar";
}

bool TUIProgressBar::isOfClassName(const char *classname) const
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

void TUIProgressBar::setValue(int type, covise::TokenBuffer &tb)
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
