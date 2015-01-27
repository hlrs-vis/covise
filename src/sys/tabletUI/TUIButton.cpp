/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include <stdio.h>

#include <QPushButton>
#include <QLabel>
#include <QPixmap>

#include "TUIButton.h"
#include "TUIApplication.h"
#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#else
#include <wce_msg.h>
#endif

/// Constructor
TUIButton::TUIButton(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
    QPushButton *b = new QPushButton(w);
    if (name.contains("."))
    {
        QPixmap pm(name);
        if (pm.isNull())
        {
            QString covisedir = QString(getenv("COVISEDIR"));
            QPixmap pm(covisedir + "/" + name);
            QPixmap qm(covisedir + "/icons/" + name);
            if (pm.isNull() && qm.isNull())
            {
                b->setText(name);
            }
            else if (pm.isNull())
            {
                b->setIcon(qm);
            }
            else
            {
                b->setIcon(pm);
            }
        }
        else
        {
            b->setIcon(pm);
        }
    }
    else
        b->setText(name);

    //b->setFixedSize(b->sizeHint());
    widget = b;
    connect(b, SIGNAL(pressed()), this, SLOT(pressed()));
    connect(b, SIGNAL(released()), this, SLOT(released()));
}

/// Destructor
TUIButton::~TUIButton()
{
    delete widget;
}

void TUIButton::pressed()
{
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_PRESSED;
    TUIMainWindow::getInstance()->send(tb);
}

void TUIButton::released()
{
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_RELEASED;
    TUIMainWindow::getInstance()->send(tb);
}

/** Set activation state of this container and all its children.
  @param en true = elements enabled
*/
void TUIButton::setEnabled(bool en)
{
    TUIElement::setEnabled(en);
}

/** Set highlight state of this container and all its children.
  @param hl true = element highlighted
*/
void TUIButton::setHighlighted(bool hl)
{
    TUIElement::setHighlighted(hl);
}

void TUIButton::setSize(int w, int h)
{
    QPushButton *b = (QPushButton *)widget;
    b->setIconSize(QSize(w, h)); /* Max size of icons, smaller icons will not be scaled up */
    b->setFixedSize(b->sizeHint());
}

char *TUIButton::getClassName()
{
    return (char *)"TUIButton";
}

bool TUIButton::isOfClassName(char *classname)
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
