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
#include <net/tokenbuffer.h>

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

void TUIButton::setSize(int w, int h)
{
    QPushButton *b = (QPushButton *)widget;
    b->setIconSize(QSize(w, h)); /* Max size of icons, smaller icons will not be scaled up */
    b->setFixedSize(b->sizeHint());
}

const char *TUIButton::getClassName() const
{
    return "TUIButton";
}

void TUIButton::setLabel(QString textl)
{
    TUIElement::setLabel(textl);
    if (QAbstractButton* b = qobject_cast<QAbstractButton*>(widget))
    {
        b->setText(textl);
    }
}
