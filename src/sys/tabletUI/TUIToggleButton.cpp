/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include "TUIToggleButton.h"
#include "TUIApplication.h"
#include <stdio.h>
#include <qcheckbox.h>
#include <qradiobutton.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qpixmap.h>
#include <net/tokenbuffer.h>

/// Constructor
TUIToggleButton::TUIToggleButton(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
    if (name == "RadioButton")
    {
        QRadioButton *b = new QRadioButton(w);
        widget = b;
        //b->setFixedSize(b->sizeHint());
        connect(b, SIGNAL(clicked(bool)), this, SLOT(valueChanged(bool)));
    }
    else if (name == "CheckBox")
    {
        QCheckBox *b = new QCheckBox(w);
        widget = b;
        //b->setFixedSize(b->sizeHint());
        connect(b, SIGNAL(stateChanged(int)), this, SLOT(stateChanged(int)));
    }
    else
    {
        QPushButton *b = new QPushButton(w);
        b->setAttribute(Qt::WA_AcceptTouchEvents, true);
        b->setCheckable(true);
        if (name.contains("."))
        {
            QPixmap pm(name);
            if (pm.isNull())
            {
                QString covisedir = QString(getenv("COVISEDIR"));
                QPixmap pm(covisedir + "/" + name);
                if (pm.isNull())
                {
                    b->setText(name);
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
        widget = b;
        //b->setFixedSize(b->sizeHint());
        connect(b, SIGNAL(clicked(bool)), this, SLOT(valueChanged(bool)));
    }
    // dont use toggle, clicked only sends event when the user actually clicked the button and not when the state has been changed by the application
}

/// Destructor
TUIToggleButton::~TUIToggleButton()
{
    delete widget;
}

void TUIToggleButton::valueChanged(bool)
{
    //QPushButton * b = (QPushButton *)widget;
    QAbstractButton *b = (QAbstractButton *)widget;

    covise::TokenBuffer tb;
    tb << ID;
    if (b->isChecked())
    {
        tb << TABLET_ACTIVATED;
    }
    else
    {
        tb << TABLET_DISACTIVATED;
    }
    TUIMainWindow::getInstance()->send(tb);
}

void TUIToggleButton::stateChanged(int)
{
    QAbstractButton *b = (QAbstractButton *)widget;

    covise::TokenBuffer tb;
    tb << ID;
    if (b->isChecked())
    {
        tb << TABLET_ACTIVATED;
    }
    else
    {
        tb << TABLET_DISACTIVATED;
    }
    TUIMainWindow::getInstance()->send(tb);
}

const char *TUIToggleButton::getClassName() const
{
    return "TUIToggleButton";
}

void TUIToggleButton::setValue(TabletValue type, covise::TokenBuffer &tb)
{
    if (type == TABLET_BOOL)
    {
        char state;
        tb >> state;
        bool bState = (bool)state;
        QAbstractButton *b = (QAbstractButton *)widget;
        b->setChecked(bState);
    }
    TUIElement::setValue(type, tb);
}

void TUIToggleButton::setLabel(QString textl)
{
    TUIElement::setLabel(textl);
    if (QAbstractButton* b = qobject_cast<QAbstractButton*>(widget))
    {
        b->setText(textl);
    }
}
