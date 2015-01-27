/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include "TUIToggleBitmapButton.h"
#include "TUIApplication.h"
#include <stdio.h>
#include <qcheckbox.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qpixmap.h>
#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#else
#include <wce_msg.h>
#endif

/// Constructor
TUIToggleBitmapButton::TUIToggleBitmapButton(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
    QPushButton *b = new QPushButton(w);
    b->setCheckable(true);
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
                upName = name;
                b->setText(name);
            }
            else if (pm.isNull())
            {
                upName = covisedir + "/icons/" + name;
                b->setIcon(qm);
            }
            else
            {
                upName = covisedir + "/" + name;
                b->setIcon(pm);
            }
        }
        else
        {
            upName = name;
            b->setIcon(pm);
        }
    }
    else
    {
        upName = name;
        b->setText(name);
    }

    //b->setFixedSize(b->sizeHint());
    widget = b;
    // dont use toggle, clicked only sends event when the user actually clicked the button and not when the state has been changed by the application
    connect(b, SIGNAL(clicked(bool)), this, SLOT(valueChanged(bool)));
}

/// Destructor
TUIToggleBitmapButton::~TUIToggleBitmapButton()
{
    delete widget;
}

void TUIToggleBitmapButton::valueChanged(bool)
{
    QCheckBox *b = (QCheckBox *)widget;

    covise::TokenBuffer tb;
    tb << ID;
    if (b->isChecked())
    {
        tb << TABLET_ACTIVATED;
        QPixmap pm(downName);

        if (pm.isNull())
            b->setText(downName);
        else
            b->setIcon(pm);
    }
    else
    {
        tb << TABLET_DISACTIVATED;
        QPixmap pm(upName);
        if (pm.isNull())
            b->setText(upName);
        else
            b->setIcon(pm);
    }
    TUIMainWindow::getInstance()->send(tb);
}

/** Set activation state of this container and all its children.
  @param en true = elements enabled
*/
void TUIToggleBitmapButton::setEnabled(bool en)
{
    TUIElement::setEnabled(en);
}

/** Set highlight state of this container and all its children.
  @param hl true = element highlighted
*/
void TUIToggleBitmapButton::setHighlighted(bool hl)
{
    TUIElement::setHighlighted(hl);
}

void TUIToggleBitmapButton::setSize(int w, int h)
{
    QPushButton *b = (QPushButton *)widget;
    b->setIconSize(QSize(w, h)); /* Max size of icons, smaller icons will not be scaled up */
    b->setFixedSize(b->sizeHint());
}

char *TUIToggleBitmapButton::getClassName()
{
    return (char *)"TUIToggleBitmapButton";
}

bool TUIToggleBitmapButton::isOfClassName(char *classname)
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

void TUIToggleBitmapButton::setValue(int type, covise::TokenBuffer &tb)
{
    if (type == TABLET_BOOL)
    {
        char state;
        tb >> state;
        bool bState = (bool)state;
        QCheckBox *b = (QCheckBox *)widget;
        b->setChecked(bState);
        if (b->isChecked())
        {
            QPixmap pm(downName);

            if (pm.isNull())
                b->setText(downName);
            else
                b->setIcon(pm);
        }
        else
        {
            QPixmap pm(upName);
            if (pm.isNull())
                b->setText(upName);
            else
                b->setIcon(pm);
        }
    }
    else if (type == TABLET_STRING)
    {
        char *v;
        tb >> v;
        QString name = v;

        if (name.contains("."))
        {
            QPixmap pm(name);
            if (pm.isNull())
            {
                QString covisedir = QString(getenv("COVISEDIR"));
                QPixmap pm(covisedir + "/" + name);
                QPixmap qm(covisedir + "/icons/" + name);
                if (pm.isNull() && qm.isNull())
                    downName = name;
                else if (pm.isNull())
                    downName = covisedir + "/icons/" + name;
                else
                    downName = covisedir + "/" + name;
            }
            else
                downName = name;
        }
        else
            downName = name;
    }
    TUIElement::setValue(type, tb);
}
