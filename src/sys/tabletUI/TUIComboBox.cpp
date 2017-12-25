/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include <stdio.h>

#include <QComboBox>
#include <QLabel>

#include "TUIComboBox.h"
#include "TUIApplication.h"
#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#else
#include <wce_msg.h>
#endif

/// Constructor
TUIComboBox::TUIComboBox(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
    QComboBox *b = new QComboBox(w);
    b->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    widget = b;
    connect(b, SIGNAL(activated(int)), this, SLOT(valueChanged(int)));
}

/// Destructor
TUIComboBox::~TUIComboBox()
{
    delete widget;
}

void TUIComboBox::valueChanged(int)
{
    QComboBox *cb = (QComboBox *)widget;

    covise::TokenBuffer tb;
    tb << ID;
    QByteArray ba = cb->currentText().toUtf8();
    tb << ba.data();
    TUIMainWindow::getInstance()->send(tb);
}

/** Set activation state of this container and all its children.
  @param en true = elements enabled
*/
void TUIComboBox::setEnabled(bool en)
{
    TUIElement::setEnabled(en);
}

/** Set highlight state of this container and all its children.
  @param hl true = element highlighted
*/
void TUIComboBox::setHighlighted(bool hl)
{
    TUIElement::setHighlighted(hl);
}

const char *TUIComboBox::getClassName() const
{
    return "TUIComboBox";
}

bool TUIComboBox::isOfClassName(const char *classname) const
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

void TUIComboBox::setValue(int type, covise::TokenBuffer &tb)
{
    //cerr << "setValue " << type << endl;
    if (type == TABLET_ADD_ENTRY)
    {
        char *en;
        tb >> en;
        QString entry(en);
        ((QComboBox *)widget)->addItem(entry);
    }
    else if (type == TABLET_REMOVE_ENTRY)
    {
        QComboBox *cb = (QComboBox *)widget;
        int num = cb->count();
        int i;
        char *en;
        tb >> en;
        QString entry(en);
        for (i = 0; i < num; i++)
        {
            if (cb->itemText(i) == entry)
            {
                cb->removeItem(i);
                break;
            }
        }
    }
    else if (type == TABLET_SELECT_ENTRY)
    {
        QComboBox *cb = (QComboBox *)widget;
        int num = cb->count();
        int i;
        char *en;
        tb >> en;
        QString entry(en);
        for (i = 0; i < num; i++)
        {
            if (cb->itemText(i) == entry)
            {
                cb->setCurrentIndex(i);
                break;
            }
        }
    }
    else if (type == TABLET_REMOVE_ALL)
    {
        QComboBox *cb = (QComboBox *)widget;
        cb->clear();
    }
    TUIElement::setValue(type, tb);
}
