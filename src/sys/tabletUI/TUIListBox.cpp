/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include <stdio.h>

#include <QListWidget>
#include <QLabel>

#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#else
#include <wce_msg.h>
#endif

#include "TUIListBox.h"
#include "TUIApplication.h"

/// Constructor
TUIListBox::TUIListBox(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
    QListWidget *b = new QListWidget(w);
    b->setFixedSize(b->sizeHint());
    widget = b;
    connect(b, SIGNAL(selected(int)), this, SLOT(valueChanged(int)));
}

/// Destructor
TUIListBox::~TUIListBox()
{
    delete widget;
}

void TUIListBox::valueChanged(int)
{
    QListWidget *cb = static_cast<QListWidget *>(widget);

    covise::TokenBuffer tb;
    tb << ID;
    QByteArray ba = cb->currentItem()->text().toUtf8();
    tb << ba.data();
    TUIMainWindow::getInstance()->send(tb);
}

/** Set activation state of this container and all its children.
  @param en true = elements enabled
*/
void TUIListBox::setEnabled(bool en)
{
    TUIElement::setEnabled(en);
}

/** Set highlight state of this container and all its children.
  @param hl true = element highlighted
*/
void TUIListBox::setHighlighted(bool hl)
{
    TUIElement::setHighlighted(hl);
}

const char *TUIListBox::getClassName() const
{
    return "TUIListBox";
}

bool TUIListBox::isOfClassName(const char *classname) const
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

void TUIListBox::setValue(int type, covise::TokenBuffer &tb)
{
    //cerr << "setValue " << type << endl;
    if (type == TABLET_ADD_ENTRY)
    {
        char *en;
        tb >> en;
        QString entry(en);
        static_cast<QListWidget *>(widget)->addItem(entry);
    }
    else if (type == TABLET_REMOVE_ENTRY)
    {
        QListWidget *cb = static_cast<QListWidget *>(widget);
        int num = cb->count();
        int i;
        char *en;
        tb >> en;
        QString entry(en);
        for (i = 0; i < num; i++)
        {
            if (cb->item(i)->text() == entry)
            {
                QListWidgetItem *item = cb->takeItem(i);
                delete item;
                break;
            }
        }
    }
    else if (type == TABLET_SELECT_ENTRY)
    {
        QListWidget *cb = static_cast<QListWidget *>(widget);
        int num = cb->count();
        int i;
        char *en;
        tb >> en;
        QString entry(en);
        for (i = 0; i < num; i++)
        {
            if (cb->item(i)->text() == entry)
            {
                cb->setCurrentRow(i);
                break;
            }
        }
    }
    TUIElement::setValue(type, tb);
}
