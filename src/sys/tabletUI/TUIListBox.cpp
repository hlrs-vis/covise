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

const char *TUIListBox::getClassName() const
{
    return "TUIListBox";
}

void TUIListBox::setValue(TabletValue type, covise::TokenBuffer &tb)
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
