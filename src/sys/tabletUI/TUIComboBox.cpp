/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include <stdio.h>

#include <QComboBox>
#include <QLabel>
#include <QGridLayout>
#include <QFrame>

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
    auto frame = new QFrame(w);
    frame->setFrameStyle(QFrame::Plain | QFrame::NoFrame);
    frame->setContentsMargins(0, 0, 0, 0);
    widget = frame;
    combo = new QComboBox(widget);
    combo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    connect(combo, SIGNAL(activated(int)), this, SLOT(valueChanged(int)));
    auto grid = new QGridLayout(frame);
    frame->setLayout(grid);
    layout = grid;
    layout->setContentsMargins(0, 0, 0, 0);
    grid->addWidget(combo, 1, 0);
}

/// Destructor
TUIComboBox::~TUIComboBox()
{
    delete layout;
    delete combo;
    delete label;
}

void TUIComboBox::valueChanged(int)
{
    covise::TokenBuffer tb;
    tb << ID;
    QByteArray ba = combo->currentText().toUtf8();
    tb << ba.data();
    TUIMainWindow::getInstance()->send(tb);
}

const char *TUIComboBox::getClassName() const
{
    return "TUIComboBox";
}

void TUIComboBox::setValue(int type, covise::TokenBuffer &tb)
{
    //cerr << "setValue " << type << endl;
    if (type == TABLET_ADD_ENTRY)
    {
        char *en;
        tb >> en;
        QString entry(en);
        combo->addItem(entry);
    }
    else if (type == TABLET_REMOVE_ENTRY)
    {
        int num = combo->count();
        int i;
        char *en;
        tb >> en;
        QString entry(en);
        for (i = 0; i < num; i++)
        {
            if (combo->itemText(i) == entry)
            {
                combo->removeItem(i);
                break;
            }
        }
    }
    else if (type == TABLET_SELECT_ENTRY)
    {
        int num = combo->count();
        int i;
        char *en;
        tb >> en;
        QString entry(en);
        for (i = 0; i < num; i++)
        {
            if (combo->itemText(i) == entry)
            {
                combo->setCurrentIndex(i);
                break;
            }
        }
    }
    else if (type == TABLET_REMOVE_ALL)
    {
        combo->clear();
    }
    TUIElement::setValue(type, tb);
}

void TUIComboBox::setLabel(QString textl)
{
    if (!label)
    {
        label = new QLabel(widget);
        auto grid = static_cast<QGridLayout *>(layout);
        grid->addWidget(label, 0, 0);
    }
    label->setText(textl);
}
