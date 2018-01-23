/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <QFrame>
#include <QTextEdit>
#include <QPushButton>
#include <QBoxLayout>

#include "TUIPopUp.h"
#include "TUIApplication.h"
#include "TUIContainer.h"
#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#else
#include <wce_msg.h>
#endif

/// Constructor
TUIPopUp::TUIPopUp(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
    popupButton = new QPushButton(w);
    if (name.contains("."))
    {
        QPixmap pm(name);
        if (pm.isNull())
        {
            QString covisedir = QString(getenv("COVISEDIR"));
            QPixmap pm(covisedir + "/" + name);
            if (pm.isNull())
            {
                popupButton->setText(name);
            }
            else
            {
                popupButton->setIcon(pm);
            }
        }
        else
        {
            popupButton->setIcon(pm);
        }
    }
    else
        popupButton->setText(name);

    //   popupButton->setText(name);
    connect(popupButton, SIGNAL(clicked()), SLOT(popupButtonClicked()));

    popup = new QFrame(popupButton, Qt::Window);
    popup->setFrameStyle(QFrame::WinPanel | QFrame::Raised);

    textEdit = new QTextEdit(popup);
    textEdit->setReadOnly(true);

    closeButton = new QPushButton(tr("&Close"), popup);
    connect(closeButton, SIGNAL(clicked()), popup, SLOT(close()));

    QBoxLayout *layout = new QVBoxLayout(popup);
    closeButton->setMaximumSize(closeButton->sizeHint());
    layout->addWidget(textEdit);
    layout->addWidget(closeButton);

    widget = popupButton;
}

/// Destructor
TUIPopUp::~TUIPopUp()
{
    delete widget;
}

void TUIPopUp::popupButtonClicked()
{
    popup->setVisible(!hidden);
}

const char *TUIPopUp::getClassName() const
{
    return "TUIPopUp";
}

void TUIPopUp::setValue(TabletValue type, covise::TokenBuffer &tb)
{
    if (type == TABLET_STRING)
    {
        char *v;
        tb >> v;
        value = v;
        textEdit->insertPlainText(value);
    }
    TUIElement::setValue(type, tb);
}
