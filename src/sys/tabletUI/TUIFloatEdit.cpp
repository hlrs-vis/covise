/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include <stdio.h>

#include <QLabel>
#include <QString>
#include <QLineEdit>

#include <net/tokenbuffer.h>

#include "TUIFloatEdit.h"
#include "TUIApplication.h"
#include "TUIContainer.h"
#include "TUILineCheck.h"

/// Constructor
TUIFloatEdit::TUIFloatEdit(int id, int type, QWidget *w, int parent, QString name)
    : TUILineEdit(id, type, w, parent, name)
{}

void TUIFloatEdit::valueChanged()
{
    value = editField->text().toFloat();
    covise::TokenBuffer tb;
    tb << ID;
    tb << value;
    TUIMainWindow::getInstance()->send(tb);
}

const char *TUIFloatEdit::getClassName() const
{
    return "TUIFloatEdit";
}

void TUIFloatEdit::setValue(TabletValue type, covise::TokenBuffer &tb)
{
    if (type == TABLET_FLOAT)
    {
        value = 123.4f;
        tb >> value;
        //cerr << "TUIFloatEdit::setValue " << value << endl;
        QString tmp;
        tmp = QString("%1").arg(value);
        editField->setText(tmp);
    }
    TUIElement::setValue(type, tb);
}
