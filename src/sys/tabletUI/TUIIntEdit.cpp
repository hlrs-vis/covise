/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include <iostream>

#include <QString>
#include <QLineEdit>
#include <QValidator>

#include <net/tokenbuffer.h>

#include "TUIIntEdit.h"
#include "TUIApplication.h"
#include "TUIContainer.h"
#include "TUILineCheck.h"

/// Constructor
TUIIntEdit::TUIIntEdit(int id, int type, QWidget *w, int parent, QString name)
    : TUILineEdit(id, type, w, parent, name)
{
    validator = new QIntValidator(editField);
}

void TUIIntEdit::valueChanged()
{
    value = editField->text().toInt();
    covise::TokenBuffer tb;
    tb << ID;
    tb << value;
    TUIMainWindow::getInstance()->send(tb);
}

const char *TUIIntEdit::getClassName() const
{
    return "TUIIntEdit";
}

void TUIIntEdit::setValue(TabletValue type, covise::TokenBuffer &tb)
{
    //cerr << "TUIIntEdit::setValue info: type = " << type << endl;
    if (type == TABLET_MIN)
    {
        tb >> min;
        //std::cerr << "TUIIntEdit::setMin " << min << std::endl;
        validator->setBottom(min);
    }
    else if (type == TABLET_MAX)
    {
        tb >> max;
        //std::cerr << "TUIIntEdit::setMax " << max << std::endl;
        validator->setTop(max);
    }
    else if (type == TABLET_INT)
    {
        tb >> value;
        //cerr << "TUIIntEdit::setValue " << value << endl;
        editField->setText(QString::number(value));
    }
    TUIElement::setValue(type, tb);
}
