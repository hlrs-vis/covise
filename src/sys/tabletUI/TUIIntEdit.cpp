/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include <iostream>

#include <QString>
#include <QLineEdit>
#include <QValidator>

#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#else
#include <wce_msg.h>
#endif

#include "TUIIntEdit.h"
#include "TUIApplication.h"
#include "TUIContainer.h"
#include "TUILineCheck.h"

/// Constructor
TUIIntEdit::TUIIntEdit(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{

    intEdit = new TUILineCheck(w);
    validator = new QIntValidator(intEdit);
    //intEdit->setValidator(validator);

    connect(intEdit, SIGNAL(contentChanged()), this, SLOT(valueChanged()));
}

/// Destructor
TUIIntEdit::~TUIIntEdit()
{
    delete widget;
}

void TUIIntEdit::setPos(int x, int y)
{
    xPos = x;
    yPos = y;
    TUIContainer *parent;
    widget = intEdit;
    if ((parent = getParent()))
    {
        parent->addElementToLayout(this);
    }
    else
    {
        TUIMainWindow::getInstance()->addElementToLayout(this);
    }
    intEdit->setVisible(!hidden);
}

void TUIIntEdit::valueChanged()
{
    value = intEdit->text().toInt();
    covise::TokenBuffer tb;
    tb << ID;
    tb << value;
    TUIMainWindow::getInstance()->send(tb);
}

const char *TUIIntEdit::getClassName() const
{
    return "TUIIntEdit";
}

void TUIIntEdit::setValue(int type, covise::TokenBuffer &tb)
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
        intEdit->setText(QString::number(value));
    }
    TUIElement::setValue(type, tb);
}
