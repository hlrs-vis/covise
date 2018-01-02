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

/** Set activation state of this container and all its children.
  @param en true = elements enabled
*/
void TUIIntEdit::setEnabled(bool en)
{
    TUIElement::setEnabled(en);
}

/** Set highlight state of this container and all its children.
  @param hl true = element highlighted
*/
void TUIIntEdit::setHighlighted(bool hl)
{
    TUIElement::setHighlighted(hl);
}

const char *TUIIntEdit::getClassName() const
{
    return "TUIIntEdit";
}

bool TUIIntEdit::isOfClassName(const char *classname) const
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
