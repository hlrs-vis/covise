/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QLineEdit>

#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#else
#include <wce_msg.h>
#endif
#include "TUILineEdit.h"
#include "TUIApplication.h"
#include "TUIContainer.h"
#include "TUILineCheck.h"

/// Constructor
TUILineEdit::TUILineEdit(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
    editField = new TUILineCheck(w);
    widget = editField;
    connect(editField, SIGNAL(contentChanged()), this, SLOT(valueChanged()));
}

/// Destructor
TUILineEdit::~TUILineEdit()
{
    delete widget;
}

void TUILineEdit::setPos(int x, int y)
{
    xPos = x;
    yPos = y;
    TUIContainer *parent;
    widget = editField;
    if ((parent = getParent()))
    {
        parent->addElementToLayout(this);
    }
    else
    {
        TUIMainWindow::getInstance()->addElementToLayout(this);
    }
    editField->setVisible(!hidden);
}

void TUILineEdit::valueChanged()
{
    value = editField->text();
    covise::TokenBuffer tb;
    tb << ID;
    QByteArray ba = value.toUtf8();
    tb << ba.data();
    TUIMainWindow::getInstance()->send(tb);
}

/** Set activation state of this container and all its children.
  @param en true = elements enabled
*/
void TUILineEdit::setEnabled(bool en)
{
    TUIElement::setEnabled(en);
}

/** Set highlight state of this container and all its children.
  @param hl true = element highlighted
*/
void TUILineEdit::setHighlighted(bool hl)
{
    TUIElement::setHighlighted(hl);
}

const char *TUILineEdit::getClassName() const
{
    return "TUILineEdit";
}

bool TUILineEdit::isOfClassName(const char *classname) const
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

void TUILineEdit::setValue(int type, covise::TokenBuffer &tb)
{
    //cerr << "TUILineEdit::setValue info: type = " << type << endl;
    if (type == TABLET_STRING)
    {
        char *v;
        tb >> v;
        value = v;
        //cerr << "TUILineEdit::setValue " << value << endl;
        editField->setText(value);
    }
    else if (type == TABLET_ECHOMODE)
    {
        int mode;
        tb >> mode;

        if (mode)
            editField->setEchoMode(QLineEdit::Password);
        else
            editField->setEchoMode(QLineEdit::Normal);
    }
    else if (type == TABLET_IPADDRESS)
    {
        int mode;
        tb >> mode;
        if (mode)
            editField->setInputMask("000.000.000.000");
        else
            editField->setInputMask("");
    }

    TUIElement::setValue(type, tb);
}
