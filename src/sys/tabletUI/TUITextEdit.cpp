/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <QTextEdit>

#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#else
#include <wce_msg.h>
#endif
#include "TUITextEdit.h"
#include "TUIApplication.h"
#include "TUIContainer.h"
#include "TUITextCheck.h"

/// Constructor
TUITextEdit::TUITextEdit(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
    editField = new TUITextCheck(w);
    editField->setMinimumHeight(200);
    widget = editField;
    connect(editField, SIGNAL(contentChanged()), this, SLOT(valueChanged()));
}

/// Destructor
TUITextEdit::~TUITextEdit()
{
    delete widget;
}

void TUITextEdit::setPos(int x, int y)
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

void TUITextEdit::setSize(int w, int h)
{
    width = w;
    height = h;
    editField->setFixedHeight(40 * h);
}

void TUITextEdit::valueChanged()
{
    value = editField->toPlainText();
    covise::TokenBuffer tb;
    tb << ID;
    QByteArray ba = value.toUtf8();
    tb << ba.data();
    TUIMainWindow::getInstance()->send(tb);
}

/** Set activation state of this container and all its children.
  @param en true = elements enabled
*/
void TUITextEdit::setEnabled(bool en)
{
    TUIElement::setEnabled(en);
}

/** Set highlight state of this container and all its children.
  @param hl true = element highlighted
*/
void TUITextEdit::setHighlighted(bool hl)
{
    TUIElement::setHighlighted(hl);
}

char *TUITextEdit::getClassName()
{
    return (char *)"TUITextEdit";
}

bool TUITextEdit::isOfClassName(char *classname)
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

void TUITextEdit::setValue(int type, covise::TokenBuffer &tb)
{
    //cerr << "TUITextEdit::setValue info: type = " << type << endl;
    if (type == TABLET_STRING)
    {
        char *v;
        tb >> v;
        value = v;
        //cerr << "TUITextEdit::setValue " << value << endl;
        editField->setText(value);
    }
    TUIElement::setValue(type, tb);
}
