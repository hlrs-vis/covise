/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QLineEdit>
#include <QLabel>
#include <QGridLayout>

#include <net/tokenbuffer.h>
#include "TUILineEdit.h"
#include "TUIApplication.h"
#include "TUIContainer.h"
#include "TUILineCheck.h"

/// Constructor
TUILineEdit::TUILineEdit(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
    editField = new TUILineCheck(w);
    connect(editField, SIGNAL(contentChanged()), this, SLOT(valueChangedSlot()));
    auto gl = createLayout(nullptr);
    gl->addWidget(editField, 1, 0, 1, width-1);
    gl->addWidget(editField, 1, width-1);
    for (int i=0; i<width-1; ++i)
        gl->setColumnStretch(i, 100);
    gl->setContentsMargins(0, 0, 0, 0);

    widgets.insert(editField);
}

/// Destructor
TUILineEdit::~TUILineEdit()
{
    delete editField;
}

void TUILineEdit::setPos(int x, int y)
{
    xPos = x;
    yPos = y;
    TUIContainer *parent;
    if ((parent = getParent()))
    {
        parent->addElementToLayout(this);
    }
    else
    {
        TUIMainWindow::getInstance()->addElementToLayout(this);
    }
    editField->setVisible(!hidden);
    if (label)
        label->setVisible(!hidden);
}

void TUILineEdit::setLabel(QString textl)
{
    TUIElement::setLabel(textl);
    if (textl.isEmpty())
    {
        widgets.erase(label);
        delete label;
        label = nullptr;
    }
    else if (!label)
    {
        label = new QLabel(editField->parentWidget());
        widgets.insert(label);
        label->setBuddy(editField);
        static_cast<QGridLayout *>(getLayout())->addWidget(label, 0, 0);
    }
    if (label)
        label->setText(textl);
}


void TUILineEdit::valueChangedSlot()
{
    valueChanged();
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

const char *TUILineEdit::getClassName() const
{
    return "TUILineEdit";
}

void TUILineEdit::setValue(TabletValue type, covise::TokenBuffer &tb)
{
    //cerr << "TUILineEdit::setValue info: type = " << type << endl;
    if (type == TABLET_STRING)
    {
        const char *v;
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
