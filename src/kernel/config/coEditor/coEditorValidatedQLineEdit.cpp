/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// this subclass of QLineEdit checks if there is a validator attached to itself,
// Then prooves if the text in the lineEdit is valid for that validator. If not,
// it emits the notValid() signal.

#include "coEditorValidatedQLineEdit.h"

#include <QValidator>
#include <QKeyEvent>

coEditorValidatedQLineEdit::coEditorValidatedQLineEdit(QWidget *parent)
    : QLineEdit(parent)
{
}

void coEditorValidatedQLineEdit::focusOutEvent(QFocusEvent *e)
{

    if (isModified() && !isValid())
    {
        emit notValid();
    }
    //todo coEditorValidatedQLineEdit::focusOutEvent else branch, here we can focus back if we want that nasty behavior
    QLineEdit::focusOutEvent(e);
    emit focusOut();
}

void coEditorValidatedQLineEdit::keyPressEvent(QKeyEvent *e)
{
    if ((e->key() == Qt::Key_Enter || e->key() == Qt::Key_Return) && !isValid())
    {
        emit notValid();
    }
    QLineEdit::keyPressEvent(e);
    //    if ( e->key() == Qt::Key_Enter || e->key() == Qt::Key_Return )
}

bool coEditorValidatedQLineEdit::isValid()
{
    if (this->validator() != 0)
    {
        QString lineText = displayText();
        int lineCursorPos = cursorPosition();
        if (this->validator()->validate(lineText, lineCursorPos) == QValidator::Invalid || this->validator()->validate(lineText, lineCursorPos) == QValidator::Intermediate)
        {
            this->setStyleSheet("background-color: red");
            return false;
        }
    }
    this->setStyleSheet("background-color:");
    return true;
}

//adapt size of QLineEdit to the longest text it sees (longest text one entrywidget sees)
QSize coEditorValidatedQLineEdit::minimumSizeHint() const
{
    static int perfectSize = 140;
    if (perfectSize <= size().width())
    {
        //       if (text() != 0)
        perfectSize = text().length() * 6;
    }
    return QSize(perfectSize, size().height());
}
