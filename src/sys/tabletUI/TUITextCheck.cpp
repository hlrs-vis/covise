/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <QFrame>
#include <QKeyEvent>
#include <QFocusEvent>
#include <QUrl>
#include <QComboBox>
#include <QDrag>
#include <QMimeData>

#include "TUITextCheck.h"

//------------------------------------------------------------------------
//
//------------------------------------------------------------------------

TUITextCheck::TUITextCheck(QWidget *parent)
    : QTextEdit(parent)
{
    setAcceptRichText(false);
    setTabChangesFocus(true);

    connect(this, SIGNAL(editingFinished()), this, SLOT(checkContent()));
    connect(this, SIGNAL(returnPressed()), this, SLOT(checkContent()));
}

//------------------------------------------------------------------------
// user has clicked into a cell
// change color
//------------------------------------------------------------------------
void TUITextCheck::keyPressEvent(QKeyEvent *e)
{
    QPalette palette;

    if ((e->key() == Qt::Key_Return) || (e->key() == Qt::Key_Enter))
    {
        palette.setBrush(QPalette::WindowText, Qt::black);
#if QT_VERSION >= 0x050000
        emit returnPressed();
#endif
    }
    else
    {
        palette.setBrush(QPalette::WindowText, Qt::red);
    }

    QTextEdit::keyPressEvent(e); // hand on event to base class
    //palette.setBrush(foregroundRole(), Qt::red);
    setPalette(palette);
}

//------------------------------------------------------------------------
// user has clicked into a cell
// store text and tell the port that a new cell has been activated
// hilight cell
//------------------------------------------------------------------------
void TUITextCheck::focusInEvent(QFocusEvent *e)
{
    currText = toPlainText();
    emit focusChanged(true);

    QTextEdit::focusInEvent(e);
}

//------------------------------------------------------------------------
// user has set the focus to another cell
// reset the cell frame style
// tell the port that the text has changed & deactivate cell label
//------------------------------------------------------------------------
void TUITextCheck::focusOutEvent(QFocusEvent *e)
{
#if QT_VERSION >= 0x050000
    emit editingFinished();
#endif
    emit focusChanged(false);

    QTextEdit::focusOutEvent(e);
}

//------------------------------------------------------------------------
//------------------------------------------------------------------------
void TUITextCheck::checkContent()
{
    QPalette palette;
    palette.setBrush(foregroundRole(), Qt::black);
    setPalette(palette);
    if (currText != toPlainText())
    {
        currText = toPlainText();
        emit contentChanged();
    }
}
