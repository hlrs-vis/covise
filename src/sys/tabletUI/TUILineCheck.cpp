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

#include "TUILineCheck.h"

//------------------------------------------------------------------------
//
//------------------------------------------------------------------------

TUILineCheck::TUILineCheck(QWidget *parent)
    : QLineEdit(parent)
{
    connect(this, SIGNAL(editingFinished()), this, SLOT(checkContent()));
    connect(this, SIGNAL(returnPressed()), this, SLOT(checkContent()));
}

//------------------------------------------------------------------------
// user has clicked into a cell
// change color
//------------------------------------------------------------------------
void TUILineCheck::keyPressEvent(QKeyEvent *e)
{
    QPalette palette;

    if (e->key() == Qt::Key_Return)
        palette.setBrush(foregroundRole(), Qt::black);
    else
        palette.setBrush(foregroundRole(), Qt::red);

    QLineEdit::keyPressEvent(e); // hand on event to base class
    palette.setBrush(foregroundRole(), Qt::red);
    setPalette(palette);
}

//------------------------------------------------------------------------
// user has clicked into a cell
// store text and tell the port that a new cell has been activated
// hilight cell
//------------------------------------------------------------------------
void TUILineCheck::focusInEvent(QFocusEvent *e)
{
    currText = text();
    emit focusChanged(true);

    QLineEdit::focusInEvent(e);
}

//------------------------------------------------------------------------
// user has set the focus to another cell
// reset the cell frame style
// tell the port that the text has changed & deactivate cell label
//------------------------------------------------------------------------
void TUILineCheck::focusOutEvent(QFocusEvent *e)
{
    emit focusChanged(false);

    QLineEdit::focusOutEvent(e);
}

//------------------------------------------------------------------------
//------------------------------------------------------------------------
void TUILineCheck::checkContent()
{
    QPalette palette;
    palette.setBrush(foregroundRole(), Qt::black);
    setPalette(palette);
    if (currText != text())
    {
        currText = text();
        emit contentChanged();
    }
}
