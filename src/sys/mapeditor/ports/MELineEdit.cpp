/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QUrl>
#include <QDebug>
#include <QDropEvent>

#include "MELineEdit.h"
#include "controlPanel/MEControlParameterLine.h"

/*!
    \class MELineEdit
    \brief Extensions for line editing
*/

MELineEdit::MELineEdit(MEControlParameterLine *parent)
    : QLineEdit(parent)
    , m_frame(NULL)
{
    connect(this, SIGNAL(editingFinished()), this, SLOT(checkContent()));
    connect(this, SIGNAL(returnPressed()), this, SLOT(checkContent()));
}

MELineEdit::MELineEdit(QFrame *parent)
    : QLineEdit(parent)
    , m_frame(parent)
{
    connect(this, SIGNAL(editingFinished()), this, SLOT(checkContent()));
    connect(this, SIGNAL(returnPressed()), this, SLOT(checkContent()));
}

MELineEdit::MELineEdit(QWidget *parent)
    : QLineEdit(parent)
    , m_frame(NULL)
{
    connect(this, SIGNAL(editingFinished()), this, SLOT(checkContent()));
    connect(this, SIGNAL(returnPressed()), this, SLOT(checkContent()));
}

MELineEdit::MELineEdit(QFrame *frame, QWidget *parent)
    : QLineEdit(parent)
    , m_frame(frame)
{
    connect(this, SIGNAL(editingFinished()), this, SLOT(checkContent()));
    connect(this, SIGNAL(returnPressed()), this, SLOT(checkContent()));
}

//!
//! Change text color
//!
void MELineEdit::keyPressEvent(QKeyEvent *e)
{
    QPalette palette;
    palette.setBrush(foregroundRole(), Qt::red);
    setPalette(palette);
    QLineEdit::keyPressEvent(e); // hand on event to base class
    e->accept();
}

//!
//! Store current text
//!
void MELineEdit::focusInEvent(QFocusEvent *e)
{
    m_currText = text();
    emit focusChanged(true);

    QLineEdit::focusInEvent(e);
}

//!
//! Reset the cell frame style
//!
void MELineEdit::focusOutEvent(QFocusEvent *e)
{
    emit focusChanged(false);

    QLineEdit::focusOutEvent(e);
}

void MELineEdit::checkContent()
{
    QPalette palette;
    palette.setBrush(foregroundRole(), Qt::black);
    setPalette(palette);
    if (m_currText != text())
    {
        m_currText = text();
        emit contentChanged(m_currText);
    }
}

/*!
    \class MECheckBox
    \brief Extensions for check boxes
*/

MECheckBox::MECheckBox(MEControlParameterLine *parent)
    : QCheckBox(parent)
{
    setFocusPolicy(Qt::StrongFocus);
}

MECheckBox::MECheckBox(QWidget *parent)
    : QCheckBox(parent)
{
    setFocusPolicy(Qt::StrongFocus);
}

void MECheckBox::focusInEvent(QFocusEvent *e)
{
    emit focusChanged(true);
    QCheckBox::focusInEvent(e);
}

void MECheckBox::focusOutEvent(QFocusEvent *e)
{
    emit focusChanged(false);
    QCheckBox::focusOutEvent(e);
}

/*!
    \class MEComboBox
    \brief Extensions for combo boxes
*/

MEComboBox::MEComboBox(MEControlParameterLine *parent)
    : QComboBox(parent)
{
    setFocusPolicy(Qt::WheelFocus);
}

MEComboBox::MEComboBox(QWidget *parent)
    : QComboBox(parent)
{
    setFocusPolicy(Qt::WheelFocus);
}

void MEComboBox::focusInEvent(QFocusEvent *e)
{
    emit focusChanged(true);
    QComboBox::focusInEvent(e);
}

void MEComboBox::focusOutEvent(QFocusEvent *e)
{
    emit focusChanged(false);
    QComboBox::focusOutEvent(e);
}
