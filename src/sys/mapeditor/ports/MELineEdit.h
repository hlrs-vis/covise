/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_LINEEDIT_H
#define ME_LINEEDIT_H

#include <QLineEdit>
#include <QCheckBox>
#include <QComboBox>

class QFocusEvent;
class QFrame;
class QKeyEvent;
class QDropEvent;
class QString;
class QColor;

class MEModuleParameterLine;
class MEControlParameterLine;

//================================================
class MELineEdit : public QLineEdit
//================================================
{
    Q_OBJECT

public:
    MELineEdit(MEControlParameterLine *parent);
    MELineEdit(QFrame *parent);
    MELineEdit(QWidget *parent = 0);
    MELineEdit(QFrame *parent, QWidget *widget);

    QString m_currText;
    QFrame *m_frame;

signals:

    void focusChanged(bool);
    void contentChanged(const QString &);

public slots:

    void checkContent();

protected:
    void focusInEvent(QFocusEvent *e);
    void focusOutEvent(QFocusEvent *e);
    void keyPressEvent(QKeyEvent *e);
};

//================================================
class MECheckBox : public QCheckBox
//================================================
{
    Q_OBJECT

public:
    MECheckBox(MEControlParameterLine *parent);
    MECheckBox(QWidget *parent = 0);

signals:

    void focusChanged(bool);

protected:
    void focusInEvent(QFocusEvent *e);
    void focusOutEvent(QFocusEvent *e);
};

//================================================
class MEComboBox : public QComboBox
//================================================
{
    Q_OBJECT

public:
    MEComboBox(MEControlParameterLine *parent);
    MEComboBox(QWidget *parent = 0);

signals:

    void focusChanged(bool);

protected:
    void focusInEvent(QFocusEvent *e);
    void focusOutEvent(QFocusEvent *e);
};
#endif
