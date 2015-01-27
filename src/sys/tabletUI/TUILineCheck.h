/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_LINEEDIT_H
#define CO_LINEEDIT_H

#include <QLineEdit>

class QFocusEvent;
class QKeyEvent;
class QString;
class QColor;

//================================================
class TUILineCheck : public QLineEdit
//================================================
{
    Q_OBJECT

public:
    TUILineCheck(QWidget *parent);

    QString currText;

signals:
    void focusChanged(bool);
    void contentChanged();

public slots:
    void checkContent();

protected:
    void focusInEvent(QFocusEvent *e);
    void focusOutEvent(QFocusEvent *e);
    void keyPressEvent(QKeyEvent *e);
};
#endif
