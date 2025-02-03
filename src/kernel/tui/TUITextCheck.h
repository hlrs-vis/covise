/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TEXTEDIT_H
#define CO_TEXTEDIT_H

#include <QTextEdit>

class QFocusEvent;
class QKeyEvent;
class QString;
class QColor;

//================================================
class TUITextCheck : public QTextEdit
//================================================
{
    Q_OBJECT

public:
    TUITextCheck(QWidget *parent);

    QString currText;

signals:
    void focusChanged(bool);
    void contentChanged();
    void editingFinished();
    void returnPressed();

public slots:
    void checkContent();

protected:
    void focusInEvent(QFocusEvent *e);
    void focusOutEvent(QFocusEvent *e);
    void keyPressEvent(QKeyEvent *e);
};
#endif
