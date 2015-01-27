/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TUI_POPUP_H
#define CO_TUI_POPUP_H

#include <QObject>
#include <QWidget>

#include "TUIElement.h"

class QTextEdit;
class QPushButton;
class QFrame;

class TUIPopUp : public QObject, public TUIElement
{
    Q_OBJECT

public:
    TUIPopUp(int id, int type, QWidget *w, int parent, QString name);
    virtual ~TUIPopUp();
    virtual void setValue(int type, covise::TokenBuffer &);
    virtual void setEnabled(bool en);
    virtual void setHighlighted(bool hl);

    /// get the Element's classname
    virtual char *getClassName();
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(char *);

public slots:
    void popupButtonClicked();

protected:
    QFrame *popup;
    QTextEdit *textEdit;
    QPushButton *popupButton, *closeButton;
    QString value;
};
#endif
