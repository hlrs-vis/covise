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
    virtual void setValue(TabletValue type, covise::TokenBuffer &) override;

    /// get the Element's classname
    virtual const char *getClassName() const override;

public slots:
    void popupButtonClicked();

protected:
    QFrame *popup;
    QTextEdit *textEdit;
    QPushButton *popupButton, *closeButton;
    QString value;
};
#endif
