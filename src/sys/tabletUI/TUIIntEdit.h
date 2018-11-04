/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TUI_INT_EDIT_H
#define CO_TUI_INT_EDIT_H

#include <QObject>

#include "TUIElement.h"

class QWidget;
class QLineEdit;
class QIntValidator;

class TUIIntEdit : public QObject, public TUIElement
{
    Q_OBJECT

public:
    TUIIntEdit(int id, int type, QWidget *w, int parent, QString name);
    ~TUIIntEdit();
    virtual void setValue(TabletValue type, covise::TokenBuffer &) override;

    /// get the Element's classname
    virtual const char *getClassName() const override;

    void setPos(int x, int y) override;

    QLineEdit *intEdit;
    QIntValidator *validator;

    int min;
    int max;
    int value;

public slots:

    void valueChanged();

protected:
};
#endif
