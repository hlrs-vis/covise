/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TUI_TEXT_EDIT_H
#define CO_TUI_TEXT_EDIT_H

#include <QObject>

#include "TUIElement.h"

class QTextEdit;
class QWidget;

class TUITextEdit : public QObject, public TUIElement
{
    Q_OBJECT

public:
    TUITextEdit(int id, int type, QWidget *w, int parent, QString name);
    virtual ~TUITextEdit();
    virtual void setValue(TabletValue type, covise::TokenBuffer &) override;

    /// get the Element's classname
    virtual const char *getClassName() const override;

    void setPos(int x, int y) override;
    void setSize(int w = 0, int h = 0) override;

public slots:
    void valueChanged();

protected:
    QTextEdit *editField;
    QString value;
};
#endif
