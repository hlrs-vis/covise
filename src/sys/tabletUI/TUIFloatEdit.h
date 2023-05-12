/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TUI_FLOAT_EDIT_H
#define CO_TUI_FLOAT_EDIT_H

#include "TUILineEdit.h"

class QLineEdit;

class TUIFloatEdit : public TUILineEdit
{
public:
    TUIFloatEdit(int id, int type, QWidget *w, int parent, QString name);
    void setValue(TabletValue type, covise::TokenBuffer &) override;
    const char *getClassName() const override;
    void valueChanged() override;

    float value;

};
#endif
