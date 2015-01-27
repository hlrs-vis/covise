/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TUI_TOGGLEBUTTON_H
#define CO_TUI_TOGGLEBUTTON_H

#include <QObject>

#include "TUIElement.h"

/** Basic Container
 * This class provides basic functionality and a
 * common interface to all Container elements.<BR>
 * The functionality implemented in this class represents a container
 * which arranges its children on top of each other.
 */
class TUIToggleButton : public QObject, public TUIElement
{
    Q_OBJECT

public:
    TUIToggleButton(int id, int type, QWidget *w, int parent, QString name);
    virtual ~TUIToggleButton();
    virtual void setEnabled(bool en);
    virtual void setHighlighted(bool hl);
    virtual void setValue(int type, covise::TokenBuffer &);

    /// get the Element's classname
    virtual char *getClassName();
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(char *);

public slots:

    void valueChanged(bool pressed);
    void stateChanged(int);

protected:
};
#endif
