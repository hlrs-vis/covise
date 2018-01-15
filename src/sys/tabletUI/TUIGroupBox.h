/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_UI_GROUPBOX_H
#define CO_UI_GROUPBOX_H

#include "TUIContainer.h"

class QWidget;

/** Basic Container
 * This class provides basic functionality and a
 * common interface to all Container elements.<BR>
 * The functionality implemented in this class represents a container
 * which arranges its children on top of each other.
 */
class TUIGroupBox : public TUIContainer
{
private:
public:
    TUIGroupBox(int id, int type, QWidget *w, int parent, QString name);
    virtual ~TUIGroupBox();
    virtual void setPos(int x, int y) override;
    virtual void setValue(TabletValue type, covise::TokenBuffer &) override;

    /// get the Element's classname
    virtual const char *getClassName() const override;
    virtual void setLabel(QString textl) override;

protected:
};
#endif
