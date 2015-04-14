/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_UI_TAB_H
#define CO_UI_TAB_H

#include "TUIContainer.h"

class QWidget;

/** Basic Container
 * This class provides basic functionality and a
 * common interface to all Container elements.<BR>
 * The functionality implemented in this class represents a container
 * which arranges its children on top of each other.
 */
class TUITab : public TUIContainer
{
private:
public:
    TUITab(int id, int type, QWidget *w, int parent, QString name);
    virtual ~TUITab();
    virtual void setPos(int x, int y);

    /// get the Element's classname
    virtual char *getClassName();
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(char *);
    virtual void deActivate(TUITab *activedTab);
    virtual void setValue(int type, covise::TokenBuffer &tb);
    virtual void setHidden(bool hide);

    virtual void activated();

    virtual void setLabel(QString textl);

protected:
    int firstTime;
};
#endif
