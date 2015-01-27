/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_UI_FRAME_H
#define CO_UI_FRAME_H

#include "TUIContainer.h"

class QWidget;

/** Basic Container
 * This class provides basic functionality and a
 * common interface to all Container elements.<BR>
 * The functionality implemented in this class represents a container
 * which arranges its children on top of each other.
 */
class TUIFrame : public TUIContainer
{
private:
public:
    TUIFrame(int id, int type, QWidget *w, int parent, QString name);
    virtual ~TUIFrame();
    virtual void setPos(int x, int y);
    virtual void setValue(int type, covise::TokenBuffer &);

    /// get the Element's classname
    virtual char *getClassName();
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(char *);

protected:
};
#endif
