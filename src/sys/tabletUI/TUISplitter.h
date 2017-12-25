/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_UI_SPLITTER_H
#define CO_UI_SPLITTER_H

#include "TUIContainer.h"

/** Basic Container
 * This class provides basic functionality and a
 * common interface to all Container elements.<BR>
 * The functionality implemented in this class represents a container
 * which arranges its children on top of each other.
 */
class TUISplitter : public TUIContainer
{
private:
public:
    TUISplitter(int id, int type, QWidget *w, int parent, QString name);
    virtual ~TUISplitter();
    virtual void setPos(int x, int y);
    virtual void setValue(int type, covise::TokenBuffer &);

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

protected:
    QHBoxLayout *hBoxLayout;
    QVBoxLayout *vBoxLayout;
};
#endif
