/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_UI_CONTAINER_H
#define CO_UI_CONTAINER_H

#include "TUIElement.h"
#include <list>

class QGridLayout;

/** Basic Container
 * This class provides basic functionality and a
 * common interface to all Container elements.<BR>
 * The functionality implemented in this class represents a container
 * which arranges its children on top of each other.
 */
class TUIContainer : public TUIElement
{
private:
public:
    TUIContainer(int id, int type, QWidget *w, int parent, QString name);
    virtual ~TUIContainer();
    virtual void setEnabled(bool en) override;
    virtual void setHighlighted(bool hl) override;
    virtual void addElementToLayout(TUIElement *el);
    virtual void addElement(TUIElement *el);
    virtual void removeElement(TUIElement *el);
    virtual void showElement(TUIElement *el);
    void removeAllChildren();

    /// get the Element's classname
    virtual const char *getClassName() const override;

    QGridLayout *gridLayout() const;

protected:
    /// List of children elements
    typedef std::list<TUIElement *> ElementList;
    ElementList elements;
};
#endif
