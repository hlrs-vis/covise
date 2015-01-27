/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/13/2010
**
**************************************************************************/

#ifndef SIGNALTREEITEM_HPP
#define SIGNALTREEITEM_HPP

#include "sectiontreeitem.hpp"

class RoadTreeItem;
class Signal;

class SignalTreeItem : public SectionTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SignalTreeItem(RoadTreeItem *parent, Signal *section, QTreeWidgetItem *fosterParent);
    virtual ~SignalTreeItem();

    // SignalSignal //
    //
    Signal *getSignal() const
    {
        return signal_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
    virtual void updateName();

private:
    SignalTreeItem(); /* not allowed */
    SignalTreeItem(const SignalTreeItem &); /* not allowed */
    SignalTreeItem &operator=(const SignalTreeItem &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // PROPERTIES     //
    //################//

private:
    // Road //
    //
    Signal *signal_;
};

#endif // SIGNALTREEITEM_HPP
