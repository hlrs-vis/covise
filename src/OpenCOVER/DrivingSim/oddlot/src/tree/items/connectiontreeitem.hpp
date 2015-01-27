/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/11/2010
**
**************************************************************************/

#ifndef CONNECTIONTREEITEM_HPP
#define CONNECTIONTREEITEM_HPP

#include "projecttreeitem.hpp"

class JunctionTreeItem;
class JunctionConnection;

class ConnectionTreeItem : public ProjectTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ConnectionTreeItem(JunctionTreeItem *parent, JunctionConnection *connection, QTreeWidgetItem *fosterParent);
    virtual ~ConnectionTreeItem();

    // Connection //
    //
    JunctionConnection *getConnection() const
    {
        return connection_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
private:
    ConnectionTreeItem(); /* not allowed */
    ConnectionTreeItem(const ConnectionTreeItem &); /* not allowed */
    ConnectionTreeItem &operator=(const ConnectionTreeItem &); /* not allowed */

    void init();

    void updateName();

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // PROPERTIES     //
    //################//

private:
    // Parent //
    //
    JunctionTreeItem *junctionTreeItem_;

    // Connection //
    //
    JunctionConnection *connection_;
};

#endif // CONNECTIONTREEITEM_HPP
