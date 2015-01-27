/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/6/2010
**
**************************************************************************/

#ifndef PROJECTTREEITEM_HPP
#define PROJECTTREEITEM_HPP

#include <QObject>
#include <QTreeWidgetItem>
#include "src/data/observer.hpp"

class DataElement;
class ProjectTree;

class ProjectTreeItem : public QObject, public QTreeWidgetItem, public Observer
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ProjectTreeItem(ProjectTreeItem *parent, DataElement *dataElement, QTreeWidgetItem *fosterParent = NULL);
    virtual ~ProjectTreeItem();

    virtual void setData(int column, int role, const QVariant &value);

    // Tree //
    //
    virtual ProjectTree *getProjectTree() const;
    bool isDescendantOf(ProjectTreeItem *projectTreeItem);

    // Garbage //
    //
    void registerForDeletion();
    virtual void notifyDeletion(); // to be implemented by subclasses
    bool isInGarbage() const
    {
        return isInGarbage_;
    }

    // DataElement //
    //
    DataElement *getDataElement() const
    {
        return dataElement_;
    }

    // ParentGraphElement //
    //
    ProjectTreeItem *getParentProjectTreeItem() const
    {
        return parentProjectTreeItem_;
    }

    // Observer Pattern //
    //
    virtual void updateObserver();

protected:
private:
    ProjectTreeItem(); /* not allowed */
    ProjectTreeItem(const ProjectTreeItem &); /* not allowed */
    ProjectTreeItem &operator=(const ProjectTreeItem &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // PROPERTIES     //
    //################//

private:
    // ParentGraphElement //
    //
    ProjectTreeItem *parentProjectTreeItem_;

    // DataElement //
    //
    DataElement *dataElement_;

    // Garbage //
    //
    bool isInGarbage_;

    //################//
    // STATIC         //
    //################//
};

#endif // PROJECTTREEITEM_HPP
