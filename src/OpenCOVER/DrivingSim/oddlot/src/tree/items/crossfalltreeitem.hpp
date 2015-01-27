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

#ifndef CROSSFALLTREEITEM_HPP
#define CROSSFALLTREEITEM_HPP

#include "sectiontreeitem.hpp"

class RoadTreeItem;
class CrossfallSection;

class CrossfallSectionTreeItem : public SectionTreeItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit CrossfallSectionTreeItem(RoadTreeItem *parent, CrossfallSection *section, QTreeWidgetItem *fosterParent);
    virtual ~CrossfallSectionTreeItem();

    // Road //
    //
    CrossfallSection *getCrossfallSection() const
    {
        return crossfallSection_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

protected:
    virtual void updateName();

private:
    CrossfallSectionTreeItem(); /* not allowed */
    CrossfallSectionTreeItem(const CrossfallSectionTreeItem &); /* not allowed */
    CrossfallSectionTreeItem &operator=(const CrossfallSectionTreeItem &); /* not allowed */

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
    CrossfallSection *crossfallSection_;
};

#endif // CROSSFALLTREEITEM_HPP
