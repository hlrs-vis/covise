/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   17.03.2010
**
**************************************************************************/

#ifndef LATERALSECTIONITEM_HPP
#define LATERALSECTIONITEM_HPP

#include <QObject>
#include <QGraphicsPathItem>
#include "src/data/observer.hpp"

class SectionItem;
class LateralSection;
class GraphElement;
class ProjectGraph;
class ProfileGraph;

class LateralSectionItem : public QObject, public QGraphicsPathItem, public Observer
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LateralSectionItem(GraphElement *parentItem, LateralSection *lateralSection);
    virtual ~LateralSectionItem();


	GraphElement *getParentSectionItem() const
    {
        return parentItem_;
    }

    // Section //
    //
	LateralSection *getLateralSection() const
    {
        return lateralSection_;
    }

	// Graph //
	//
	ProjectGraph *getProjectGraph() const;
	virtual ProfileGraph *getProfileGraph() const;

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest();
	void registerForDeletion();
	virtual void notifyDeletion(); // to be implemented by subclasses
	bool isInGarbage() const
	{
		return isInGarbage_;
	}

private:
    LateralSectionItem(); /* not allowed */
    LateralSectionItem(const LateralSectionItem &); /* not allowed */
    LateralSectionItem &operator=(const LateralSectionItem &); /* not allowed */

    void init();

    //################//
    // SLOTS          //
    //################//

public slots:
    virtual bool removeSection() = 0;

    //################//
    // EVENTS         //
    //################//

protected:

    //################//
    // PROPERTIES     //
    //################//

protected:
    // GraphElement //
    //
	GraphElement *parentItem_;


private:
    // Section //
    //
    LateralSection *lateralSection_;

	// Garbage //
	//
	bool isInGarbage_;
};

#endif // LATERALSECTIONITEM_HPP
