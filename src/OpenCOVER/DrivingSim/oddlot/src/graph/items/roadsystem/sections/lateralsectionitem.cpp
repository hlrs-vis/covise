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

#include "lateralsectionitem.hpp"

// Data //
//
#include "src/data/roadsystem/lateralsections/lateralsection.hpp"

// Graph //
//
#include "src/graph/items/roadsystem/sections/sectionitem.hpp"
#include "src/graph/profilegraph.hpp"

// Qt //
//
#include <QCursor>
#include <QGraphicsSceneHoverEvent>

//################//
// CONSTRUCTOR    //
//################//

LateralSectionItem::LateralSectionItem(GraphElement *parentItem, LateralSection *lateralSection)
    : QObject()
	, QGraphicsPathItem(parentItem)
	, Observer()
    , parentItem_(parentItem)
    , lateralSection_(lateralSection)
	, isInGarbage_(false)
{
    init();
}

LateralSectionItem::~LateralSectionItem()
{
    // Observer Pattern //
    //
	lateralSection_->detachObserver(this);
}

void
LateralSectionItem::init()
{

    // Observer Pattern //
    //
	lateralSection_->attachObserver(this);

	setZValue(0.0);


    // ContextMenu //
    //

/*    QAction *removeSectionAction = getRemoveMenu()->addAction(tr("Section"));
    connect(removeSectionAction, SIGNAL(triggered()), this, SLOT(removeSection())); */
}

void
LateralSectionItem::registerForDeletion()
{
	if (parentItem_)
	{
		ProfileGraph *profileGraph = parentItem_->getProfileGraph();
		if (profileGraph)
		{
			profileGraph->addToGarbage(this);
		}
	}

	notifyDeletion();
}

void
LateralSectionItem::notifyDeletion()
{
	isInGarbage_ = true;
}

ProfileGraph *
LateralSectionItem::getProfileGraph() const
{
	if (parentItem_)
	{
		return parentItem_->getProfileGraph();
	}
	else
	{
		return NULL;
	}
}

ProjectGraph *
LateralSectionItem::getProjectGraph() const
{
    if (getProfileGraph())
	{
		return getProfileGraph();
	}
	else
	{
		return NULL;
	}
}


//################//
// SLOTS          //
//################//



/*
bool
	LateralSectionItem
	::removeSection()
{
	// does nothing by default - to be implemented by subclasses
}*/


//################//
// OBSERVER       //
//################//

void
LateralSectionItem::updateObserver()
{

	// Get change flags //
	//
	int changes = lateralSection_->getDataElementChanges();

	// Deletion //
	//
	if ((changes & DataElement::CDE_DataElementDeleted)
		|| (changes & DataElement::CDE_DataElementRemoved))
	{
		registerForDeletion();
		return;
	}
}

//*************//
// Delete Item
//*************//

bool
LateralSectionItem::deleteRequest()
{
    if (removeSection())
    {
        return true;
    }

    return false;
}


