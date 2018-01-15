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

#include "roadtreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/typesection.hpp"
#include "src/data/roadsystem/track/trackcomponent.hpp"
#include "src/data/roadsystem/sections/elevationsection.hpp"
#include "src/data/roadsystem/sections/superelevationsection.hpp"
#include "src/data/roadsystem/sections/crossfallsection.hpp"
#include "src/data/roadsystem/sections/shapesection.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/signalobject.hpp"
#include "src/data/roadsystem/sections/objectobject.hpp"
#include "src/data/roadsystem/sections/bridgeobject.hpp"

// Tree //
//
#include "roadsystemtreeitem.hpp"

#include "typesectiontreeitem.hpp"
#include "trackcomponenttreeitem.hpp"
#include "elevationsectiontreeitem.hpp"
#include "superelevationtreeitem.hpp"
#include "crossfalltreeitem.hpp"
#include "shapetreeitem.hpp"
#include "lanesectiontreeitem.hpp"
#include "objecttreeitem.hpp"
#include "crosswalktreeitem.hpp"
#include "signaltreeitem.hpp"
#include "sensortreeitem.hpp"
#include "bridgetreeitem.hpp"

RoadTreeItem::RoadTreeItem(RoadSystemTreeItem *parent, RSystemElementRoad *road, QTreeWidgetItem *fosterParent)
    : ProjectTreeItem(parent, road, fosterParent)
    , roadSystemTreeItem_(parent)
    , road_(road)
{
    init();
}

RoadTreeItem::~RoadTreeItem()
{
}

void
RoadTreeItem::init()
{
    // Text //
    //
    updateName();

    // TypeSections //
    //
    typesItem_ = new QTreeWidgetItem(this);
    typesItem_->setText(0, tr("types"));

    foreach (TypeSection *element, road_->getTypeSections())
    {
        new TypeSectionTreeItem(this, element, typesItem_);
    }

    // Objectss //
    //
    objectsItem_ = new QTreeWidgetItem(this);
    objectsItem_->setText(0, tr("objects"));

    foreach (Object *element, road_->getObjects())
    {
        new ObjectTreeItem(this, element, objectsItem_);
    }
    foreach (Crosswalk *element, road_->getCrosswalks())
    {
        new CrosswalkTreeItem(this, element, objectsItem_);
    }

    // Bridges //
    //
    bridgesItem_ = new QTreeWidgetItem(this);
    bridgesItem_->setText(0, tr("bridges and tunnels"));
    foreach (Bridge *element, road_->getBridges())
    {
        new BridgeTreeItem(this, element, bridgesItem_);
    }

    // Signals //
    //
    signalsItem_ = new QTreeWidgetItem(this);
    signalsItem_->setText(0, tr("signals"));
    foreach (Signal *element, road_->getSignals())
    {
        new SignalTreeItem(this, element, signalsItem_);
    }
    // Sensors //
    //
    sensorsItem_ = new QTreeWidgetItem(this);
    sensorsItem_->setText(0, tr("sensors"));
    foreach (Sensor *element, road_->getSensors())
    {
        new SensorTreeItem(this, element, sensorsItem_);
    }

    // Tracks //
    //
    tracksItem_ = new QTreeWidgetItem(this);
    tracksItem_->setText(0, tr("tracks"));

    foreach (TrackComponent *element, road_->getTrackSections())
    {
        new TrackComponentTreeItem(this, element, tracksItem_);
    }

    // ElevationSections //
    //
    elevationsItem_ = new QTreeWidgetItem(this);
    elevationsItem_->setText(0, tr("elevations"));

    foreach (ElevationSection *element, road_->getElevationSections())
    {
        new ElevationSectionTreeItem(this, element, elevationsItem_);
    }

    // SuperelevationSections //
    //
    superelevationsItem_ = new QTreeWidgetItem(this);
    superelevationsItem_->setText(0, tr("superelevations"));

    foreach (SuperelevationSection *element, road_->getSuperelevationSections())
    {
        new SuperelevationSectionTreeItem(this, element, superelevationsItem_);
    }

    // CrossfallSections //
    //
    crossfallsItem_ = new QTreeWidgetItem(this);
    crossfallsItem_->setText(0, tr("crossfalls"));

    foreach (CrossfallSection *element, road_->getCrossfallSections())
    {
        new CrossfallSectionTreeItem(this, element, crossfallsItem_);
    }

	// RoadShapeSections //
	//
	shapesItem_ = new QTreeWidgetItem(this);
	shapesItem_->setText(0, tr("shapes"));

	foreach(ShapeSection *element, road_->getShapeSections())
	{
		new ShapeSectionTreeItem(this, element, shapesItem_);
	}

    // LaneSections //
    //
    lanesItem_ = new QTreeWidgetItem(this);
    lanesItem_->setText(0, tr("lanes"));

    foreach (LaneSection *element, road_->getLaneSections())
    {
        new LaneSectionTreeItem(this, element, lanesItem_);
    }
}

void
RoadTreeItem::updateName()
{
    QString text = road_->getName();
    text.append(" (");
    text.append(road_->getID().speakingName());
    text.append(")");

    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
RoadTreeItem::updateObserver()
{
    // Parent //
    //
    ProjectTreeItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // RSystemElement //
    //
    int changes = road_->getRSystemElementChanges();

    if ((changes & RSystemElement::CRE_NameChange)
        || (changes & RSystemElement::CRE_IdChange))
    {
        updateName();
    }

    // RSystemElementRoad //
    //
    changes = road_->getRoadChanges();

    if (changes & RSystemElementRoad::CRD_TypeSectionChange)
    {
        foreach (TypeSection *element, road_->getTypeSections())
        {
            if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new TypeSectionTreeItem(this, element, typesItem_);
            }
        }
    }

    if (changes & RSystemElementRoad::CRD_TrackSectionChange)
    {
        foreach (TrackComponent *element, road_->getTrackSections())
        {
            if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new TrackComponentTreeItem(this, element, tracksItem_);
            }
        }
    }

    if (changes & RSystemElementRoad::CRD_ElevationSectionChange)
    {
        foreach (ElevationSection *element, road_->getElevationSections())
        {
            if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new ElevationSectionTreeItem(this, element, elevationsItem_);
            }
        }
    }

    if (changes & RSystemElementRoad::CRD_SuperelevationSectionChange)
    {
        foreach (SuperelevationSection *element, road_->getSuperelevationSections())
        {
            if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new SuperelevationSectionTreeItem(this, element, superelevationsItem_);
            }
        }
    }

    if (changes & RSystemElementRoad::CRD_CrossfallSectionChange)
    {
        foreach (CrossfallSection *element, road_->getCrossfallSections())
        {
            if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new CrossfallSectionTreeItem(this, element, crossfallsItem_);
            }
        }
    }

	if (changes & RSystemElementRoad::CRD_ShapeSectionChange)
	{
		foreach(ShapeSection *element, road_->getShapeSections())
		{
			if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
				|| (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
			{
				new ShapeSectionTreeItem(this, element, shapesItem_);
			}
		}
	}

    if (changes & RSystemElementRoad::CRD_LaneSectionChange)
    {
        foreach (LaneSection *element, road_->getLaneSections())
        {
            if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new LaneSectionTreeItem(this, element, lanesItem_);
            }
        }
    }

    if (changes & RSystemElementRoad::CRD_SignalChange)
    {
        foreach (Signal *element, road_->getSignals())
        {
            if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new SignalTreeItem(this, element, signalsItem_);
            }
        }
    }
    if (changes & RSystemElementRoad::CRD_ObjectChange)
    {
        foreach (Object *element, road_->getObjects())
        {
            if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new ObjectTreeItem(this, element, objectsItem_);
            }
        }
    }
    if (changes & RSystemElementRoad::CRD_BridgeChange)
    {
        foreach (Bridge *element, road_->getBridges())
        {
            if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new BridgeTreeItem(this, element, bridgesItem_);
            }
        }
    }
}
