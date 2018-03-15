/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   12.03.2010
**
**************************************************************************/

#ifndef OSCBASEITEM_HPP
#define OSCBASEITEM_HPP

#include "src/graph/items/svgelement.hpp"

namespace OpenScenario
{
class oscCatalog;
class oscActions;
class oscObject;
class oscCatalogs;
class oscPrivateAction;
class oscPrivate;
}

class TopviewGraph;
class OSCItem;
class OSCBase;
class RoadSystem;
class OSCRoadSystemItem;

class OSCBaseItem : public SVGElement
{

    //################//
    // FUNCTIONS      //
    //################//

public:
	explicit OSCBaseItem(TopviewGraph *topviewGraph, OSCBase *oscBase);
    virtual ~OSCBaseItem();

	void init();

	 // Graph //
    //
    virtual TopviewGraph *getTopviewGraph() const
    {
        return topviewGraph_;
    }

	// RoadSystemItem //
	//
	OSCRoadSystemItem *getRoadSystemItem()
	{
		return oscRoadSystemItem_;
	}

	void setRoadSystemItem(OSCRoadSystemItem *roadSystemItem)
	{
		oscRoadSystemItem_ = roadSystemItem;
	}

	 // OSCItems //
    //
    void appendOSCItem(OSCItem *oscItem);
    bool removeOSCItem(OSCItem *oscObjectItem);
    OSCItem *getOSCItem(const QString &id) const
    {
        return oscItems_.value(id, NULL);
    }
    QMap<QString, OSCItem *> getOSCItems() const
    {
        return oscItems_;
    }


    // delete this item
    virtual bool deleteRequest()
    {
        return false;
    };

    // Garbage //
    //
    //	virtual void			notifyDeletion();

    // Obsever Pattern //
    //
    virtual void updateObserver();

 
    //################//
    // PROPERTIES     //
    //################//

private:
	OSCBase *oscBase_;
	OpenScenario::oscActions *actions_;
	OpenScenario::oscCatalogs *catalogs_;

	OpenScenario::oscPrivateAction *getPrivateAction(OpenScenario::oscObject *object, OpenScenario::oscPrivate *privateObject);
	OpenScenario::oscCatalog *getCatalog(OpenScenario::oscObject *object);

	 // OSCObjectItems //
    //
    QMap<QString, OSCItem *> oscItems_;

	RoadSystem *roadSystem_;
	OSCRoadSystemItem *oscRoadSystemItem_;

	TopviewGraph *topviewGraph_;

};

#endif // OSCITEM_HPP
