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

#include "src/graph/items/graphelement.hpp"

class TopviewGraph;
class OSCObjectItem;
class OSCBase;
class RoadSystem;
class OSCRoadSystemItem;

class OSCBaseItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
	explicit OSCBaseItem(TopviewGraph *topviewGraph, OSCBase *oscBase);
    virtual ~OSCBaseItem();

	void init();

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

	 // OSCObjectItems //
    //
    void appendOSCObjectItem(OSCObjectItem *oscObjectItem);
    bool removeOSCObjectItem(OSCObjectItem *oscObjectItem);
    OSCObjectItem *getOSCObjectItem(const QString &id) const
    {
        return oscObjectItems_.value(id, NULL);
    }
    QMap<QString, OSCObjectItem *> getOSCObjectItems() const
    {
        return oscObjectItems_;
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

	 // OSCObjectItems //
    //
    QMap<QString, OSCObjectItem *> oscObjectItems_;

	RoadSystem *roadSystem_;
	OSCRoadSystemItem *oscRoadSystemItem_;

	TopviewGraph *topviewGraph_;

};

#endif // OSCITEM_HPP
