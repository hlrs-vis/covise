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

#ifndef OSCBASESHAPEITEM_HPP
#define OSCBASESHAPEITEM_HPP

#include "src/graph/items/graphelement.hpp"

namespace OpenScenario
{
class oscCatalog;
class oscActions;
class oscObject;
class oscCatalogs;
class oscPrivateAction;
class oscPrivate;
}

class OSCShapeItem;
class OSCBase;


class OSCBaseShapeItem : public GraphElement
{

    //################//
    // FUNCTIONS      //
    //################//

public:
	explicit OSCBaseShapeItem(TopviewGraph *topviewGraph, OSCBase *oscBase);
    virtual ~OSCBaseShapeItem();

	void init();

	// Graph //
	//
	virtual TopviewGraph *getTopviewGraph() const
	{
		return topviewGraph_;
	}

     // OSCShapeItems //
    //
    void appendOSCShapeItem(OSCShapeItem *oscShapeItem);
    bool removeOSCShapeItem(OSCShapeItem *oscShapeItem);

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
	TopviewGraph *topviewGraph_;

	 // OSCObjectShapeItems //
    //
    QMap<QString, OSCShapeItem *> oscShapeItems_;

};

#endif // OSCBASESHAPEITEM_HPP
