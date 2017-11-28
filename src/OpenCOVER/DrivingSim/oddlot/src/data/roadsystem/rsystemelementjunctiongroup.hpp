/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   02.02.2010
**
**************************************************************************/

#ifndef RSYSTEMELEMENTJUNCTIONGROUP_HPP
#define RSYSTEMELEMENTJUNCTIONGROUP_HPP

#include "roadsystem.hpp"


//####################//
// JunctionElement    //
//####################//

class RSystemElementJunctionGroup : public RSystemElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum JunctionGroupChange
    {
        CJG_ConnectionChanged = 0x1
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RSystemElementJunctionGroup(const QString &name, const QString &id, const QString &type);
    virtual ~RSystemElementJunctionGroup();

	QString getType()
	{
		return type_;
	}

	void setType(const QString &type)
	{
		type_ = type;
	}

	void addJunction(const QString junctionReference);
	bool delJunction(const QString junctionReference);
	QList<QString> getJunctionReferences()
	{
		return junctionReferences_;
	}

	void clearReferences();

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getJunctionGroupChanges() const
    {
        return junctionGroupChanges_;
    }
    void addJunctionGroupChanges(int changes);

    // Prototype Pattern //
    //
    RSystemElementJunctionGroup *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    RSystemElementJunctionGroup(); /* not allowed */
    RSystemElementJunctionGroup(const RSystemElementJunctionGroup &); /* not allowed */
    RSystemElementJunctionGroup &operator=(const RSystemElementJunctionGroup &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change //
    //
    int junctionGroupChanges_;


	QString type_;

    // JunctionReferences //
    //
    QList<QString> junctionReferences_; // owned
};

#endif // RSYSTEMELEMENTJUNCTIONGROUP_HPP
