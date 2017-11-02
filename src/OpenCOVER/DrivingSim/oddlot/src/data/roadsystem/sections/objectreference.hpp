/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   22.03.2010
**
**************************************************************************/

#ifndef OBJECTREFERENCE_HPP
#define OBJECTREFERENCE_HPP

#include "src/data/roadsystem/sections/roadsection.hpp"
#include "src/data/roadsystem/sections/signalobject.hpp"
#include "src/data/roadsystem/sections/objectobject.hpp"

class ObjectReference : public RoadSection
{

    //################//
    // STATIC         //
    //################//

public:
    enum ObjectReferenceChange
    {
        ORC_ParameterChange = 0x1,
		ORC_ObjectChange = 0x2
    };


    //################//
    // FUNCTIONS      //
    //################//

public:
	explicit ObjectReference(const QString &id, Object *object, const QString &referenceId, double s, double t, double zOffset, double validLength, Signal::OrientationType orientation, QList<Signal::Validity> validity);
    virtual ~ObjectReference()
    { /* does nothing */
    }

	QString getId()
	{
		return id_;
	}
	void setId(const QString &id)
	{
		id_ = id;
	}

    // ObjectReference //
    //
	QString getReferenceId() const
	{
		return refId_;
	}
	void setReferenceId(const QString &refId)
	{
		refId_ = refId;
	}

	Object *getObject();
	void setObject(Object *object);

    double getReferenceT() const
    {
        return refT_;
    }
	void setReferenceT(const double refT);

	double getReferenceZOffset() const
	{
		return refZOffset_;
	}
	void setReferenceZOffset(const double zOffset);

	double getReferenceValidLength() const
	{
		return refValidLength_;
	}
	void setReferenceValidLength(const double refValidLength);

	Signal::OrientationType getReferenceOrientation() const
	{
		return refOrientation_;
	}
	void setReferenceOrientation(Signal::OrientationType orientation);

	//Validity //
	//
	QList<Signal::Validity> getValidityList()
	{
		return validity_;
	}
	bool addValidity(int fromLane, int toLane);

	// virtual roadsection methods //
	//
	double getSEnd() const
	{
		return getSStart();
	};
	double getLength() const
	{
		return 0;
	};


    // Observer Pattern //
    //
    virtual void notificationDone();
    int getObjectReferenceChanges() const
    {
        return objectReferenceChanges_;
    }
	void addObjectReferenceChanges(int changes);

    // Prototype Pattern //
    //
    ObjectReference *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    ObjectReference(); /* not allowed */
    ObjectReference(const ObjectReference &); /* not allowed */
    ObjectReference &operator=(const ObjectReference &); /* not allowed */


    //################//
    // PROPERTIES     //
    //################//

private:
    // ObjectReference //
    //
	QString id_;

	QString refId_;
    double refT_;
	double refZOffset_;
	double refValidLength_;
	Signal::OrientationType refOrientation_;
	Object *object_;

	QList<Signal::Validity> validity_;


    // Change flags //
    //
    int objectReferenceChanges_;
};

#endif // OBJECTREFERENCE_HPP
