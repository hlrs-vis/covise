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

#ifndef SIGNALREFERENCE_HPP
#define SIGNALREFERENCE_HPP

#include "src/data/roadsystem/sections/roadsection.hpp"
#include "src/data/roadsystem/sections/signalobject.hpp"
#include "src/data/roadsystem/odrID.hpp"

class SignalReference : public RoadSection
{

    //################//
    // STATIC         //
    //################//

public:
    enum SignalReferenceChange
    {
        SRC_ParameterChange = 0x1,
		SRC_SignalChange = 0x2
    };


    //################//
    // FUNCTIONS      //
    //################//

public:
	explicit SignalReference(const odrID &id, Signal *signal, const QString &referenceId, double s, double t, Signal::OrientationType orientation, QList<Signal::Validity> validity);
    virtual ~SignalReference()
    { /* does nothing */
    }

	odrID getId()
	{
		return id_;
	}
	void setId(const odrID &id)
	{
		id_ = id;
	}

    // SignalReference //
    //
	odrID getReferenceId() const
	{
		return refId_;
	}
	void setReferenceId(const odrID &refId)
	{
		refId_ = refId;
	}

	Signal *getSignal();
	void setSignal(Signal *signal);

    double getReferenceT() const
    {
        return refT_;
    }
	void setReferenceT(const double refT);

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
    int getSignalReferenceChanges() const
    {
        return signalReferenceChanges_;
    }
	void addSignalReferenceChanges(int changes);

    // Prototype Pattern //
    //
    SignalReference *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    SignalReference(); /* not allowed */
    SignalReference(const SignalReference &); /* not allowed */
    SignalReference &operator=(const SignalReference &); /* not allowed */


    //################//
    // PROPERTIES     //
    //################//

private:
    // SignalReference //
    //
	odrID id_;

	odrID refId_;
    double refT_;
	Signal::OrientationType refOrientation_;
	Signal *signal_;

	QList<Signal::Validity> validity_;


    // Change flags //
    //
    int signalReferenceChanges_;
};

#endif // SIGNALREFERENCE_HPP
