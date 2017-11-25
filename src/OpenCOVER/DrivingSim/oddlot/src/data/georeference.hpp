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

#ifndef GEOREFERENCE_HPP
#define GEOREFERENCE_HPP

#include "dataelement.hpp"

class GeoReference : public DataElement
{
	//################//
	// STATIC         //
	//################//

public:
	enum GeoReferenceChange
	{
		CEL_ParameterChange = 0x1
	};


    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit GeoReference(const QString &geoReferenceParams);
    virtual ~GeoReference();

    // OpenDRIVE:header //
    //
    QString getParams() const
    {
        return geoReferenceParams_;
    }

	void setParams(const QString &geoParams)
	{
		geoReferenceParams_ = geoParams;
	}


    // Observer Pattern //
    //
    int getGeoReferenceChanges() const
    {
        return geoReferenceChanges_;
    }
    void addGeoReferenceChanges(int changes);
    virtual void notificationDone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
	GeoReference(); /* not allowed */
	GeoReference(const GeoReference &); /* not allowed */
	GeoReference &operator=(const GeoReference &); /* not allowed */

//################//
// SIGNALS        //
//################//

signals:

    //################//
    // SLOTS          //
    //################//

public slots:


    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int geoReferenceChanges_;

	// Georeference common params according to PROj.4
	QString geoReferenceParams_;

};

#endif // GEOREFERENCE_HPP
