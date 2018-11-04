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

#ifndef TUNNELOBJECT_HPP
#define TUNNELOBJECT_HPP

#include "src/data/roadsystem/sections/bridgeobject.hpp"

class Tunnel : public Bridge
{

    //################//
    // STATIC         //
    //################//

public:
    enum TunnelChange
    {
        CEL_ParameterChange = 0x1
    };

	enum TunnelType
	{
		TT_STANDARD,
		TT_UNDERPASS
	};

	static TunnelType parseTunnelType(const QString &type);
	static QString parseTunnelTypeBack(int type);

    struct TunnelUserData
    {
        QString fileName;
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit Tunnel(const odrID &id, const QString &file, const QString &name, int type, double s, double length, double lighting, double daylight);
    virtual ~Tunnel()
    { /* does nothing */
    }

    // Tunnel //
    //

    double getLighting() const
    {
        return lighting_;
    }
    void setLighting(const double lighting)
    {
        lighting_ = lighting;
    }

	double getDaylight() const
    {
        return daylight_;
    }
    void setDaylight(const double daylight)
    {
        daylight_ = daylight;
    }


    // Observer Pattern //
    //
    virtual void notificationDone();
    int getTunnelChanges() const
    {
        return tunnelChanges_;
    }
    void addTunnelChanges(int changes);

    // Prototype Pattern //
    //
    Tunnel *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    Tunnel(); /* not allowed */
    Tunnel(const Tunnel &); /* not allowed */
    Tunnel &operator=(const Tunnel &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Tunnel //
    //

    double daylight_;
	double lighting_;

    TunnelUserData userData_;

    // Change flags //
    //
    int tunnelChanges_;
};

#endif // OBJECTOBJECT_HPP
