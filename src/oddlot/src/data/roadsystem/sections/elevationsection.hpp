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

#ifndef ELEVATIONSECTION_HPP
#define ELEVATIONSECTION_HPP

#include "roadsection.hpp"
#include "src/util/math/polynomial.hpp"

class ElevationSection : public RoadSection, public Polynomial
{

    //################//
    // STATIC         //
    //################//

public:
    enum ElevationSectionChange
    {
        CEL_ParameterChange = 0x1
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ElevationSection(double s, double a, double b, double c, double d);
    virtual ~ElevationSection()
    { /* does nothing */
    }

    // ElevationSection //
    //
    double getElevation(double s)
    {
        return f(s - getSStart());
    }
    double getSlope(double s)
    {
        return df(s - getSStart());
    }
    double getCurvature(double s)
    {
        return ddf(s - getSStart());
    }

    void setParameters(double a, double b, double c, double d);

    bool isEqualTo(ElevationSection *otherSection) const;

    // RoadSection //
    //
    //virtual double		getSStart() const { return s_; }
    virtual double getSEnd() const;
    virtual double getLength() const;

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getElevationSectionChanges() const
    {
        return elevationSectionChanges_;
    }
    void addElevationSectionChanges(int changes);

    // Prototype Pattern //
    //
    ElevationSection *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    ElevationSection(); /* not allowed */
    ElevationSection(const ElevationSection &); /* not allowed */
    ElevationSection &operator=(const ElevationSection &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int elevationSectionChanges_;
};

#endif // ELEVATIONSECTION_HPP
