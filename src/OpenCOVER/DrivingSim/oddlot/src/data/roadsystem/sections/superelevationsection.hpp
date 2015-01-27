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

#ifndef SUPERELEVATIONSECTION_HPP
#define SUPERELEVATIONSECTION_HPP

#include "roadsection.hpp"
#include "src/util/math/polynomial.hpp"

class SuperelevationSection : public RoadSection, public Polynomial
{

    //################//
    // STATIC         //
    //################//

public:
    enum SuperelevationSectionChange
    {
        CSE_ParameterChange = 0x1
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SuperelevationSection(double s, double a, double b, double c, double d);
    virtual ~SuperelevationSection()
    { /* does nothing */
    }

    // SuperelevationSection //
    //
    double getSuperelevationRadians(double s);
    double getSuperelevationDegrees(double s);
    double getSuperelevationSlopeRadians(double s);
    double getSuperelevationSlopeDegrees(double s);
    double getSuperelevationCurvatureRadians(double s);
    double getSuperelevationCurvatureDegrees(double s);

    void setParametersDegrees(double a, double b, double c, double d);

    // RoadSection //
    //
    //virtual double		getSStart() const { return s_; }
    virtual double getSEnd() const;
    virtual double getLength() const;

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getSuperelevationSectionChanges() const
    {
        return superelevationSectionChanges_;
    }
    void addSuperelevationSectionChanges(int changes);

    // Prototype Pattern //
    //
    SuperelevationSection *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    SuperelevationSection(); /* not allowed */
    SuperelevationSection(const SuperelevationSection &); /* not allowed */
    SuperelevationSection &operator=(const SuperelevationSection &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int superelevationSectionChanges_;
};

#endif // SUPERELEVATIONSECTION_HPP
