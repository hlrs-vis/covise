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

#ifndef CROSSFALLSECTION_HPP
#define CROSSFALLSECTION_HPP

#include "roadsection.hpp"
#include "src/util/math/polynomial.hpp"

class CrossfallSection : public RoadSection, public Polynomial
{

    //################//
    // STATIC         //
    //################//

public:
    enum CrossfallSectionChange
    {
        CCF_ParameterChange = 0x1,
        CCF_SideChange = 0x2
    };

    enum DCrossfallSide
    {
        DCF_SIDE_NONE,
        DCF_SIDE_LEFT,
        DCF_SIDE_RIGHT,
        DCF_SIDE_BOTH
    };
    static CrossfallSection::DCrossfallSide parseCrossfallSide(const QString &side);
    static QString parseCrossfallSideBack(CrossfallSection::DCrossfallSide side);

    //################//
    // FUNCTIONS      //
    //################//

    // Notes //
    //
    // The sides are understood as beeing leftonly, rightonly and both. So when a section is leftonly, the right side is equal zero!
    //
    // The Crossfall is saved in degrees not rad!

public:
    explicit CrossfallSection(CrossfallSection::DCrossfallSide side, double s, double a, double b, double c, double d);
    virtual ~CrossfallSection()
    { /* does nothing */
    }

    // CrossfallSection //
    //
    double getCrossfallRadians(double s);
    double getCrossfallDegrees(double s);
    double getCrossfallSlopeRadians(double s);
    double getCrossfallSlopeDegrees(double s);
    double getCrossfallCurvatureRadians(double s);
    double getCrossfallCurvatureDegrees(double s);

    void setParametersDegrees(double a, double b, double c, double d);

    CrossfallSection::DCrossfallSide getSide() const
    {
        return side_;
    }
    void setSide(CrossfallSection::DCrossfallSide side);

    bool isEqualTo(CrossfallSection *otherSection) const;

    // RoadSection //
    //
    //virtual double		getSStart() const { return s_; }
    virtual double getSEnd() const;
    virtual double getLength() const;

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getCrossfallSectionChanges() const
    {
        return crossfallSectionChanges_;
    }
    void addCrossfallSectionChanges(int changes);

    // Prototype Pattern //
    //
    CrossfallSection *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    CrossfallSection(); /* not allowed */
    CrossfallSection(const CrossfallSection &); /* not allowed */
    CrossfallSection &operator=(const CrossfallSection &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int crossfallSectionChanges_;

    // Side //
    //
    CrossfallSection::DCrossfallSide side_;
};

#endif // CROSSFALLSECTION_HPP
