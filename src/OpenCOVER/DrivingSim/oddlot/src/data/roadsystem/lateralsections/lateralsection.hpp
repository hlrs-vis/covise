/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   18.03.2010
**
**************************************************************************/

#ifndef LATERALSECTION_HPP
#define LATERALSECTION_HPP

#include "src/data/dataelement.hpp"

class ShapeSection;

class LateralSection : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum LateralSectionChange
    {
        CLS_TChange = 0x1,
        CLS_LengthChange = 0x2,
        CLS_ParentSectionChange = 0x4
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LateralSection(double t);
    virtual ~LateralSection();

    // Section //
    //
    virtual double getTStart() const
    {
        return t_;
    }
    virtual double getTEnd() const = 0;
    virtual double getLength() const = 0;

    void setTStart(double t);

    // Road //
    //
    ShapeSection *getParentSection() const
    {
        return parentSection_;
    }
    void setParentSection(ShapeSection *parentSection);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getLateralSectionChanges() const
    {
        return lateralSectionChanges_;
    }
    void addLateralSectionChanges(int changes);

private:
    LateralSection(); /* not allowed */
    LateralSection(const LateralSection &); /* not allowed */
    LateralSection &operator=(const LateralSection &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Road //
    //
	ShapeSection *parentSection_; // linked

    // Section //
    //
    double t_;

    // Change //
    //
    int lateralSectionChanges_;
};

#endif // LATERALSECTION_HPP
