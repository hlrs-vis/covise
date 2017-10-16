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

#ifndef SHAPESECTION_HPP
#define SHAPESECTION_HPP

#include "roadsection.hpp"
#include "src/util/math/polynomial.hpp"

class ShapeSection : public RoadSection
{

    //################//
    // STATIC         //
    //################//

public:
    enum ShapeSectionChange
    {
        CSS_ParameterChange = 0x1,
		CSS_ShapeSectionChange = 0x2
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ShapeSection(double s);
    virtual ~ShapeSection()
    { /* does nothing */
    }

    // ShapeSection //
    //
	void addShape(double t, Polynomial *poly);
	bool delShape(double t);
	Polynomial *getShape(double t) const;
	double getShapeElevationDegree(double t);

	QMap<double, Polynomial *> getShapes()
	{
		return shapes_;
	}


    // RoadSection //
    //
    //virtual double		getSStart() const { return s_; }
    virtual double getSEnd() const;
    virtual double getLength() const;

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getShapeSectionChanges() const
    {
        return shapeSectionChanges_;
    }
    void addShapeSectionChanges(int changes);

    // Prototype Pattern //
    //
    ShapeSection *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    ShapeSection(); /* not allowed */
    ShapeSection(const ShapeSection &); /* not allowed */
    ShapeSection &operator=(const ShapeSection &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int shapeSectionChanges_;

	// Shapes for consecutive stations //
	//
	QMap<double, Polynomial *> shapes_;
};

#endif // SHAPESECTION_HPP
