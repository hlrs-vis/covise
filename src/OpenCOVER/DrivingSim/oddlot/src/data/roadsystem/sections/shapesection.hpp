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
#include "src/data/roadsystem/lateralsections/polynomiallateralsection.hpp"

class ShapeSection : public RoadSection
{

    //################//
    // STATIC         //
    //################//

public:
    enum ShapeSectionChange
    {
        CSS_ParameterChange = 0x1,
		CSS_ShapeSectionChange = 0x2,
		CSS_LengthChange=0x4
    };

    //################//	
    // FUNCTIONS      //
    //################//

public:
	explicit ShapeSection(double s, double t, PolynomialLateralSection *lateralSection = NULL);

	virtual ~ShapeSection();

    // ShapeSection //
    //

	void addShape(double t, PolynomialLateralSection *poly);
	bool delShape(double t);
	PolynomialLateralSection *getShape(double t) const;
	bool moveLateralSection(LateralSection *section, double newT);
	double getShapeElevationDegree(double t);
	double getWidth();
	double getLength(double t);

	QMap<double, PolynomialLateralSection *> getShapes()
	{
		return shapes_;
	}
	int getShapesMaxDegree();

	void checkSmooth(PolynomialLateralSection *lateralSectionBefore, PolynomialLateralSection *lateralSection);

//	QVector<QPointF> getControlPoints();
//	void setPolynomialParameters(QVector<QPointF> controlPoints);


    // RoadSection //
    //
    //virtual double		getSStart() const { return s_; }
    virtual double getSEnd() const;
	virtual double getLength() const;

	// PolynomialLateralSections //
	//
	double getPolynomialLateralSectionEnd(double t) const;
	PolynomialLateralSection *getPolynomialLateralSection(double t) const;
	PolynomialLateralSection *getPolynomialLateralSectionBefore(double t) const;
	PolynomialLateralSection *getPolynomialLateralSectionNext(double t) const;
	PolynomialLateralSection *getFirstPolynomialLateralSection() const;
	PolynomialLateralSection *getLastPolynomialLateralSection() const;

	QMap<double, PolynomialLateralSection *> getPolynomialLateralSections()
	{
		return shapes_;
	}
	void setPolynomialLateralSections(QMap<double, PolynomialLateralSection *> newShapes);

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
	QMap<double, PolynomialLateralSection *> shapes_;
};

#endif // SHAPESECTION_HPP
