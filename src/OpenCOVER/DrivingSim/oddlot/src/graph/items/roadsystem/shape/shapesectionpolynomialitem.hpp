/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   15.07.2010
**
**************************************************************************/

#ifndef SHAPESECTIONPOLYNOMIALITEM_HPP
#define SHAPESECTIONPOLYNOMIALITEM_HPP

#include "src/graph/items/roadsystem/sections/lateralsectionitem.hpp"

#include "src/graph/editors/shapeeditor.hpp"


class ShapeSectionPolynomialItems;
class ShapeSection;
class PolynomialLateralSection;
class SplineMoveHandle;
class ShapeEditor;

#include <QEasingCurve>
#include <QAction>

class ShapeSectionPolynomialItem : public LateralSectionItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ShapeSectionPolynomialItem(ShapeSectionPolynomialItems *parentShapeSectionPolynomialItems, PolynomialLateralSection *polynomialLateralSection, ShapeEditor *shapeEditor);
    virtual ~ShapeSectionPolynomialItem();


    // Section //
    //
	PolynomialLateralSection *getPolynomialLateralSection() const
    {
        return polynomialLateralSection_;
    }

	// Root Item //
	//
	ShapeSectionPolynomialItems *getParentPolynomialItems()
	{
		return parentShapeSectionPolynomialItems_;
	}

    // Graphics //
    //
    void updateColor();
    virtual void createPath();

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest()
    {
        return false;
    };

private:
    ShapeSectionPolynomialItem(); /* not allowed */
    ShapeSectionPolynomialItem(const ShapeSectionPolynomialItem &); /* not allowed */
    ShapeSectionPolynomialItem &operator=(const ShapeSectionPolynomialItem &); /* not allowed */

    void init();

public:

	//################//
	// SIGNALS        //
	//################//

signals:

    //################//
    // SLOTS          //
    //################//

public slots:
    virtual bool removeSection();

    //################//
    // EVENTS         //
    //################//


public:


    //################//
    // PROPERTIES     //
    //################//

private:
    // ShapeSectionPolynomialItems //
    //
	ShapeSectionPolynomialItems *parentShapeSectionPolynomialItems_;

    // Section //
    //
    PolynomialLateralSection *polynomialLateralSection_;
	ShapeSection *shapeSection_;

	// Editor //
	//
	ShapeEditor *shapeEditor_;

	// SplineMoveHandles //
	//
	SplineMoveHandle *realPointLowHandle_, *realPointHighHandle_; 
};

#endif // SHAPESECTIONPOLYNOMIALITEM_HPP
