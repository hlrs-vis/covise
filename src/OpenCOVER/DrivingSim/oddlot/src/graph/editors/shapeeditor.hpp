/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   14.07.2010
**
**************************************************************************/

#ifndef SHAPEEDITOR_HPP
#define SHAPEEDITOR_HPP

#include "projecteditor.hpp"

class ProjectData;

class TopviewGraph;
class ProfileGraph;


class RoadSystemItem;

class ShapeRoadSystemItem;
class SectionHandle;
class ShapeSection;
class ShapeSectionPolynomialItems;
class SplineMoveHandle;
class SplineControlPoint;
class PolynomialLateralSection;

#include <QMap>
#include <QRectF>

class ShapeEditor : public ProjectEditor
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ShapeEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph, ProfileGraph *profileGraph);
    virtual ~ShapeEditor();

    // TODO look for better solution
    SectionHandle *getInsertSectionHandle();

    // ProfileGraph //
    //
    ProfileGraph *getProfileGraph() const
    {
        return profileGraph_;
    }


    // Selected ShapeSections //
    //
    void addSelectedShapeSection(ShapeSection *shapeSection);
    int delSelectedShapeSection(ShapeSection *shapeSection);

/*	QMap<double, PolynomialLateralSection *>::ConstIterator addLateralSectionsBeforeToEasingCurve(QEasingCurve &easingCurve, QMap<double, PolynomialLateralSection *>::ConstIterator it, 
		PolynomialLateralSection *lateralSectionBefore, PolynomialLateralSection *nextLateralSection, ShapeSectionPolynomialItems *polyItems);
	void addLateralSectionsNextToEasingCurve(QEasingCurve &easingCurve, QMap<double, PolynomialLateralSection *>::ConstIterator it, ShapeSection *shapeSection, ShapeSectionPolynomialItems *polyItems); */
	QMap<double, PolynomialLateralSection *>::ConstIterator addLateralSectionsBefore(QList<QPointF> &scenePoints, QMap<double, PolynomialLateralSection *>::ConstIterator it,
		PolynomialLateralSection *lateralSectionBefore, ShapeSectionPolynomialItems *polyItems);
	void addLateralSectionsNext(QList<QPointF> &scenePoints, QMap<double, PolynomialLateralSection *>::ConstIterator it, ShapeSection *shapeSection, ShapeSectionPolynomialItems *polyItems);

	// Transformations //
	//
	void translateMoveHandles(const QPointF &mousePos, SplineControlPoint *corner);
//	QList<QPointF> setEasingCurve(ShapeSectionPolynomialItems *,SplineControlPoint *start, const QEasingCurve &easingCurve);
	void fitBoundingBoxInView();

	// Add lateral section (insert real point in spline) //
	//
	void addLateralSection(ShapeSection *shapeSection, const QPointF &mousePos);
	void deleteLateralSection(SplineControlPoint *corner);

	// Smooth real point //
	//
/*	void smoothPoint(bool smooth, SplineControlPoint *corner);
	void cornerPoint(SplineControlPoint *corner); */


    // Tool, Mouse & Key //
    //
    virtual void toolAction(ToolAction *toolAction);
    virtual void mouseAction(MouseAction * mouseAction);
    //	virtual void			keyAction(KeyAction * keyAction);

protected:
    virtual void init();
    virtual void kill();

private:
    ShapeEditor(); /* not allowed */
    ShapeEditor(const ShapeEditor &); /* not allowed */
    ShapeEditor &operator=(const ShapeEditor &); /* not allowed */

    //################//
    // SLOTS          //
    //################//

public slots:

    //################//
    // PROPERTIES     //
    //################//

private:
    // Graph //
    //
    ShapeRoadSystemItem *roadSystemItem_;

    // ProfileGraph //
    //
    ProfileGraph *profileGraph_;

    // TODO look for better solution
    SectionHandle *insertSectionHandle_;


	// ProfileGraph: Selected Items //
	//
	QMap<ShapeSection *, ShapeSectionPolynomialItems *> selectedShapeSectionItems_;
	QList<SplineMoveHandle *> moveHandles_;

	QRectF boundingBox_;

};

#endif // SHAPEEDITOR_HPP
