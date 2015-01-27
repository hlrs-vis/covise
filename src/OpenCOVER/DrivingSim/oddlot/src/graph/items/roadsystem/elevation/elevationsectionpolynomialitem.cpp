/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   22.06.2010
**
**************************************************************************/

#include "elevationsectionpolynomialitem.hpp"

#include "src/gui/projectwidget.hpp"

// Data //
//
#include "src/data/roadsystem/sections/elevationsection.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/commands/elevationsectioncommands.hpp"

// Graph //
//
#include "src/graph/items/roadsystem/elevation/elevationroadpolynomialitem.hpp"
#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"

#include "src/graph/profilegraph.hpp"
#include "src/graph/editors/elevationeditor.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"
#include "src/gui/lodsettings.hpp"

// Qt //
//
#include <QCursor>
#include <QBrush>
#include <QPen>

// Utils //
//
#include "math.h"
#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"

//################//
// CONSTRUCTOR    //
//################//

ElevationSectionPolynomialItem::ElevationSectionPolynomialItem(ElevationRoadPolynomialItem *parentRoadItem, ElevationSection *section)
    : SectionItem(parentRoadItem, section)
    , parentRoadItem_(parentRoadItem)
    , elevationSection_(section)
{
    init();
}

ElevationSectionPolynomialItem::~ElevationSectionPolynomialItem()
{
}

void
ElevationSectionPolynomialItem::init()
{
    // Color & Path //
    //
    updateColor();
    createPath();

    // ContextMenu //
    //
    splitAction_ = getContextMenu()->addAction("Cut in half");
    connect(splitAction_, SIGNAL(triggered()), this, SLOT(splitSection()));
}

//################//
// GRAPHICS       //
//################//

/*! \brief Sets the color according to the road elevation.
*/
void
ElevationSectionPolynomialItem::updateColor()
{
    QPen pen;
    pen.setWidth(2);
    pen.setCosmetic(true); // constant size independent of scaling

    if (elevationSection_->getDegree() == 3)
    {
        pen.setColor(ODD::instance()->colors()->darkRed());
    }
    else if (elevationSection_->getDegree() == 2)
    {
        pen.setColor(ODD::instance()->colors()->darkOrange());
    }
    else if (elevationSection_->getDegree() == 1)
    {
        pen.setColor(ODD::instance()->colors()->darkGreen());
    }
    else if (elevationSection_->getDegree() == 0)
    {
        pen.setColor(ODD::instance()->colors()->darkCyan());
    }
    else
    {
        pen.setColor(ODD::instance()->colors()->darkBlue());
    }

    setPen(pen);
}

void
ElevationSectionPolynomialItem::createPath()
{
    // Initialization //
    //
    double sStart = elevationSection_->getSStart();
    double sEnd = elevationSection_->getSEnd();
    if (sEnd < sStart)
        sEnd = sStart;

    //	double pointsPerMeter = 2.0; // BAD: hard coded!
    double pointsPerMeter = getProjectGraph()->getProjectWidget()->getLODSettings()->HeightEditorPointsPerMeter;
    int pointCount = int(ceil((sEnd - sStart) * pointsPerMeter)); // TODO curvature...
    if (pointCount < 2)
    {
        pointCount = 2;
    }

    QVector<QPointF> points(pointCount);
    double segmentLength = (sEnd - sStart) / (pointCount - 1);

    // Points //
    //
    for (int i = 0; i < pointCount; ++i)
    {
        double s = sStart + i * segmentLength; // [sStart, sEnd]
        points[i] = QPointF(s, elevationSection_->getElevation(s));
        //		if ((points[i].x() < 0.0) || (points[i].y() < 0)) fprintf(stderr, "Points: %f %f ", points[i].x(), points[i].y());
    }

    // Psycho-Path //
    //
    QPainterPath path;
    path.addPolygon(QPolygonF(points));

    setPath(path);
}

//################//
// OBSERVER       //
//################//

void
ElevationSectionPolynomialItem::updateObserver()
{
    // Parent //
    //
    SectionItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // ElevationSection //
    //
    int elevationChanges = elevationSection_->getElevationSectionChanges();
    if (elevationChanges & ElevationSection::CEL_ParameterChange)
    {
        createPath();
        updateColor();
    }
}

//################//
// SLOTS          //
//################//

bool
ElevationSectionPolynomialItem::removeSection()
{
    RemoveElevationSectionCommand *command = new RemoveElevationSectionCommand(elevationSection_, NULL);
    return getProjectGraph()->executeCommand(command);
}

void
ElevationSectionPolynomialItem::splitSection()
{
    double s = 0.5 * (elevationSection_->getSStart() + elevationSection_->getSEnd());

    SplitElevationSectionCommand *command = new SplitElevationSectionCommand(elevationSection_, s, NULL);
    getProjectGraph()->executeCommand(command);
}
