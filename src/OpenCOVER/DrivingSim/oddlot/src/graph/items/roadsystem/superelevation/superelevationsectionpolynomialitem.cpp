/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   16.07.2010
**
**************************************************************************/

#include "superelevationsectionpolynomialitem.hpp"

#include "src/gui/projectwidget.hpp"

// Data //
//
#include "src/data/roadsystem/sections/superelevationsection.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/commands/superelevationsectioncommands.hpp"

// Graph //
//
#include "src/graph/items/roadsystem/superelevation/superelevationroadpolynomialitem.hpp"
#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"

#include "src/graph/profilegraph.hpp"
#include "src/graph/editors/superelevationeditor.hpp"

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

SuperelevationSectionPolynomialItem::SuperelevationSectionPolynomialItem(SuperelevationRoadPolynomialItem *parentRoadItem, SuperelevationSection *section)
    : SectionItem(parentRoadItem, section)
    , parentRoadItem_(parentRoadItem)
    , superelevationSection_(section)
{
    init();
}

SuperelevationSectionPolynomialItem::~SuperelevationSectionPolynomialItem()
{
}

void
SuperelevationSectionPolynomialItem::init()
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

/*! \brief Sets the color according to the road superelevation.
*/
void
SuperelevationSectionPolynomialItem::updateColor()
{
    QPen pen;
    pen.setWidth(2);
    pen.setCosmetic(true); // constant size independent of scaling

    if (superelevationSection_->getDegree() == 3)
    {
        pen.setColor(ODD::instance()->colors()->darkRed());
    }
    else if (superelevationSection_->getDegree() == 2)
    {
        pen.setColor(ODD::instance()->colors()->darkOrange());
    }
    else if (superelevationSection_->getDegree() == 1)
    {
        pen.setColor(ODD::instance()->colors()->darkGreen());
    }
    else if (superelevationSection_->getDegree() == 0)
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
SuperelevationSectionPolynomialItem::createPath()
{
    // Initialization //
    //
    double sStart = superelevationSection_->getSStart();
    double sEnd = superelevationSection_->getSEnd();
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
        points[i] = QPointF(s, superelevationSection_->getSuperelevationDegrees(s));
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
SuperelevationSectionPolynomialItem::updateObserver()
{
    // Parent //
    //
    GraphElement::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // SuperelevationSection //
    //
    int superelevationChanges = superelevationSection_->getSuperelevationSectionChanges();
    if (superelevationChanges & SuperelevationSection::CSE_ParameterChange)
    {
        createPath();
        updateColor();
    }
}

//################//
// SLOTS          //
//################//

bool
SuperelevationSectionPolynomialItem::removeSection()
{
    RemoveSuperelevationSectionCommand *command = new RemoveSuperelevationSectionCommand(superelevationSection_, NULL);
    return getProjectGraph()->executeCommand(command);
}

void
SuperelevationSectionPolynomialItem::splitSection()
{
    double s = 0.5 * (superelevationSection_->getSStart() + superelevationSection_->getSEnd());

    SplitSuperelevationSectionCommand *command = new SplitSuperelevationSectionCommand(superelevationSection_, s, NULL);
    getProjectGraph()->executeCommand(command);
}
