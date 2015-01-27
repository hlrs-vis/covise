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

#include "crossfallsectionpolynomialitem.hpp"

#include "src/gui/projectwidget.hpp"

// Data //
//
#include "src/data/roadsystem/sections/crossfallsection.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/commands/crossfallsectioncommands.hpp"

// Graph //
//
#include "src/graph/items/roadsystem/crossfall/crossfallroadpolynomialitem.hpp"
#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"

#include "src/graph/profilegraph.hpp"
#include "src/graph/editors/crossfalleditor.hpp"

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

CrossfallSectionPolynomialItem::CrossfallSectionPolynomialItem(CrossfallRoadPolynomialItem *parentRoadItem, CrossfallSection *section)
    : SectionItem(parentRoadItem, section)
    , parentRoadItem_(parentRoadItem)
    , crossfallSection_(section)
{
    init();
}

CrossfallSectionPolynomialItem::~CrossfallSectionPolynomialItem()
{
}

void
CrossfallSectionPolynomialItem::init()
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

/*! \brief Sets the color according to the road crossfall.
*/
void
CrossfallSectionPolynomialItem::updateColor()
{
    QPen pen;
    pen.setWidth(2);
    pen.setCosmetic(true); // constant size independent of scaling

    if (crossfallSection_->getDegree() == 3)
    {
        pen.setColor(ODD::instance()->colors()->darkRed());
    }
    else if (crossfallSection_->getDegree() == 2)
    {
        pen.setColor(ODD::instance()->colors()->darkOrange());
    }
    else if (crossfallSection_->getDegree() == 1)
    {
        pen.setColor(ODD::instance()->colors()->darkGreen());
    }
    else if (crossfallSection_->getDegree() == 0)
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
CrossfallSectionPolynomialItem::createPath()
{
    // Initialization //
    //
    double sStart = crossfallSection_->getSStart();
    double sEnd = crossfallSection_->getSEnd();
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
        points[i] = QPointF(s, crossfallSection_->getCrossfallDegrees(s));
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
CrossfallSectionPolynomialItem::updateObserver()
{
    // Parent //
    //
    SectionItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // CrossfallSection //
    //
    int crossfallChanges = crossfallSection_->getCrossfallSectionChanges();
    if (crossfallChanges & CrossfallSection::CCF_ParameterChange)
    {
        createPath();
        updateColor();
    }
}

//################//
// SLOTS          //
//################//

bool
CrossfallSectionPolynomialItem::removeSection()
{
    RemoveCrossfallSectionCommand *command = new RemoveCrossfallSectionCommand(crossfallSection_, NULL);
    return getProjectGraph()->executeCommand(command);
}

void
CrossfallSectionPolynomialItem::splitSection()
{
    double s = 0.5 * (crossfallSection_->getSStart() + crossfallSection_->getSEnd());

    SplitCrossfallSectionCommand *command = new SplitCrossfallSectionCommand(crossfallSection_, s, NULL);
    getProjectGraph()->executeCommand(command);
}
