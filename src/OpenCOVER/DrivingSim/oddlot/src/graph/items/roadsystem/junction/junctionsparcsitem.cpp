/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   09.04.2010
**
**************************************************************************/

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   06.04.2010
**
**************************************************************************/

#include "junctionsparcsitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/roadsystem/track/trackspiralarcspiral.hpp"
#include "src/data/roadsystem/track/trackelementarc.hpp"
#include "src/data/roadsystem/track/trackelementspiral.hpp"

#include "src/data/commands/trackcommands.hpp"

// Graph //
//
#include "src/graph/items/roadsystem/roaditem.hpp"
#include "junctionsparcshandle.hpp"

// Editor //
//
#include "src/graph/editors/junctioneditor.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QGraphicsSceneHoverEvent>
//#include <QMessageBox>
#include <QVector2D>

// Utils //
//
#include "src/util/colorpalette.hpp"

#include <QDebug>

#include "math.h"

//################//
// CONSTRUCTOR    //
//################//

JunctionSpArcSItem::JunctionSpArcSItem(JunctionRoadItem *parentJunctionRoadItem, TrackSpiralArcSpiral *trackSpArcS)
    : JunctionCompositeItem(parentJunctionRoadItem, trackSpArcS)
    , trackSpArcS_(trackSpArcS)
{
    // Init //
    //
    init();
}

JunctionSpArcSItem::JunctionSpArcSItem(JunctionComponentItem *parentJunctionComponentItem, TrackSpiralArcSpiral *trackSpArcS)
    : JunctionCompositeItem(parentJunctionComponentItem, trackSpArcS)
    , trackSpArcS_(trackSpArcS)
{
    // Init //
    //
    init();
}

JunctionSpArcSItem::~JunctionSpArcSItem()
{
}

void
JunctionSpArcSItem::init()
{
    // Junction //
    //
    inSpiral_ = trackSpArcS_->getInSpiral();
    arc_ = trackSpArcS_->getArc();
    outSpiral_ = trackSpArcS_->getOutSpiral();

    // Selection/Highlighting //
    //
    //setAcceptHoverEvents(true);
    //setFlag(QGraphicsItem::ItemIsMovable, true);
    //setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);

    // Slider //
    //
    factorHandle_ = new JunctionSparcsHandle(this);
    updateFactorHandle();

    // Color & Path //
    //
    //updateColor();
    setBrush(QBrush(ODD::instance()->colors()->brightGreen()));
    setPen(QPen(ODD::instance()->colors()->darkGreen()));
    createPath();
}

//################//
// GRAPHICS       //
//################//

void
JunctionSpArcSItem::updateFactorHandle()
{
    // Use local position because handles inherit their parens translation //
    //
    factorHandle_->setPos((inSpiral_->getLocalPoint(inSpiral_->getSStart()) + outSpiral_->getLocalPoint(outSpiral_->getSEnd())) / 2.0);

    // Use global heading because handles don't inherit their parents rotation (only translation) //
    //
    QPointF dP = outSpiral_->getGlobalPoint(outSpiral_->getSEnd()) - inSpiral_->getGlobalPoint(inSpiral_->getSStart());

    // Sign depends on being a left or right turn //
    //
    double side = 1.0;
    if (arc_->getCurvature(arc_->getSStart()) > 0.0)
    {
        side = -1.0;
    }

    // Some easy trigonometrics //
    //
    factorHandle_->setRotation(atan2(dP.y(), dP.x()) * 360.0 / (2.0 * M_PI) + 90.0 * side);

    // Factor //
    //
    factorHandle_->setFactor(trackSpArcS_->getFactor());
}

/*! \brief Draws the two tangents.
*
*/
void
JunctionSpArcSItem::createPath()
{
    // Path //
    //
    QPainterPath path;
    path.moveTo(inSpiral_->getLocalPoint(inSpiral_->getSStart()));
    path.lineTo((inSpiral_->getLocalTangent(inSpiral_->getSStart()) * trackSpArcS_->getInTangentLength()).toPointF());

    path.moveTo(outSpiral_->getLocalPoint(outSpiral_->getSEnd()));
    path.lineTo(outSpiral_->getLocalPoint(outSpiral_->getSEnd()) - (outSpiral_->getLocalTangent(outSpiral_->getSEnd()) * trackSpArcS_->getOutTangentLength()).toPointF());

    setPath(path);
}

//################//
// OBSERVER       //
//################//

void
JunctionSpArcSItem::updateObserver()
{
    // Parent //
    //
    JunctionCompositeItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Get change flags //
    //
    int changes = trackSpArcS_->getTrackSpArcSChanges();
    if (changes & TrackSpiralArcSpiral::CTV_ParameterChange)
    {
        createPath();
        updateFactorHandle();
    }
}

//################//
// EVENTS         //
//################//
