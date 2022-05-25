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

#include "signalsectionpolynomialitems.hpp"

// Data //
//

#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/signalmanager.hpp"
#include "src/data/roadsystem/sections/signalobject.hpp"

// Graph //
//
#include "src/graph/editors/signaleditor.hpp"
#include "src/graph/items/roadsystem/signal/signalpoleitem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Editor //
//
#include "src/graph/editors/signaleditor.hpp"

// GUI //
//
#include "src/util/colorpalette.hpp"

// Qt //
//
#include <QPen>

//################//
// CONSTRUCTOR    //
//################//
///
/// \brief Constructor.
/// \param profileGraph
/// \param roadSystemItem
/// \param road
///
SignalSectionPolynomialItems::SignalSectionPolynomialItems(SignalEditor *signalEditor, SignalManager *signalManager, RSystemElementRoad *road, const double & s)
    : GraphElement(NULL, road)
    , signalEditor_(signalEditor)
    , signalManager_(signalManager)
    , road_(road)
{
    // Init //
    //
    setS(s);
    init();
}

///
/// \brief Destructor.
///
SignalSectionPolynomialItems::~SignalSectionPolynomialItems()
{
    
}

///
/// \brief Initialize signaleditor and build street.
///
void
SignalSectionPolynomialItems::init()
{
    QList<Signal *> poleSignals = road_->getSignals(s_);

    bool hasChanged = false;
    foreach(Signal * signal, poleSignals)  // signal added
    {
        double t = signal->getT();
        if (!signalPoleSystemItems_.contains(t))
        {
            QMap<Signal *, SignalPoleItem *> signalPoles;
            SignalPoleItem *signalPoleItem = new SignalPoleItem(this, signal, signalEditor_, signalManager_);
            signalPoles.insert(signal, signalPoleItem);
            signalPoleSystemItems_.insert(t, signalPoles);
            hasChanged = true;
        }
        else
        {
            QMap<double, QMap<Signal *, SignalPoleItem *>>::iterator iter = signalPoleSystemItems_.find(t);
            if (!iter.value().contains(signal))
            {
                SignalPoleItem *signalPoleItem = new SignalPoleItem(this, signal, signalEditor_, signalManager_);
                iter.value().insert(signal, signalPoleItem);
                hasChanged = true;
            }
        }
    }

    if (hasChanged)
    {
        createPath();
    }
}

///
/// \brief Building the road in side view.
///
void
SignalSectionPolynomialItems::createPath()
{

    double roadHeight = -0.1;
    LaneSection *laneSection = road_->getLaneSection(s_);
    QPainterPath roadPathR;
    QPainterPath roadPathL;
    roadPathL.moveTo(0,0.1);
    roadPathL.lineTo(0,-0.1);
    roadPathL.moveTo(0,0);

    QPen pen(ODD::instance()->colors()->brightGrey(), 0.05);
    setPen(pen);
    foreach(Lane *lane, laneSection->getLanes())
    {
        if (lane->getId() > 0)
        {
            roadPathL.lineTo(-laneSection->getLaneSpanWidth(0, lane->getId(), s_) + road_->getLaneOffset(s_) , 0);
            roadPathL.lineTo(-laneSection->getLaneSpanWidth(0, lane->getId(), s_) + road_->getLaneOffset(s_), roadHeight);
            roadPathL.lineTo(0,roadHeight);
            roadPathL.moveTo(-laneSection->getLaneSpanWidth(0, lane->getId(), s_) + road_->getLaneOffset(s_) + laneSection->getLaneSpanWidth(0, lane->getId() - 1, s_) , 0);
        }
        else if (lane->getId() < 0)
        {
            roadPathR.lineTo(laneSection->getLaneSpanWidth(0, lane->getId(), s_) + road_->getLaneOffset(s_)  , 0);
            roadPathR.lineTo(laneSection->getLaneSpanWidth(0, lane->getId(), s_) + road_->getLaneOffset(s_), roadHeight);
            roadPathR.lineTo(0,roadHeight);
            roadPathR.moveTo(laneSection->getLaneSpanWidth(0, lane->getId(), s_) + road_->getLaneOffset(s_)  - laneSection->getLaneSpanWidth(0, lane->getId() - 1, s_), 0);
        }
    }

    QPainterPath roadPath;
    roadPath.addPath(roadPathR);
    roadPath.addPath(roadPathL);


    QMap<double, QMap<Signal *, SignalPoleItem *>>::const_iterator iter = signalPoleSystemItems_.constBegin();
    while (iter != signalPoleSystemItems_.constEnd())
    {
        QPainterPath pole;
        pole.moveTo(iter.key(), 0.0);
        double poleLength = 0.0;
        foreach(Signal *signal, iter.value().keys())
        {
            double zOffset = signal->getZOffset();
            if (zOffset > poleLength)
            {
                poleLength = zOffset;
            }
        }
        pole.lineTo(iter.key(), poleLength);
        roadPath.addPath(pole);
        iter++;
    }
    setPath(roadPath);
}

const double
SignalSectionPolynomialItems::getClosestT(double t)
{
    QMap<double, QMap<Signal *, SignalPoleItem *>>::iterator iter = signalPoleSystemItems_.lowerBound(t);
    if (iter == signalPoleSystemItems_.end())
    {
        iter--;
    }
    double closestT = iter.key();
    if (iter != signalPoleSystemItems_.begin())
    {
        iter--;
        if (abs(iter.key() - t) < abs(closestT - t))
        {
            return iter.key();
        }
    }
   
    return closestT;
}

void 
SignalSectionPolynomialItems::deselectSignalPoles(Signal *signal)
{
    double t = signal->getT();
    QMap<double, QMap<Signal *, SignalPoleItem *>>::const_iterator iter = signalPoleSystemItems_.find(t);
    QMap<Signal *, SignalPoleItem *> signalPoles = iter.value();
    QMap<Signal *, SignalPoleItem *>::const_iterator it = signalPoles.constBegin();
    while (it != signalPoles.constEnd())
    {
        if ((it.key() != signal) && it.value()->isSelected())
        {
            it.value()->setSelected(false);
        }
        it++;
    }
}

//################//
// OBSERVER       //
//################//

///
/// \brief SignalSectionPolynomialItems::updateObserver
///
void
SignalSectionPolynomialItems::updateObserver()
{
    // Parent //
    //
    GraphElement::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    int changes = road_->getRoadChanges();
    if (changes & RSystemElementRoad::CRD_SignalChange)
    {
        // A signal has been added or moved.
        //
        init();
    }
    else if (road_->getSignalChanges() & Signal::CEL_ParameterChange)
    {
        createPath();
    }
}

