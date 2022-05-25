/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   12.03.2010
**
**************************************************************************/

#ifndef SIGNALSECTIONPOLYNOMIALITEMS_HPP
#define SIGNALSECTIONPOLYNOMIALITEMS_HPP

#include "src/graph/items/graphelement.hpp"


class RSystemElementRoad;
class SignalEditor;
class SignalManager;
class Signal;
class SignalPoleItem;

class SignalSectionPolynomialItems : public GraphElement
{
    //################//
    // FUNCTIONS      //
    //################//

public:

    explicit SignalSectionPolynomialItems(SignalEditor *signalEditor, SignalManager *signalManager, RSystemElementRoad *road, const double &s);
    virtual ~SignalSectionPolynomialItems();

    // Graphics //
    //
    virtual void createPath();

    // Observer Pattern //
    //
    virtual void updateObserver();
    virtual bool deleteRequest()
    {
        return false;
    }

    RSystemElementRoad *getRoad()
    {
        return road_;
    }

    const double &getS()
    {
        return s_;
    }

    void setS(const double &s)
    {
        s_=s;
    }

    const double getClosestT(double t);

    void deselectSignalPoles(Signal *signal);


private:

    void init();

protected:

    //################//
    // SLOTS          //
    //################//


public slots:

    //################//
    // PROPERTIES     //
    //################//

private:

    double s_;

    // SignalManager //
    //
    SignalManager *signalManager_;

    // signalEditor //
    //
    SignalEditor *signalEditor_;

    // Road //
    //
    RSystemElementRoad *road_;

    // Poles //
    //
    QMap<double, QMap<Signal *, SignalPoleItem *>> signalPoleSystemItems_;


};
#endif // SIGNALSECTIONPOLYNOMIALITEMS_HPP
