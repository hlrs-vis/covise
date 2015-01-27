/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _WANDINT_H_
#define _WANDINT_H_

// Inspace:
#include <ivrflygrab.H>

// OSG:
#include <osg/Vec3>
#include <osg/Matrix>
#include <osg/Geode>

// Local:
#include "Interaction.H"

class WandInt : public Interactor<WandInt>
{
protected:
    cui::Interaction *_interaction;
    void updateWandData(cEVENTroom6dptr &);
    osg::Matrix filterTrackerData(osg::Matrix &);

public:
    osg::Vec3 _rWandPos, _rLastWandPos;
    osg::Vec3 _rWandDir, _rLastWandDir;
    osg::Matrix _wand2r, _lastWand2r;
    float _joystick[2]; ///< values are [-1..1], 0 in neutral position

    WandInt(cui::Interaction *);
    void idle(cEVENTroom6dptr &, STATE *&);
    void down0(cEVENTbtnptr &, STATE *&);
    void move0(cEVENTroom6dptr &, STATE *&);
    void up0(cEVENTbtnptr &, STATE *&);
    void down1(cEVENTbtnptr &, STATE *&);
    void move1(cEVENTroom6dptr &, STATE *&);
    void up1(cEVENTbtnptr &, STATE *&);
    void down2(cEVENTbtnptr &, STATE *&);
    void move2(cEVENTroom6dptr &, STATE *&);
    void up2(cEVENTbtnptr &, STATE *&);
    void joystick(cEVENT2dptr &, STATE *&);
    void wheelUp(cEVENTbtnptr &, STATE *&);
    void wheelDown(cEVENTbtnptr &, STATE *&);
    void wheelMove(cEVENT2dptr &e, STATE *&);
    void wheelButton0down(cEVENTbtnptr &, STATE *&);
    void wheelButton0up(cEVENTbtnptr &, STATE *&);
    void wheelButton1down(cEVENTbtnptr &, STATE *&);
    void wheelButton1up(cEVENTbtnptr &, STATE *&);
    void wheelButton2down(cEVENTbtnptr &, STATE *&);
    void wheelButton2up(cEVENTbtnptr &, STATE *&);
};

#endif
