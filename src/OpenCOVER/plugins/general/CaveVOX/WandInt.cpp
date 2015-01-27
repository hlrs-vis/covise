/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Inspace:
#include <ISVREngine.H>
#include <DefaultCursor.H>

// OSG:
#include <osg/Geode>
#include <osg/MatrixTransform>

// Local:
#include "WandInt.H"
#include "Interaction.H"

using namespace osg;

WandInt::WandInt(cui::Interaction *interact)
    : Interactor<WandInt>("WandInt")
{
    static STATE move_yellow("move_yellow");
    static STATE move_red("move_red");
    static STATE move_blue("move_blue");
    static STATE wheel_start("wheel_start");

    _interaction = interact;
    _rWandPos.set(0, 0, 0);
    _rWandDir.set(0, 0, 1);
    _rLastWandDir = _rWandDir;
    _rLastWandPos = _rWandPos;
    _lastWand2r = _wand2r;
    _joystick[0] = _joystick[1] = 0.0f;

    str_ptr wand_name = CONFIGval("WAND_NAME", "wand_polhemus", false);
    str_ptr button_0_name = CONFIGval("BUTTON_0_NAME", "wanda_but_red", false);
    str_ptr button_1_name = CONFIGval("BUTTON_1_NAME", "wanda_but_yellow", false);
    str_ptr button_2_name = CONFIGval("BUTTON_2_NAME", "wanda_but_blue", false);
    str_ptr joystick_name = CONFIGval("JOYSTICK_NAME", "wanda_dev2d", false);

    // Wand idle:
    _entry += Arc(this, Gd(new EVENTroom6d(wand_name)), &WandInt::idle);

    // Blue button:
    _entry += Arc(this, Gd(new EVENTbtn(button_0_name, 1)), &WandInt::down0, &move_blue);
    move_blue += Arc(this, Gd(new EVENTroom6d(wand_name)), &WandInt::move0);
    move_blue += Arc(this, Gd(new EVENTbtn(button_0_name, 0)), &WandInt::up0, STATEstart);

    // Yellow button:
    _entry += Arc(this, Gd(new EVENTbtn(button_1_name, 1)), &WandInt::down1, &move_yellow);
    move_yellow += Arc(this, Gd(new EVENTroom6d(wand_name)), &WandInt::move1);
    move_yellow += Arc(this, Gd(new EVENTbtn(button_1_name, 0)), &WandInt::up1, STATEstart);

    // Red button:
    _entry += Arc(this, Gd(new EVENTbtn(button_2_name, 1)), &WandInt::down2, &move_red);
    move_red += Arc(this, Gd(new EVENTroom6d(wand_name)), &WandInt::move2);
    move_red += Arc(this, Gd(new EVENTbtn(button_2_name, 0)), &WandInt::up2, STATEstart);

    // Trackball/Joystick:
    _entry += Arc(this, Gd(new EVENT2d(joystick_name)), &WandInt::joystick);

    // Wheel mouse:
    wheel_start += Arc(this, Gd(new EVENTbtn("wheel_up", 1)), &WandInt::wheelUp);
    wheel_start += Arc(this, Gd(new EVENTbtn("wheel_down", 1)), &WandInt::wheelDown);
    wheel_start += Arc(this, Gd(new EVENT2d("wheelmouse_dev2d")), &WandInt::wheelMove);
    wheel_start += Arc(this, Gd(new EVENTbtn("wheelmouse_button0", 1)), &WandInt::wheelButton0down);
    wheel_start += Arc(this, Gd(new EVENTbtn("wheelmouse_button0", 0)), &WandInt::wheelButton0up);
    wheel_start += Arc(this, Gd(new EVENTbtn("wheelmouse_button1", 1)), &WandInt::wheelButton1down);
    wheel_start += Arc(this, Gd(new EVENTbtn("wheelmouse_button1", 0)), &WandInt::wheelButton1up);
    wheel_start += Arc(this, Gd(new EVENTbtn("wheelmouse_button2", 1)), &WandInt::wheelButton2down);
    wheel_start += Arc(this, Gd(new EVENTbtn("wheelmouse_button2", 0)), &WandInt::wheelButton2up);

    EVENTmgr::add_handler(new FSA(&_entry));
    EVENTmgr::add_handler(new FSA(&wheel_start));
}

void WandInt::wheelUp(cEVENTbtnptr &, STATE *&)
{
    _interaction->_head->wheelTurned(1);
}

void WandInt::wheelDown(cEVENTbtnptr &, STATE *&)
{
    _interaction->_head->wheelTurned(-1);
}

void WandInt::wheelMove(cEVENT2dptr &e, STATE *&)
{
    cerr << "wheelmouseMove" << endl;
}

void WandInt::wheelButton0down(cEVENTbtnptr &, STATE *&)
{
    _interaction->_head->buttonStateChanged(0, 1);
}

void WandInt::wheelButton0up(cEVENTbtnptr &, STATE *&)
{
    _interaction->_head->buttonStateChanged(0, 0);
}

void WandInt::wheelButton1down(cEVENTbtnptr &, STATE *&)
{
    _interaction->_head->buttonStateChanged(1, 1);
}

void WandInt::wheelButton1up(cEVENTbtnptr &, STATE *&)
{
    _interaction->_head->buttonStateChanged(1, 0);
}

void WandInt::wheelButton2down(cEVENTbtnptr &, STATE *&)
{
    _interaction->_head->buttonStateChanged(2, 1);
}

void WandInt::wheelButton2up(cEVENTbtnptr &, STATE *&)
{
    _interaction->_head->buttonStateChanged(2, 0);
}

void WandInt::idle(cEVENTroom6dptr &evt, STATE *&)
{
    updateWandData(evt);
}

void WandInt::down0(cEVENTbtnptr &, STATE *&)
{
    _interaction->_wandR->buttonStateChanged(0, 1);
}

void WandInt::move0(cEVENTroom6dptr &evt, STATE *&)
{
    updateWandData(evt);
}

void WandInt::up0(cEVENTbtnptr &, STATE *&)
{
    _interaction->_wandR->buttonStateChanged(0, 0);
}

void WandInt::down1(cEVENTbtnptr &, STATE *&)
{
    _interaction->_wandR->buttonStateChanged(1, 1);
}

void WandInt::move1(cEVENTroom6dptr &evt, STATE *&)
{
    updateWandData(evt);
}

void WandInt::up1(cEVENTbtnptr &, STATE *&)
{
    _interaction->_wandR->buttonStateChanged(1, 0);
}

void WandInt::down2(cEVENTbtnptr &, STATE *&)
{
    _interaction->_wandR->buttonStateChanged(2, 1);
}

void WandInt::move2(cEVENTroom6dptr &evt, STATE *&)
{
    updateWandData(evt);
}

void WandInt::up2(cEVENTbtnptr &, STATE *&)
{
    _interaction->_wandR->buttonStateChanged(2, 0);
}

void WandInt::joystick(cEVENT2dptr &evt, STATE *&)
{
    _joystick[0] = evt->cur()[0];
    _joystick[1] = evt->cur()[1];
    _interaction->_wandR->joystickValueChanged(_joystick[0], _joystick[1]);
}

void WandInt::updateWandData(cEVENTroom6dptr &evt)
{
    _rLastWandDir = _rWandDir;
    _rLastWandPos = _rWandPos;
    _lastWand2r = _wand2r;

    Wtransf rXF = evt->cur();
    _wand2r.set(rXF.matrix());

    //  _wand2r = filterTrackerData(_wand2r);

    _rWandPos = _wand2r.getTrans();
    double *mat = _wand2r.ptr();
    _rWandDir.set(mat[8], mat[9], mat[10]);
    _rWandDir.normalize();
}

Matrix WandInt::filterTrackerData(Matrix &h2r)
{
    const float THRESHOLD = 15.0f; // head move speed threshold beyond which tracker values are ignored
    static Matrix prevH2R; // h2r matrix from previous frame
    static bool firstRun = true;
    static double prevTime = WallClockInt::time(); // seconds
    Matrix newH2R; // current h2r matrix
    Vec3 newPos, prevPos; // new and previous head position
    Vec3 diff; // difference in head positions now and previous frame
    double timeNow;
    double ds; // distance head moved since last call
    double dt; // time delta between last and current call

    timeNow = WallClockInt::time();

    if (firstRun)
    {
        newH2R = h2r;
        firstRun = false;
    }
    else if (timeNow == prevTime)
        return h2r; // same frame as last one
    else
    {
        newPos = h2r.getTrans();
        prevPos = prevH2R.getTrans();
        diff = newPos - prevPos;
        ds = diff.length();
        dt = timeNow - prevTime;
        if ((ds / dt) < THRESHOLD)
            newH2R = h2r; // check move speed
        else
        {
            newH2R = prevH2R;
            cerr << "wand tracker data filtered out, speed = " << ds / dt << endl;
        }
    }
    prevH2R = newH2R;
    prevTime = timeNow;
    return newH2R;
}
