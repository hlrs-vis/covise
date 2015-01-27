/************************************************************************
 *									*
 *          								*
 *                            (C) 2001					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			Wiimote.cpp 				*
 *									*
 *	Description		Wiimote optical tracking system interface class				*
 *									*
 *	Author			Uwe Woessner				*
 *									*
 *	Date			Jan 2004				*
 *									*
 *	Status			in dev					*
 *									*
 ************************************************************************/

#include <iostream>
#include <config/CoviseConfig.h>

#ifdef HAVE_WIIMOTE
#include <string.h>
extern "C" {
#include <wiimote.h>
#include <wiimote_api.h>
}
#endif

#include <cstdio>

#include <device/VRTracker.h>
#include "Wiimote.h"

using std::cerr;
using std::endl;

#define VERBOSE

Wiimote::Wiimote()
    : connected(false)
    , wheelcounter(0)
{
#ifdef VERBOSE
    cerr << "new Wiimote" << endl;
#endif
#ifdef HAVE_WIIMOTE
    wiimote = new wiimote_t;
    memset(wiimote, 0, sizeof(wiimote_t));
    WiiAddress = covise::coCoviseConfig::getEntry("COVER.Input.Wii.Address");
//connected = tryToConnect();
#endif
}

bool Wiimote::tryToConnect()
{
#ifdef HAVE_WIIMOTE
    if (WiiAddress.length() == 0)
    {
        int nmotes = wiimote_discover(wiimote, 1);
        if (nmotes <= 0)
        {
#ifdef VERBOSE
            cerr << "no wiimote discovered" << endl;
#endif
            return false;
        }
        else
        {
            fprintf(stderr, "found wiimote: %s\n", wiimote->link.r_addr);
        }
        WiiAddress = wiimote->link.r_addr;
    }
    if (wiimote_connect(wiimote, WiiAddress.c_str()) < 0)
    {
        fprintf(stderr, "unable to open wiimote: %s\n", wiimote_get_error());
        return false;
    }
    else
    {
        fprintf(stderr, "wiimote connected\n");
    }
    wiimote->led.bits = 1;
#endif
    return true;
}

bool Wiimote::update()
{
#ifdef HAVE_WIIMOTE
    if (!connected)
        connected = tryToConnect();
    if (!connected)
        return false;

    if (!wiimote_is_open(wiimote))
    {
        wiimote_disconnect(wiimote);
        connected = tryToConnect();
    }
    if (!connected)
        return false;

    if (!wiimote_pending(wiimote))
        return true;

    if (wiimote_update(wiimote) < 0)
    {
        wiimote_disconnect(wiimote);
        connected = false;
        return false;
    }

    if (wiimote->keys.up)
        wheelcounter++;
    if (wiimote->keys.right)
        wheelcounter++;
    if (wiimote->keys.down)
        wheelcounter--;
    if (wiimote->keys.left)
        wheelcounter--;
#endif
    return true;
}

Wiimote::~Wiimote()
{
#ifdef HAVE_WIIMOTE
    if (connected)
        wiimote_disconnect(wiimote);
    delete wiimote;
    wiimote = NULL;
#endif
}

void
Wiimote::getButtons(int /* station */, unsigned int *button)
{
    *button = 0;
#ifdef HAVE_WIIMOTE
    update();
    if (wiimote->keys.b)
        *button |= 1;
    if (wiimote->keys.one)
        *button |= 2;
    if (wiimote->keys.a)
        *button |= 4;
    if (wiimote->keys.two)
        *button |= 8;
    if (wiimote->keys.plus)
        *button |= 0x10;
    if (wiimote->keys.minus)
        *button |= 0x20;
    if (wiimote->keys.home)
        *button |= 0x40;
    if (wiimote->keys.left)
        *button |= JOYSTICK_LEFT;
    if (wiimote->keys.right)
        *button |= JOYSTICK_RIGHT;
    if (wiimote->keys.up)
        *button |= JOYSTICK_UP;
    if (wiimote->keys.down)
        *button |= JOYSTICK_DOWN;
#endif
}

int Wiimote::getWheel(int /* station */)
{
    update();
    int ret = wheelcounter;
    wheelcounter = 0;
    return ret;
}
