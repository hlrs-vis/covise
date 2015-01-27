/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//=========================================================================
//
// Project: PPM2USB, a AVR-based PPM to USB-Joystick Converter using V-USB.
// Author: Thomas Pfeifer
// Creation Date: 2010-09-10
//
// WWW: http://thomaspfeifer.net/ppm2usb_adapter.htm
//
// Copyright: (c) 2010, Thomas Pfeifer - www.thomaspfeifer.net
//
//            This software is free for non-commercial use. It may be copied,
//            modified, and redistributed provided that this copyright notice
//            is preserved on all copies.
//
//            You may NOT use this software, in whole or in part, in support
//            of any commercial product without the express consent of the
//            author.
//
//            There is no warranty or other guarantee of fitness of this
//            software for any purpose. It is provided solely "as is".
//
//=========================================================================

#include <avr/io.h>
#include <avr/wdt.h>
#include <avr/interrupt.h> /* for sei() */
#include <util/delay.h> /* for _delay_ms() */

#include <avr/pgmspace.h> /* required by usbdrv.h */
#include "usbdrv.h"
#include "oddebug.h"
#include "ppm.h"

/* ------------------------------------------------------------------------- */
/* ----------------------------- USB interface ----------------------------- */
/* ------------------------------------------------------------------------- */

PROGMEM char usbHidReportDescriptor[50] = { /* USB report descriptor, size must match usbconfig.h */
                                            0x05, 0x01, // USAGE_PAGE (Generic Desktop)
                                            0x15, 0x00, // LOGICAL_MINIMUM (0)
                                            0x26, 0xff, 0x00, // LOGICAL_MAXIMUM (255)
                                            0x75, 0x08, // REPORT_SIZE (6)
                                            0x09, 0x04, // USAGE (Joystick)
                                            0xa1, 0x01, // COLLECTION (Application)
                                            0x09, 0x01, //  USAGE (Pointer)
                                            0xa1, 0x00, //  COLLECTION (Physical)
                                            0x09, 0x30, // USAGE (X)
                                            0x09, 0x31, // USAGE (Y)
                                            0x95, 0x02, // REPORT_COUNT (2)
                                            0x81, 0x82, // INPUT (Data,Var,Abs,Vol)
                                            0xc0, //  END_COLLECTION
                                            0xa1, 0x00, //  COLLECTION (Physical)
                                            0x09, 0x32, // USAGE (Z)
                                            0x09, 0x33, // USAGE (Rx)
                                            0x95, 0x02, // REPORT_COUNT (2)
                                            0x81, 0x82, // INPUT (Data,Var,Abs,Vol)
                                            0xc0, //  END_COLLECTION
                                            0x09, 0x34, //  USAGE (Ry)
                                            0x09, 0x35, //  USAGE (Rz)
                                            0x09, 0x36, //  USAGE (Slider)
                                            0x09, 0x37, //  USAGE (Dial)
                                            0x95, 0x04, //  REPORT_COUNT (2)
                                            0x81, 0x82, //  INPUT (Data,Var,Abs,Vol)
                                            0xc0 // END_COLLECTION
};

static uchar reportBuffer[8];
static uchar idleRate; /* repeat rate for keyboards, never used for mice */

/* ------------------------------------------------------------------------- */

usbMsgLen_t usbFunctionSetup(uchar data[8])
{
    usbRequest_t *rq = (void *)data;

    if ((rq->bmRequestType & USBRQ_TYPE_MASK) == USBRQ_TYPE_CLASS)
    { /* class request type */
        DBG1(0x50, &rq->bRequest, 1); /* debug output: print our request */
        if (rq->bRequest == USBRQ_HID_GET_REPORT)
        { /* wValue: ReportType (highbyte), ReportID (lowbyte) */
            /* we only have one report type, so don't look at wValue */
            usbMsgPtr = (void *)&reportBuffer;
            return sizeof(reportBuffer);
        }
        else if (rq->bRequest == USBRQ_HID_GET_IDLE)
        {
            usbMsgPtr = &idleRate;
            return 1;
        }
        else if (rq->bRequest == USBRQ_HID_SET_IDLE)
        {
            idleRate = rq->wValue.bytes[1];
        }
    }
    else
    {
        /* no vendor specific requests implemented */
    }
    return 0; /* default for not implemented requests: return no data back to host */
}

/* ------------------------------------------------------------------------- */

int main(void)
{
    wdt_enable(WDTO_1S);

    odDebugInit();
    DBG1(0x00, 0, 0); /* debug output: main starts */

    ppmInit();

    usbInit();

    usbDeviceDisconnect(); /* enforce re-enumeration, do this while interrupts are disabled! */
    uchar i = 0;
    while (--i)
    { /* fake USB disconnect for > 250 ms */
        wdt_reset();
        _delay_ms(1);
    }
    usbDeviceConnect();

    sei();
    DBG1(0x01, 0, 0); /* debug output: main loop starts */

    uchar changed = 0;
    ppmNewData = 1;
    for (;;)
    { /* main event loop */
        DBG1(0x02, 0, 0); /* debug output: main loop iterates */
        wdt_reset();
        usbPoll();

        if (ppmNewData)
        {
            ppmNewData = 0;
            for (i = 0; i < sizeof(reportBuffer); i++)
            {
                unsigned char val = ppmGet(i);
                if (reportBuffer[i] != val)
                {
                    reportBuffer[i] = val;
                    changed = 1;
                }
            }
            if (changed)
            {
                if (usbInterruptIsReady())
                {
                    changed = 0;
                    // called after every poll of the interrupt endpoint
                    DBG1(0x03, 0, 0); // debug output: interrupt report prepared
                    usbSetInterrupt((void *)&reportBuffer, sizeof(reportBuffer));
                }
            }
        }
    }
}
