// ============================================================================
//
// Power Supply
// GPIO Pin 01 - 3V3
// GPIO Pin 30 - GND
// GPIO Pin 39 - GND
// 
// KeyPadController 74C922N
// 74C922N          -   GPIO
// -------------------------
// Pin 12 (DATA AV) - Pin 08   -> mod changed Pin 08 to Pin 11
// Pin 13 (-OE)     -    GND
// Pin 14 (OUT A)   - Pin 18
// Pin 15 (OUT B)   - Pin 16
// Pin 16 (OUT C)   - Pin 12
// Pin 17 (OUT D)   - Pin 10
// Pin 06 (KBM)     -   n.c.
//
// I/O Port Expander MCP23017
// i2c address = 0x20
// MCP23017        -   GPIO
// ------------------------
// Pin 12 (SCL)    - Pin 05
// Pin 13 (SDA)    - Pin 03
// Pin 15 (A0)     -    GND
// Pin 16 (A1)     -    GND
// Pin 17 (A2)     -    GND
// Pin 18 (-Reset) -    VDD
// Pin 20 (INT A)  -   n.c.
// Pin 20 (INT B)  -   n.c.
//
// MCP23017 register
// ---------------------
// 0x00 - IODIRA
// 0x01 - IODIRB
// 0x12 - GPIOA
// 0x13 - GPIOB
//
// LED Strip Driver ULN2003AN -> MOSFET
// GPIO 36 - Red   (? using softPWM)
// GPIO 38 - Green (? using softPWM)
// GPIO 40 - Blue  (? using softPWM)
//
//
// led <-> matrix old
// [ 00 | 01 | 03 | 02 ]
// [ 04 | 05 | 07 | 06 ]
// [ 15 | 08 | 10 | 09 ]
// [ 11 | 12 | 14 | 13 ]
//
// matrix <-> btn old
// 01 - 14    06 - 09   >|< - 01
// 02 - 13    07 - 08   <|> - 03
// 03 - 12    08 - 07
// 04 - 11    09 - 05
// 05 - 10    10 - 04

// now 1-1 2-2 ... >< 12 <> 11 bell 13 key 14 light 0
//
// ============================================================================

#include <stdio.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include <iostream>

#include <wiringPi.h>
#include <wiringPiI2C.h>
#include "Thyssen.h"

using namespace std;

// ============================================================================

// MCP23017 register
const uint8_t MCP23017_IODIRA = 0x00;
const uint8_t MCP23017_IODIRB = 0x01;
const uint8_t MCP23017_GPIOA = 0x12;
const uint8_t MCP23017_GPIOB = 0x13;

// ----------------------------------------------------------------------------
//! LEDController::LEDController()
// ----------------------------------------------------------------------------
LEDController::LEDController(uint8_t i2cAddress)
    : mI2CAddress(i2cAddress),
      mLEDStatus(0x0)
{
    wiringPiSetup();
     
    mFD = wiringPiI2CSetup(mI2CAddress);
    
    if (mFD == -1)
    {
        cout << "error: could not setup i2c device " << mI2CAddress << endl;
        return;
    }
    
    if ((wiringPiI2CWriteReg8(mFD, MCP23017_IODIRA, 0x00) < 0) ||
        (wiringPiI2CWriteReg8(mFD, MCP23017_IODIRB, 0x00) < 0))
    {
        cout << "error: could not write init to MCP23017" << endl;
        return;
    }
}

// ----------------------------------------------------------------------------
//! LEDController::~LEDController()
// ----------------------------------------------------------------------------
LEDController::~LEDController()
{
    mLEDStatus = 0x0;
    writeLEDStatus();
}

// ----------------------------------------------------------------------------
//! void LEDController::writeLEDStatus()
// ----------------------------------------------------------------------------
void LEDController::writeLEDStatus()
{
    uint8_t a = mLEDStatus & 0xFF;
    uint8_t b = (mLEDStatus >> 8) & 0xFF;
    
    if ((wiringPiI2CWriteReg8(mFD, MCP23017_GPIOA, a) < 0) ||
        (wiringPiI2CWriteReg8(mFD, MCP23017_GPIOB, b) < 0))
    {
        cout << "error: could not write data to MCP23017" << endl;
        return;
    }
}

// ----------------------------------------------------------------------------
//! void LEDController::setLED()
//! sets led 0..15 on(1) or off(0)
// ----------------------------------------------------------------------------
void LEDController::setLED(uint8_t led, bool onoff)
{
    assert(led <= 15);

    if (onoff == true)
    {
        mLEDStatus |= (1 << LEDMapping[led]);
    }
    else
    {
        mLEDStatus &= ~(1 << LEDMapping[led]);
    }
    
    writeLEDStatus();
}

// ----------------------------------------------------------------------------
//! void LEDController::setLED()
// ----------------------------------------------------------------------------
void LEDController::setAllLEDs(bool onoff)
{
    if (onoff == true)
    {
        mLEDStatus = 0xFFFF;
    }
    else
    {
        mLEDStatus = 0X0;
    }
    
    writeLEDStatus();
}

// ============================================================================

// 74C922N GPIO Pins
const uint8_t IC74C922N_DATA_AV = 0; // Pin 11 | GPIO. 0 | 0 | 17 |
const uint8_t IC74C922N_OUT_A = 5; // Pin 18 | GPIO. 5 | 5   | 24  |
const uint8_t IC74C922N_OUT_B = 4; // Pin 16 | GPIO. 4 | 4   | 23  |
const uint8_t IC74C922N_OUT_C = 1; // Pin 12 | GPIO. 1 | 1   | 18  |
const uint8_t IC74C922N_OUT_D = 16; // Pin 10 | RxD     | 16  | 15  |

void (KeyPadController::*pHandler)() = NULL; 
KeyPadController *pKPC = NULL;

void isrHandler()
{
    if ((pHandler != NULL) && (pKPC != NULL))
    {
        (*pKPC.*pHandler)();
    }
}

// ----------------------------------------------------------------------------
//! KeyPadController::KeyPadController()
// ----------------------------------------------------------------------------
KeyPadController::KeyPadController()
    : mBtnStatus(0x0)
{

    wiringPiSetup();

    pinMode(IC74C922N_DATA_AV, INPUT);
    pinMode(IC74C922N_OUT_A, INPUT);
    pinMode(IC74C922N_OUT_B, INPUT);
    pinMode(IC74C922N_OUT_C, INPUT);
    pinMode(IC74C922N_OUT_D, INPUT);

    if (wiringPiISR (IC74C922N_DATA_AV, INT_EDGE_RISING, &isrHandler) < 0)
    {
        cout << "error: could not setup ISR" << endl;
    }

    pHandler = &KeyPadController::isrPressed;
    pKPC = this;
}

// ----------------------------------------------------------------------------
//! KeyPadController::~KeyPadController()
// ----------------------------------------------------------------------------
KeyPadController::~KeyPadController()
{
}

// ----------------------------------------------------------------------------
//! void readBtnStatus()
// ----------------------------------------------------------------------------
void KeyPadController::readBtnStatus()
{
    mBtnStatus =
        (digitalRead(IC74C922N_OUT_A)) |
        (digitalRead(IC74C922N_OUT_B) << 1) |
        (digitalRead(IC74C922N_OUT_C) << 2) |
        (digitalRead(IC74C922N_OUT_D) << 3);
}

// ----------------------------------------------------------------------------
//! uint8_t getBtnStatus()
// ----------------------------------------------------------------------------
uint8_t KeyPadController::getBtnStatus()
{
    readBtnStatus();
    
    return mBtnStatus;
}

// ----------------------------------------------------------------------------
//! bool isPressed();
// ----------------------------------------------------------------------------
bool KeyPadController::isPressed()
{
    return digitalRead(IC74C922N_DATA_AV);
}

// ----------------------------------------------------------------------------
//! bool isrPressed();
// ----------------------------------------------------------------------------
void KeyPadController::isrPressed()
{
    readBtnStatus();

    cout << int(mBtnStatus) << endl;
}

// ============================================================================

const uint8_t MCP23017_ADDRESS = 0x20;
const int iBtnArray[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

// ----------------------------------------------------------------------------
//! ThyssenPanel::ThyssenPanel()
// ----------------------------------------------------------------------------
ThyssenPanel::ThyssenPanel()
{

    led = new LEDController(MCP23017_ADDRESS);
    kpc = new KeyPadController();
    
    return;
}

// ----------------------------------------------------------------------------
//! ThyssenPanel::~ThyssenPanel()
// ----------------------------------------------------------------------------
ThyssenPanel::~ThyssenPanel()
{

    delete kpc;
    delete led;
    
    return;
}

// ----------------------------------------------------------------------------
