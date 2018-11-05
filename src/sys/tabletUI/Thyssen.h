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
// button mapping:
// 1-1 2-2 ... <>-11 ><-12 bell-13 key-14 light-0
//
// ============================================================================

#ifndef THYSSEN_HEADER
#define THYSSEN_HEADER
#include <stdint.h>

// ----------------------------------------------------------------------------
//! class LEDController
// ----------------------------------------------------------------------------
class LEDController
{
private:

    uint8_t LEDMapping[16]={0,13,12,11,10,9,8,15,7,6,5,4,3,2,14,15};

    uint8_t mI2CAddress;
    int mFD;
    uint16_t mLEDStatus;

    void writeLEDStatus();

public:

    LEDController(uint8_t i2cAddress);
    ~LEDController();

    void setLED(uint8_t t, bool onoff);
    void setAllLEDs(bool onoff);
};

// ----------------------------------------------------------------------------
//! class KeyPadController
// ----------------------------------------------------------------------------
class KeyPadController
{
private:
    uint8_t mBtnStatus;

    void readBtnStatus();
    
public:
    KeyPadController();
    ~KeyPadController();

    uint8_t getBtnStatus();
    bool isPressed();
    void isrPressed();
};

// ----------------------------------------------------------------------------
//! class ThyssenPanel
// ----------------------------------------------------------------------------
class ThyssenPanel
{
private:

    
public:

    ThyssenPanel();
    ~ThyssenPanel();

    LEDController* led;
    KeyPadController* kpc;

};

#endif

// ----------------------------------------------------------------------------
