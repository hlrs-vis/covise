/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ============================================================================
// SPI BUS INIT
// ============================================================================

// ----------------------------------------------------------------------------
// spiInit
// init of spi bus, configures the spi controller
// ----------------------------------------------------------------------------
#define csNONE 0 // unselect all
#define csRFM73 1 // spi chip select  rfm73
#define csTOUCH 2 // spi chip select  Touchsensor

void spiInit(void)
{
    // SPI-Bus
    // DDB3 - MOSI
    // DDB4 - MISO
    // DDB5 - SCK
    // DDB7 - CSN (SPI Chip select RFM70)
    // DDB1 - CSN (Touchpad)

    cbi(DDRB, DDB4); // MISO in
    cbi(PORTB, DDB4); // MISO no  pullup

    sbi(DDRB, DDB1); // CSN out
    sbi(PORTB, DDB1); // chip select Touchpad high

    sbi(DDRB, DDB7); // CSN out
    sbi(PORTB, DDB7); // SPI Chip select RFM73

    sbi(DDRB, DDB5); // SCK out
    cbi(PORTB, DDB5); // SCK low

    sbi(DDRB, DDB3); // MOSI out
    cbi(PORTB, DDB3); // MOSI low

    sbi(DDRB, DDB2); // SS = Chip Enable out
    cbi(PORTB, DDB2); // chip Enable low

    cbi(DDRB, DDB0); // Data Ready in
    sbi(PORTB, DDB2); // Data Ready pullup on

    //////////////////////////////////////////////////////////////////////////
    // high level init
    // controller init, write registers

    // SPI Controller
    // write SPCR
    // SPIE 0 - SPI Interrupt disabled
    // SPE  1 - SPI enabled
    // DORD 0 - Data order is MSB is transmitted first
    // MSTR 1 - Attiny is master
    // CPOL 0 - clock polarity is rising edge
    // CPHA 0 - clock phase is sample/setup
    // SPR1 0 - set clock rate fck/8
    // SPR0 1
    // write SPSR
    // SPI2X 1 - enable double speed

    SPCR = (1 << SPE) | (1 << MSTR) | (1 << SPR0);
    //SPSR &= ~(1 << SPI2X);
    SPSR |= (1 << SPI2X);
}

// ----------------------------------------------------------------------------
// spiSendMsg
// sends byte to spi bus data register and waits for transmission
// todo: perhaps insert a timeout later?
// ----------------------------------------------------------------------------
uint8_t spiSendMsg(uint8_t data)
{
    // Start transmission
    SPDR = data;

    // wait for transmission complete
    // until SPIF is set to 1
    while (!(SPSR & (1 << SPIF)))
        ;

    // return result
    return SPDR;
}

// ----------------------------------------------------------------------------
// spiSendMsg16
// sends 2 bytes to spi bus data register and waits for transmission
// todo: perhaps insert a timeout later?
// ----------------------------------------------------------------------------
uint8_t spiSendMsg16(uint16_t data)
{
    // Start transmission
    SPDR = (data & 0xff);

    // wait for transmission complete
    // until SPIF is set to 1
    while (!(SPSR & (1 << SPIF)))
        ;

    // Start transmission
    SPDR = ((data >> 8) & 0xff);

    // wait for transmission complete
    // until SPIF is set to 1
    while (!(SPSR & (1 << SPIF)))
        ;

    // return result
    return SPDR;
}
// ----------------------------------------------------------------------------
// spiReadMaster
// sends two bytes to spi bus, waits for transmission and reads the two bytes sent back
// ----------------------------------------------------------------------------
uint8_t spiReadMaster(uint16_t *val, uint16_t send)
{

    // wait for transmission complete
    // until SPIF is set to 1
    uint16_t timeout = 0;
    // Start transmission
    SPDR = (send & 0xff);
    while (!(SPSR & (1 << SPIF)))
    {
        if (timeout > 50)
        {

            return 0; // nothing read
        }
        timeout++;
    }
    // return result
    uint8_t low = SPDR;
    timeout = 0;
    // Start transmission
    SPDR = ((send >> 8) & 0xff);
    while (!(SPSR & (1 << SPIF)))
    {
        if (timeout > 50)
        {

            return 1; // only one byte read
        }
        timeout++;
    }
    uint8_t high = SPDR;

    *val = high << 8 | low;
    return 2;
}
// ----------------------------------------------------------------------------
// spiSelect
// sets the CS pin
// ----------------------------------------------------------------------------
void spiSelect(uint8_t chip)
{
    switch (chip)
    {
    case csRFM73:
        cbi(PORTB, DDB7);
        sbi(PORTB, DDB1); // Touch
        break;
    case csTOUCH:
        cbi(PORTB, DDB1);
        sbi(PORTB, DDB7); // RFM70
        break;
    case csNONE:
        sbi(PORTB, DDB1); // Touch
        sbi(PORTB, DDB7); // RFM70
        break;
    }
}