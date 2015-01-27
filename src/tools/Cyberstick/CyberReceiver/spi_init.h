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
    // DDB0 - CSN (SPI Chip select RFM70)

    DDRB |= (1 << DDB3) | (1 << DDB5) | (1 << DDB0);
    PORTB |= (1 << DDB0); // chip select RFM70 to high
    PORTB &= ~((1 << DDB3) | (1 << DDB5)); // MOSI & SCK to low
    DDRB &= ~(1 << DDB4);

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
    // SPR1 0 - set clock rate fck/4
    // SPR0 0
    // write SPSR
    // SPI2X 0 - disable double speed

    SPCR = (1 << SPE) | (1 << MSTR) | (1 << SPR0);
    SPSR &= ~(1 << SPI2X);
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
// spiSelect
// sets the CS pin
// ----------------------------------------------------------------------------
void spiSelect(uint8_t chip)
{
    switch (chip)
    {
    case csRFM73:
        cbi(PORTB, 0);
        break;
    case csNONE:
        sbi(PORTB, 0); // RFM70
        break;
    }
}