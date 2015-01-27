/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ============================================================================
// RFM70
// ============================================================================

// ----------------------------------------------------------------------------
// rfm70 constants
// ----------------------------------------------------------------------------
#define RFM70_FIFO_STATUS_TX_REUSE 0x40
#define RFM70_FIFO_STATUS_TX_FULL 0x20
#define RFM70_FIFO_STATUS_TX_EMPTY 0x10
#define RFM70_FIFO_STATUS_RX_FULL 0x02
#define RFM70_FIFO_STATUS_RX_EMPTY 0x01

#define RFM70_CARRIER_DETECTION 0x01

// ----------------------------------------------------------------------------
// interrupt status
// ----------------------------------------------------------------------------
#define RFM70_IRQ_STATUS_RX_DR 0x40 // Status bit RX_DR IRQ
#define RFM70_IRQ_STATUS_TX_DS 0x20 // Status bit TX_DS IRQ
#define RFM70_IRQ_STATUS_MAX_RT 0x10 // Status bit MAX_RT IRQ

#define RFM70_IRQ_STATUS_TX_FULL 0x01

// ----------------------------------------------------------------------------
// rfm70 commands
// ----------------------------------------------------------------------------
#define RFM70_CMD_WRITE_REG 0x20 // write register
#define RFM70_CMD_WR_TX_PLOAD 0xA0 // Define TX payload command
#define RFM70_CMD_W_TX_PAYLOAD_NOACK 0xb0 // Define TX payload NOACK command
#define RFM70_CMD_W_ACK_PAYLOAD 0xa8 // Define Write ACK command
#define RFM70_CMD_RX_PL_WID 0x60 // Define received payload width command
#define RFM70_CMD_RD_RX_PLOAD 0x61 // Define RX payload command

// ----------------------------------------------------------------------------
// rfm70 register addresses BANK 0
// ----------------------------------------------------------------------------
#define RFM70_REG_FEATURE 0x1d // feature register
#define RFM70_REG_STATUS 0x07 // status register
#define RFM70_REG_CONFIG 0x00 // config register
#define RFM70_REG_FIFO_STATUS 0x17 // 'FIFO Status Register' register address
#define RFM70_REG_CD 0x09 // 'Carrier Detect' register address

// ----------------------------------------------------------------------------
// rfm70 command sequences
// ----------------------------------------------------------------------------
const uint8_t PROGMEM RFM70_CMD_ACTIVATE_SWITCH[] = { 0x50, 0x53 }; // switch Register Bank
const uint8_t PROGMEM RFM70_CMD_ACTIVATE_FEATURE[] = { 0x50, 0x73 }; // activation command
const uint8_t PROGMEM RFM70_CMD_FLUSH_RX[] = { 0xE2, 0x00 }; // flush RX FIFO
const uint8_t PROGMEM RFM70_CMD_FLUSH_TX[] = { 0xE1, 0x00 }; // flush TX FIFO

// ----------------------------------------------------------------------------
// rfm70 bank register initialization
// ----------------------------------------------------------------------------
const uint8_t PROGMEM RFM70_Bank0Init[][2] = {
    { (0x20 | 0x00), 0x0A }, // 7:0 only 0 allowed
    // 6:0 reflect RX_DR as active low interrupt on the IR pin
    // 5:0 reflect TX_TS as active low interrupt on the IR pin
    // 4:0 reflect MAX_RT as active low interrupt on the IR pin
    // 3:1 enable CRC
    // 2:0 CRC code scheme is '2 bytes'
    // 1:1 POWER_UP
    // 0:0 Primary RX (to save power?)
    { (0x20 | 0x01), 0x00 }, // disable auto acknowledgement data pipe 0-5   //TODO: enable it, later
    { (0x20 | 0x02), 0x03 }, // enable RX addresses data pipe 0,1
    { (0x20 | 0x03), 0x03 }, // RX/TX address field width is 5 bytes
    { (0x20 | 0x04), 0x00 }, // re-Transmit disabled
    { (0x20 | 0x05), 0x01 }, // RF Channel 23 = 2400 MHz + 1*1 MHz = 2401 MHz
    { (0x20 | 0x06), 0x37 }, // 7-4 fix
    // 3  :0  air data rate = 1Mbps
    // 2,1:11 output power 5dbm (max)
    // 0  :1  setup LNA high gain
    { (0x20 | 0x07), 0x70 }, // status register, writing 1 resets IRQ pins
    { (0x20 | 0x08), 0x00 }, // observe TX (read only)
    { (0x20 | 0x09), 0x00 }, // carrier detection (read only)
    { (0x20 | 0x0A), 0x00 }, // receive address data pipe 0, 5 bytes (not defined here)
    { (0x20 | 0x0B), 0x00 }, // receive address data pipe 1, 5 bytes (not defined here)
    { (0x20 | 0x0C), 0xC3 }, // receive address data pipe 2
    { (0x20 | 0x0D), 0xC4 }, // receive address data pipe 3
    { (0x20 | 0x0E), 0xC5 }, // receive address data pipe 4
    { (0x20 | 0x0F), 0xC6 }, // receive address data pipe 5
    { (0x20 | 0x10), 0xC6 }, // transmit address data pipe 0, 5 bytes (not defined here)
    // should be equal to RX P0 to handle auto ACK
    { (0x20 | 0x11), 0x20 }, // number of bytes in RX payload in data pipe 0 - 3 bytes
    { (0x20 | 0x12), 0x20 }, // number of bytes in RX payload in data pipe 1 - 3 bytes
    { (0x20 | 0x13), 0x20 }, // number of bytes in RX payload in data pipe 2 - 3 bytes
    { (0x20 | 0x14), 0x20 }, // number of bytes in RX payload in data pipe 3 - 3 bytes
    { (0x20 | 0x15), 0x20 }, // number of bytes in RX payload in data pipe 4 - 3 bytes
    { (0x20 | 0x16), 0x20 }, // number of bytes in RX payload in data pipe 5 - 3 bytes
    { (0x20 | 0x17), 0x20 }, // FIFO status register
    // 7:0 fix
    // 6:0 TX_REUSE - retransmit as long as CE is high (read only)
    // 5:0 TX FIFO full flag
    // 4:1 TX FIFO empty flag
    // 3:0 fix
    // 2:0 fix
    // 1:0 RX FIFO full flag
    // 0:1 RX FIFO empty flag
    // 1C using default - dynamic payload disabled
    // 1D using default - feature register
};

const uint8_t PROGMEM RFM70_Feature[][2] = {
    { (0x20 | 0x1C), 0x00 }, // using default - dynamic payload disabled
    { (0x20 | 0x1D), 0x00 } // disable dynamic payload length no payload with ACK
};

const uint8_t PROGMEM RFM70_Bank1Init_PART1[][5] = {
    { (0x20 | 0x00), 0x40, 0x4B, 0x01, 0xE2 }, // fix
    { (0x20 | 0x01), 0xC0, 0x4B, 0x00, 0x00 }, // fix
    { (0x20 | 0x02), 0xD0, 0xFC, 0x8C, 0x02 }, // fix
    { (0x20 | 0x03), 0x99, 0x00, 0x39, 0x41 }, // fix
    { (0x20 | 0x04), 0xD9, 0x9E, 0x86, 0x0B }, // exampe=B9? another example=F9? datasheet=D9
    { (0x20 | 0x05), 0x3C, 0x02, 0x7F, 0xA6 }, // fix (enable RSSI)
    { (0x20 | 0x06), 0x00, 0x00, 0x00, 0x00 }, // fix
    { (0x20 | 0x07), 0x00, 0x00, 0x00, 0x00 }, // fix
    { (0x20 | 0x08), 0x00, 0x00, 0x00, 0x00 }, // fix
    { (0x20 | 0x09), 0x00, 0x00, 0x00, 0x00 }, // fix
    { (0x20 | 0x0A), 0x00, 0x00, 0x00, 0x00 }, // fix
    { (0x20 | 0x0B), 0x00, 0x00, 0x00, 0x00 }, // fix
    { (0x20 | 0x0C), 0x00, 0x12, 0x73, 0x00 }, // fix
    { (0x20 | 0x0D), 0x36, 0xB4, 0x80, 0x00 }, // fix
    { (0x20 | 0x0E), 0x00, 0x00, 0x00, 0x00 }, // fix
    { (0x20 | 0x0F), 0x00, 0x00, 0x00, 0x00 } // fix
};

const uint8_t PROGMEM RFM70_Bank1_TOGGLE1[] = { (0x20 | 0x04), 0xD9 | 0x06, 0x9E, 0x86, 0x0B }; // set bits
const uint8_t PROGMEM RFM70_Bank1_TOGGLE2[] = { (0x20 | 0x04), 0xD9 & ~0x06, 0x9E, 0x86, 0x0B }; // clear bits

const uint8_t PROGMEM RFM70_Bank1Init_PART2[] = {
    (0x20 | 0x0E), 0x41, 0x20, 0x08, 0x04, 0x81, 0x20, 0xCF, 0xF7, 0xFE, 0xFF, 0xFF // (datasheet)
};

// ----------------------------------------------------------------------------
// address definition commands
// ----------------------------------------------------------------------------
const uint8_t PROGMEM RFM70_ADR_RX0[] = { (0x20 | 0x0A), 0x66, 0x88, 0x68, 0x68, 0x68 };
const uint8_t PROGMEM RFM70_ADR_RX1[] = { (0x20 | 0x0B), 0x88, 0x66, 0x86, 0x86, 0x86 };
const uint8_t PROGMEM RFM70_ADR_TX[] = { (0x20 | 0x10), 0x66, 0x88, 0x68, 0x68, 0x68 };

// ----------------------------------------------------------------------------
// rfm70ReadRegValue
// Read register value over SPI bus
// ----------------------------------------------------------------------------
uint8_t rfm70ReadRegValue(uint8_t cmd)
{
    uint8_t result;

    spiSelect(csRFM70);
    _delay_ms(0);

    spiSendMsg(cmd);
    result = spiSendMsg(0);

    spiSelect(csNONE);
    _delay_ms(0);

    return (result);
}

// ----------------------------------------------------------------------------
// rfm70WriteRegValue
// Write register value over SPI bus
// write command is 001AAAAAb so mask register addresses with 0x20
// register only can be written to in power down or standby mode
// ----------------------------------------------------------------------------
void rfm70WriteRegValue(uint8_t cmd, uint8_t val)
{
    spiSelect(csRFM70);
    _delay_ms(0);

    spiSendMsg(cmd);
    spiSendMsg(val);

    spiSelect(csNONE);
    _delay_ms(0);
}

// ----------------------------------------------------------------------------
// rfm70ReadRegPgmBuf
// Read register from pgm buffer
// ----------------------------------------------------------------------------
void rfm70ReadRegPgmBuf(uint8_t reg, uint8_t *buf, uint8_t len)
{

    uint8_t byte_ctr;

    spiSelect(csNONE);
    _delay_ms(0);
    spiSendMsg(reg); // Select register to write, and read status UINT8
    for (byte_ctr = 0; byte_ctr < len; byte_ctr++)
        buf[byte_ctr] = spiSendMsg(0); // Perform SPI_RW to read UINT8 from RFM70
    spiSelect(csNONE);
    _delay_ms(0);
}

// ----------------------------------------------------------------------------
// rfm70WriteRegPgmBuf
// Write register from pgm buffer
// ----------------------------------------------------------------------------
void rfm70WriteRegPgmBuf(uint8_t *cmdbuf, uint8_t length)
{
    spiSelect(csRFM70);
    _delay_ms(0);

    while (length--)
    {
        spiSendMsg(pgm_read_byte(cmdbuf++));
    }

    spiSelect(csNONE);
    _delay_ms(0);
}

// ----------------------------------------------------------------------------
// rfm70SetCE
// sets the CE pin to enable/disable the chip
// ----------------------------------------------------------------------------
void rfm70SetCE(uint8_t status)
{
    if (status == 0)
    {
        // clear bit
        PORTB &= ~(1 << DDB6);
    }
    else
    {
        // set bit
        PORTB |= (1 << DDB6);
    }
}

// ----------------------------------------------------------------------------
// rfm70SelectBank
// There are two register banks, which can be toggled by SPI command ACTIVATE
// followed with 0x53 byte, and bank status can be read from REG7.
// So we read REG7 from the active Bank and mask the activated Bank ID.
// If its not the Bank we'd like to select, we should toggle banks.
// ----------------------------------------------------------------------------
void rfm70SelectBank(uint8_t bank)
{
    uint8_t tmp;

    tmp = rfm70ReadRegValue(0x07); // read REG7 from activated bank
    tmp &= 0x80; // mask register bank selection states
    // 0: register bank 0
    // 1: register bank 1
    if (bank)
    {
        if (!tmp)
        {
            // bank=1 & tmp==0
            rfm70WriteRegPgmBuf((uint8_t *)RFM70_CMD_ACTIVATE_SWITCH, sizeof(RFM70_CMD_ACTIVATE_SWITCH));
        }
    }
    else
    {
        if (tmp)
        {
            // bank=0 & tmp==1
            rfm70WriteRegPgmBuf((uint8_t *)RFM70_CMD_ACTIVATE_SWITCH, sizeof(RFM70_CMD_ACTIVATE_SWITCH));
        }
    }
}

// ----------------------------------------------------------------------------
// rfm70InitRegisters
// selecting bank 0 writing bank 0 config then
// selecting bank 1 writing bank 1 config then
// just to check that the rfm70 is present return chipID register == 0x63
// ----------------------------------------------------------------------------
uint8_t rfm70InitRegisters(void)
{
    // register only can be written to in power down or standby mode
    // TODO goto standby mode

    // select register bank 0
    rfm70SelectBank(0);

    // write init values of the first 8 registers
    for (int i = 0; i < 0x17; i++)
    {
        rfm70WriteRegValue(pgm_read_byte(&RFM70_Bank0Init[i][0]),
                           pgm_read_byte(&RFM70_Bank0Init[i][1]));
    }

    // write address registers in bank 0
    rfm70WriteRegPgmBuf((uint8_t *)RFM70_ADR_RX0, sizeof(RFM70_ADR_RX0));
    rfm70WriteRegPgmBuf((uint8_t *)RFM70_ADR_RX1, sizeof(RFM70_ADR_RX1));
    rfm70WriteRegPgmBuf((uint8_t *)RFM70_ADR_TX, sizeof(RFM70_ADR_TX));

    // activate feature register
    if (!rfm70ReadRegValue(RFM70_REG_FEATURE))
    {
        rfm70WriteRegPgmBuf((uint8_t *)RFM70_CMD_ACTIVATE_FEATURE, sizeof(RFM70_CMD_ACTIVATE_FEATURE));
    }

    // now set the feature registers 1D and 1C
    rfm70WriteRegValue(pgm_read_byte(&RFM70_Feature[1][0]), pgm_read_byte(&RFM70_Feature[1][1]));
    rfm70WriteRegValue(pgm_read_byte(&RFM70_Feature[0][0]), pgm_read_byte(&RFM70_Feature[0][1]));

    // select register bank 1
    rfm70SelectBank(1);

    // write first part (registers 00 - 0D)
    for (int i = 0; i < 14; i++)
    {
        rfm70WriteRegPgmBuf((uint8_t *)RFM70_Bank1Init_PART1[i], sizeof(RFM70_Bank1Init_PART1[i]));
    }

    // write 2nd part (registers 0E, 0F)
    //for (int i=0; i < 2; i++)
    //{
    //	rfm70WriteRegPgmBuf((uint8_t*)RFM70_Bank1Init_PART2[i], sizeof(RFM70_Bank1Init_PART2[i]));
    //}

    // set ramp curve
    rfm70WriteRegPgmBuf((uint8_t *)RFM70_Bank1Init_PART2, sizeof(RFM70_Bank1Init_PART2));

    // do we have to toggle some bits here like in the example code?
    rfm70WriteRegPgmBuf((uint8_t *)RFM70_Bank1_TOGGLE1, sizeof(RFM70_Bank1_TOGGLE1));
    rfm70WriteRegPgmBuf((uint8_t *)RFM70_Bank1_TOGGLE2, sizeof(RFM70_Bank1_TOGGLE2));

    _delay_ms(100);

    // read chipID just to be sure, someone is there ;)
    uint8_t chipid = rfm70ReadRegValue(0x08);

    if (!(chipid == 0x63))
    {
        return false;
    }

    // its important to keep bank0 active
    rfm70SelectBank(0);

    // set ModeRX

    return true;
}

// ----------------------------------------------------------------------------
// rfm70SetModeRX
// configures and starts RX Mode
// ----------------------------------------------------------------------------
void rfm70SetModeRX(void)
{
    uint8_t value;

    // Flush RX FIFO
    rfm70WriteRegPgmBuf((uint8_t *)RFM70_CMD_FLUSH_RX, sizeof(RFM70_CMD_FLUSH_RX));

    // reset/clear interrupt status register
    value = rfm70ReadRegValue(RFM70_REG_STATUS);
    rfm70WriteRegValue(RFM70_CMD_WRITE_REG | RFM70_REG_STATUS, value);

    // switch to RX mode
    rfm70SetCE(0);
    value = rfm70ReadRegValue(RFM70_REG_CONFIG);
    value |= 0x02; // set PWR_UP bit
    value |= 0x01; // set RX bit

    rfm70WriteRegValue(RFM70_CMD_WRITE_REG | RFM70_REG_CONFIG, value);
    rfm70SetCE(1);
}

// ----------------------------------------------------------------------------
// rfm70SetModeTX
// configures and starts TX Mode
// ----------------------------------------------------------------------------
void rfm70SetModeTX(void)
{
    uint8_t value;

    // Flush TX FIFO
    rfm70WriteRegPgmBuf((uint8_t *)RFM70_CMD_FLUSH_TX, sizeof(RFM70_CMD_FLUSH_TX));

    // reset/clear interrupt status register
    value = rfm70ReadRegValue(RFM70_REG_STATUS);
    value |= 0x70;
    rfm70WriteRegValue(RFM70_CMD_WRITE_REG | RFM70_REG_STATUS, value);

    // switch CE=0 rfm goes to standby-1 mode in
    rfm70SetCE(0);
    value = rfm70ReadRegValue(RFM70_REG_CONFIG);
    value &= 0xFE; // clear RX bit (= TX enabled)
    //value |= 0x02; // set PWR_UP bit
    rfm70WriteRegValue(RFM70_CMD_WRITE_REG | RFM70_REG_CONFIG, value);
    // now go to TX mode
    rfm70SetCE(1);
}
