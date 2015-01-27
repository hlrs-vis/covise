/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ===========================================================================
// i2c Master
// ===========================================================================
#define i2c_READ 0x01
#define i2c_WRITE 0x00

void i2c_init(void); // initialize the i2c master interface
void i2c_stop(void); // terminates the data transfer and releases the i2c bus
unsigned char i2c_start(unsigned char addr); // issues a start condition and sends address and transfer direction
unsigned char i2c_rep_start(unsigned char addr); // Issues a repeated start condition and sends address and transfer direction
void i2c_start_wait(unsigned char addr); // Issues a start condition and sends address and transfer direction
unsigned char i2c_write(unsigned char data); // Send one byte to I2C device
unsigned char i2c_readAck(void); // read one byte from the I2C device, request more data from device
unsigned char i2c_readNak(void); // read one byte from the I2C device, read is followed by a stop condition
unsigned char i2c_read(unsigned char ack); // read one byte from the I2C device
// Implemented as a macro, which calls either i2c_readAck or i2c_readNak

// I2C clock in Hz
#define SCL_CLOCK 100000L

// I2C timer max delay
#define I2C_TIMER_DELAY 0xFF

//---------------------------------------------------------------------------
// Initialization of the I2C bus interface. Need to be called only once
//---------------------------------------------------------------------------
void i2c_init(void)
{
    // initialize TWI clock: 100 kHz clock, TWPS = 0 => prescaler = 1

    TWSR = 0; // no prescaler
    TWBR = (uint8_t)((F_CPU / SCL_CLOCK) - 16) / 2; // must be > 10 for stable operation
}

//---------------------------------------------------------------------------
// Issues a start condition and sends address and transfer direction.
// return 0 = device accessible, 1= failed to access device
//---------------------------------------------------------------------------
unsigned char i2c_start(unsigned char address)
{
    uint32_t i2c_timer = 0;
    uint8_t twst;

    // send START condition
    TWCR = (1 << TWINT) | (1 << TWSTA) | (1 << TWEN);

    // wait until transmission completed
    i2c_timer = I2C_TIMER_DELAY;
    while (!(TWCR & (1 << TWINT)) /* && i2c_timer--*/)
        ;
    if (i2c_timer == 0)
    {
        return 1;
        //sbi(PORTC, LED_WHITE);
    }
    // check value of TWI Status Register. Mask prescaler bits.
    twst = TW_STATUS & 0xF8;
    if ((twst != TW_START) && (twst != TW_REP_START))
    {
        return 1;
    }
    // send device address
    TWDR = address;
    TWCR = (1 << TWINT) | (1 << TWEN);

    // wail until transmission completed and ACK/NACK has been received
    i2c_timer = I2C_TIMER_DELAY;
    while (!(TWCR & (1 << TWINT)) /* && i2c_timer--*/)
        ;
    if (i2c_timer == 0)
    {
        return 1;
    }
    // check value of TWI Status Register. Mask prescaler bits.
    twst = TW_STATUS & 0xF8;
    if ((twst != TW_MT_SLA_ACK) && (twst != TW_MR_SLA_ACK))
    {
        return 1;
    }
    return 0;
}

//-----------------------------------------------------------------------
// Issues a start condition and sends address and transfer direction.
// If device is busy, use ack polling to wait until device is ready
//
// Input:   address and transfer direction of I2C device
//-----------------------------------------------------------------------
void i2c_start_wait(unsigned char address)
{
    uint32_t i2c_timer = 0;
    uint8_t twst;

    while (1)
    {
        // send START condition
        TWCR = (1 << TWINT) | (1 << TWSTA) | (1 << TWEN);

        // wait until transmission completed
        i2c_timer = I2C_TIMER_DELAY;
        while (!(TWCR & (1 << TWINT)) && i2c_timer--)
            ;

        // check value of TWI Status Register. Mask prescaler bits.
        twst = TW_STATUS & 0xF8;
        if ((twst != TW_START) && (twst != TW_REP_START))
            continue;

        // send device address
        TWDR = address;
        TWCR = (1 << TWINT) | (1 << TWEN);

        // wail until transmission completed
        i2c_timer = I2C_TIMER_DELAY;
        while (!(TWCR & (1 << TWINT)) && i2c_timer--)
            ;

        // check value of TWI Status Register. Mask prescaler bits.
        twst = TW_STATUS & 0xF8;
        if ((twst == TW_MT_SLA_NACK) || (twst == TW_MR_DATA_NACK))
        {
            /* device busy, send stop condition to terminate write operation */
            TWCR = (1 << TWINT) | (1 << TWEN) | (1 << TWSTO);

            // wait until stop condition is executed and bus released
            i2c_timer = I2C_TIMER_DELAY;
            while ((TWCR & (1 << TWSTO)) && i2c_timer--)
                ;

            continue;
        }
        //if( twst != TW_MT_SLA_ACK) return 1;
        break;
    }
}

//---------------------------------------------------------------------------
// Issues a repeated start condition and sends address and transfer direction
//
// Input:   address and transfer direction of I2C device
//
// Return:  0 device accessible
//          1 failed to access device
//----------------------------------------------------------------------------
unsigned char i2c_rep_start(unsigned char address)
{
    return i2c_start(address);
}

//----------------------------------------------------------------------------
// Terminates the data transfer and releases the I2C bus
//----------------------------------------------------------------------------
void i2c_stop(void)
{
    uint32_t i2c_timer = 0;

    // send stop condition
    TWCR = (1 << TWINT) | (1 << TWEN) | (1 << TWSTO);

    // wait until stop condition is executed and bus released
    i2c_timer = I2C_TIMER_DELAY;
    while ((TWCR & (1 << TWSTO)) && i2c_timer--)
        ;
}

//---------------------------------------------------------------------------
//  Send one byte to I2C device
//
//  Input:    byte to be transfered
//  Return:   0 write successful
//            1 write failed
//---------------------------------------------------------------------------
unsigned char i2c_write(unsigned char data)
{
    uint32_t i2c_timer = 0;
    uint8_t twst;

    // send data to the previously addressed device
    TWDR = data;
    TWCR = (1 << TWINT) | (1 << TWEN);

    // wait until transmission completed
    i2c_timer = I2C_TIMER_DELAY;
    while (!(TWCR & (1 << TWINT)) && i2c_timer--)
        ;
    if (i2c_timer == 0)
        return 1;

    // check value of TWI Status Register. Mask prescaler bits
    twst = TW_STATUS & 0xF8;
    if (twst != TW_MT_DATA_ACK)
        return 1;
    return 0;
}

//-------------------------------------------------------------------------
// Read one byte from the I2C device, request more data from device
//
// Return:  byte read from I2C device
//-------------------------------------------------------------------------
unsigned char i2c_readAck(void)
{
    uint32_t i2c_timer = 0;

    TWCR = (1 << TWINT) | (1 << TWEN) | (1 << TWEA);
    i2c_timer = I2C_TIMER_DELAY;
    while (!(TWCR & (1 << TWINT)) && i2c_timer--)
        ;
    if (i2c_timer == 0)
        return 0;

    return TWDR;
}

//-------------------------------------------------------------------------
// Read one byte from the I2C device, read is followed by a stop condition
//
// Return:  byte read from I2C device
//-------------------------------------------------------------------------
unsigned char i2c_readNak(void)
{
    uint32_t i2c_timer = 0;

    TWCR = (1 << TWINT) | (1 << TWEN);
    i2c_timer = I2C_TIMER_DELAY;
    while (!(TWCR & (1 << TWINT)) && i2c_timer--)
        ;
    if (i2c_timer == 0)
        return 0;

    return TWDR;
}