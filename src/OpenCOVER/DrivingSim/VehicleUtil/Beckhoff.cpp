/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Beckhoff.h"

//Beckhoff/////////////////////////////////////////////////////

// set protected static pointer for singleton to NULL
Beckhoff *Beckhoff::p_Beckhoff = NULL;

// constructor, destructor, instance ---------------------------------
Beckhoff::Beckhoff()
    : CanOpenDevice(*(CANProvider::instance()->p_CANOpenDisplay), (uint8_t)covise::coCoviseConfig::getInt("nodeID", "COVER.VehicleUtil.Beckhoff", 11))
{
    p_CANProv->registerDevice(this);
}

Beckhoff::~Beckhoff()
{
    p_Beckhoff = NULL;
}

// singleton
Beckhoff *Beckhoff::instance()
{
    if (p_Beckhoff == NULL)
    {
        p_Beckhoff = new Beckhoff();
    }
    return p_Beckhoff;
}
//--------------------------------------------------------------------

// Beckhoff::Beckhoff(CanOpenController& con, uint8_t id)
//    :  CanOpenDevice(con, id)
// {
// }

// public methods ----------------------------------------------------
void Beckhoff::initCANOpenDevice()
{
    std::cerr << "init Beckhoff" << std::endl;
    rt_task_sleep(100000000);
    enterPreOp();
    rt_task_sleep(100000000);
    //uint8_t transType = 254;
    //uint8_t transType = 0x1; //1. TPDO transmission type: synchronous after one SYNC
    /*if(!writeSDO(0x1a00, 2, &transType, 1)) {
      std::cerr << "1. tpdo transmission type set failed" << std::endl;
   }*/
    /*if(!writeSDO(0x1800, 2, &transType, 1)) {
      std::cerr << "1. tpdo transmission type set failed" << std::endl;
   }*/

    RPDODigital[0] = 0; //all outputs off
    RPDODigital[1] = 0; //all outputs off
    RPDODigital[2] = 0; //all outputs off
    RPDODigital[3] = 0; //all outputs off
    RPDODigital[4] = 0; //all outputs off
    RPDODigital[5] = 0; //all outputs off
    RPDOAnalog[0] = 0; //all outputs off
    RPDOAnalog[1] = 0; //all outputs off
    RPDOAnalog[2] = 0; //all outputs off
    RPDOAnalog[3] = 0; //all outputs off
    RPDOAnalog[4] = 0; //all outputs off
    RPDOAnalog[5] = 0; //all outputs off
    RPDOAnalog[6] = 0; //all outputs off
    RPDOAnalog[7] = 0; //all outputs off
    writeRPDO(1, RPDODigital, 6);
    startNode();
}

bool Beckhoff::getDigitalIn(int module, int port)
{
    TPDODigital = readTPDO(1);
    return (TPDODigital[module] & (1 << port));
}

uint8_t Beckhoff::getDigitalIn(int module)
{
    TPDODigital = readTPDO(1);
    return TPDODigital[module];
}

void Beckhoff::setDigitalOut(int module, int port, bool state)
{
    if (state)
        RPDODigital[module] |= (1 << port);
    else
        RPDODigital[module] &= ~(1 << port);
    writeRPDO(1, RPDODigital, 6);
}

float Beckhoff::getAnalogIn(int module, int port)
{

    return ((float *)TPDOAnalog)[module + port];
}

void Beckhoff::setAnalogOut(int module, int port, float voltage)
{
    *((unsigned short int *)(&RPDOAnalog[module + (port * 2)])) = (unsigned short int)((voltage / 10.0) * 0x7fff);
    writeRPDO(2, RPDOAnalog, 8);
}
//--------------------------------------------------------------------
