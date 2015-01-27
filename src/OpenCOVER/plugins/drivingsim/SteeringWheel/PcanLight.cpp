/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PcanLight.h"
#ifdef HAVE_PCAN

PcanLight::PcanLight(HardwareType HWT, Baudrates BTR0BTR1)
{
    std::cout << "Connecting to PCAN PCI... ";

    HWType = HWT;
    if (CANLight::Init(HWType, BTR0BTR1, INIT_TYPE_ST) != ERR_OK)
    {
        std::cout << "error initializing CAN Interface!" << std::endl;
    }
}

PcanLight::~PcanLight()
{
    if (CANLight::Close(HWType) != ERR_OK)
    {
        std::cout << "error closing CAN Interface!" << std::endl;
    }
}

bool PcanLight::sendFrame(TPCANMsg &msg)
{

    int errorcode = (int)CANLight::Write(HWType, &msg);
    if (errorcode == 0)
    {
        //std::cerr << "Sent "; printMsg(msg);
        return true;
    }
    else
    {
        std::cerr << "sendFrame error, code: " << std::hex << errorcode << std::endl;
        return false;
    }
}

bool PcanLight::readFrame(TPCANMsg &msg)
{
    int errorcode = (int)CANLight::Read(HWType, &msg);
    if (errorcode != 0)
    {
        std::cerr << "readFrame error, code: " << std::hex << errorcode << std::endl;
        return false;
    }
    else
    {
        //std::cerr << "Received "; printMsg(msg);
        return true;
    }
}

void PcanLight::printMsg(TPCANMsg &msg)
{
    if (msg.MSGTYPE == MSGTYPE_EXTENDED)
    {
        std::cout << "extended ";
    }
    else if (msg.MSGTYPE == MSGTYPE_STANDARD)
    {
        std::cout << "standard ";
    }
    else if (msg.MSGTYPE == MSGTYPE_RTR)
    {
        std::cout << "rtr ";
    }
    else
    {
        std::cout << "pcan status ";
    }

    std::cout << "message: id=" << std::hex << msg.ID;

    if (msg.MSGTYPE != MSGTYPE_RTR)
    {
        std::cout << " len=" << (int)msg.LEN;
        if (msg.LEN)
        {
            std::cout << " data=";

            for (int j = 0; j < msg.LEN; j++)
            {
                std::cout << std::hex << (int)msg.DATA[j] << " ";
            }
        }
    }

    std::cout << std::endl;
}

#endif
