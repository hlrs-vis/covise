/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef HAVE_PCAN
#include "PcanPci.h"

PcanPci::PcanPci(unsigned char port, unsigned short speed)
{
    std::cout << "Connecting to PCAN PCI... ";
    pcanHandle = CAN_Open(HW_PCI, port);

    if (pcanHandle == NULL)
        std::cout << "error!" << std::endl;
    else if (!CAN_Init(pcanHandle, speed, CAN_INIT_TYPE_ST) == 0)
        std::cout << "error!" << std::endl;
    else if (!resetMsgFilter())
        std::cout << "error!" << std::endl;
    else
    {
        emptyReadQueue();
        std::cout << "done!" << std::endl;
    }
}

PcanPci::~PcanPci()
{
    std::cout << "Disconnecting!" << std::endl;
    CAN_Close(pcanHandle);
}

bool PcanPci::sendFrame(TPCANMsg &msg)
{
#ifdef WIN32
    int errorcode = CAN_Write(pcanHandle, &msg);
#else
    int errorcode = LINUX_CAN_Write_Timeout(pcanHandle, &msg, 100000);
#endif
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

bool PcanPci::readFrame(TPCANMsg &msg)
{
//while((getStatus() & 0x20)==0x20) std::cerr << "0";
//std::cerr << std::endl;
//std::cerr << "Status: " << std::hex << getStatus() << std::endl;
//std::cerr << "Error: " << std::hex << getError() << std::endl;
#ifdef WIN32
    int errorcode = CAN_Read(pcanHandle, &msg);
#else
    TPCANRdMsg rmsg;
    int errorcode = LINUX_CAN_Read_Timeout(pcanHandle, &rmsg, 100000);
    memcpy(&msg, &rmsg.Msg, sizeof(TPCANMsg));
#endif
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

void PcanPci::emptyReadQueue()
{
    TPCANMsg msg;
    while ((getStatus() & 0x20) != 0x20)
        CAN_Read(pcanHandle, &msg);
}

void PcanPci::printMsg(TPCANMsg &msg)
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

int PcanPci::getStatus()
{
    return CAN_Status(pcanHandle);
}

int PcanPci::getError()
{
    return nGetLastError();
}

bool PcanPci::setMsgFilter(int fromID, int toID)
{
    if (CAN_MsgFilter(pcanHandle, fromID, toID, MSGTYPE_STANDARD) == 0)
        return true;
    else
        return false;
}

bool PcanPci::resetMsgFilter()
{
    /*
	if(CAN_ResetFilter(pcanHandle)==0)
		return true;
	else
		return false;
	*/
    if (CAN_MsgFilter(pcanHandle, 0, 0x800, MSGTYPE_STANDARD) == 0)
        return true;
    else
        return false;
}
#endif
