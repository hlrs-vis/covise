/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "XenomaiPcanPci.h"

XenomaiPcanPci::XenomaiPcanPci(const std::string &device)
    : can(device)
{
}

XenomaiPcanPci::~XenomaiPcanPci()
{
    std::cout << "Disconnecting!" << std::endl;
}

bool XenomaiPcanPci::sendFrame(TPCANMsg &msg)
{
    can_frame msgFrame;
    msgFrame.can_id = msg.ID;
    msgFrame.can_dlc = msg.LEN;
    memcpy(msgFrame.data, msg.DATA, 8 * sizeof(uint8_t));

    can.sendFrame(msgFrame);

    return true;
}

bool XenomaiPcanPci::readFrame(TPCANMsg &msg)
{
    can_frame msgFrame;

    can.recvFrame(msgFrame);

    msg.ID = msgFrame.can_id;
    msg.LEN = msgFrame.can_dlc;
    msg.MSGTYPE = MSGTYPE_STANDARD;
    memcpy(msg.DATA, msgFrame.data, 8 * sizeof(uint8_t));

    return true;
}

void XenomaiPcanPci::emptyReadQueue()
{
    /*TPCANMsg msg;
		while((getStatus() & 0x20)!=0x20) CAN_Read(pcanHandle, &msg);*/
}

void XenomaiPcanPci::printMsg(TPCANMsg &msg)
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

int XenomaiPcanPci::getStatus()
{
    //return CAN_Status(pcanHandle);
    return 0;
}

int XenomaiPcanPci::getError()
{
    //return nGetLastError();
    return 0;
}

bool XenomaiPcanPci::setMsgFilter(int, int)
{
    /*if(CAN_MsgFilter(pcanHandle, fromID, toID, MSGTYPE_STANDARD)==0)
		return true;
	else
		return false;*/

    return true;
}

bool XenomaiPcanPci::resetMsgFilter()
{
    /*
	if(CAN_ResetFilter(pcanHandle)==0)
		return true;
	else
		return false;
	*/
    /*
	if(CAN_MsgFilter(pcanHandle, 0, 0x800, MSGTYPE_STANDARD)==0)
		return true;
	else
		return false;
   */
    return true;
}
