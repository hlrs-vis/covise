/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CanOpenBus.h"
#include <cstring>

#ifdef HAVE_PCAN
CanOpenBus::CanOpenBus(CanInterface *can)
{
    this->can = can;

    initMsg();
}

bool CanOpenBus::sendNMT(unsigned char id, unsigned char cs)
{
    msgNMT.DATA[0] = cs;
    msgNMT.DATA[1] = id;

    if (!can->sendFrame(msgNMT))
        return false;
    else
        return true;
}

bool CanOpenBus::sendRTR(unsigned short cob, unsigned char length)
{
    msgRTR.ID = cob;
    msgRTR.LEN = length;
    if (!can->sendFrame(msgRTR))
        return false;
    else
        return true;
}

bool CanOpenBus::sendSYNC()
{
    if (!can->sendFrame(msgSYNC))
        return false;
    else
        return true;
}

TPCANMsg *CanOpenBus::readObject(unsigned char id, unsigned short index, unsigned char subindex)
{
    TPCANMsg *msg = sendSDO(id, false, index, subindex, 0, NULL);
    if (msg && (msg->DATA[0] >> 5) == 2)
        return &recvMsg;
    else
        return NULL;
}

bool CanOpenBus::writeObject(unsigned char id, unsigned short index, unsigned char subindex, unsigned char length, unsigned char *data)
{
    TPCANMsg *msg = sendSDO(id, true, index, subindex, length, data);
    //can->printMsg(*msg);
    //std::cout << "Compare: " << std::hex << idSDOtx << " " << 0x60 << " " << (index & 0xff) << " " << (index >> 8) << std::endl;
    if (msg && (msg->ID == (unsigned int)(0x580 + id)) && (msg->DATA[0] == (unsigned char)0x60) && (msg->DATA[1] == (unsigned char)(index & 0xff)) && (msg->DATA[2] == (unsigned char)(index >> 8)))
        //if( (msg->ID==(0x580 + id)) && ( *(unsigned short *)(&(msg->DATA[1])) == index ) )
        return true;
    else
    {
        if (msg)
        {
            std::cout << "Writing object failed: Received ";
            can->printMsg(*msg);
        }
        else
        {
            std::cout << "Writing object failed: Received nothing\n";
        }
        return false;
    }
}

bool CanOpenBus::sendPDO(unsigned char id, unsigned char pdonum, unsigned char length, unsigned char *data)
{
    msgPDO.ID = (pdonum + 1) * 0x100 + id;
    msgPDO.LEN = length;
    memcpy(msgPDO.DATA, data, sizeof(unsigned char) * 8);

    return (can->sendFrame(msgPDO));
}

TPCANMsg *CanOpenBus::recvPDO()
{
    if (!can->readFrame(recvMsg))
        return NULL;
    else
        return &recvMsg;
}

bool CanOpenBus::recvPDO(unsigned char id, unsigned char pdonum, unsigned char *data)
{
    unsigned short rightPDOID = pdonum * 0x100 + 0x80 + id;

    recvMsg.ID = 0;
    if (!can->readFrame(recvMsg))
        return false;
    if (recvMsg.ID != rightPDOID)
        return false;

    memcpy(data, recvMsg.DATA, sizeof(unsigned char) * recvMsg.LEN);

    return true;
}

bool CanOpenBus::recvEmergencyObject(unsigned char id, unsigned char *data)
{
    unsigned short rightEOID = 0x80 + id;

    recvMsg.ID = 0;
    if (!can->readFrame(recvMsg))
        return false;
    if (recvMsg.ID != rightEOID)
        return false;

    if (data != NULL)
        memcpy(data, recvMsg.DATA, sizeof(unsigned char) * recvMsg.LEN);

    return true;
}

void CanOpenBus::initMsg()
{
    msgNMT.ID = 0;
    msgNMT.LEN = 2;
    msgNMT.MSGTYPE = MSGTYPE_STANDARD;
    msgNMT.DATA[0] = 1;
    msgNMT.DATA[1] = 1;
    msgNMT.DATA[2] = 0x0;
    msgNMT.DATA[3] = 0x0;
    msgNMT.DATA[4] = 0x0;
    msgNMT.DATA[5] = 0x0;
    msgNMT.DATA[6] = 0x0;
    msgNMT.DATA[7] = 0x0;

    msgRTR.ID = 0;
    msgRTR.LEN = 0;
    msgRTR.MSGTYPE = MSGTYPE_RTR;
    msgRTR.DATA[0] = 0x0;
    msgRTR.DATA[1] = 0x0;
    msgRTR.DATA[2] = 0x0;
    msgRTR.DATA[3] = 0x0;
    msgRTR.DATA[4] = 0x0;
    msgRTR.DATA[5] = 0x0;
    msgRTR.DATA[6] = 0x0;
    msgRTR.DATA[7] = 0x0;

    msgSYNC.ID = 0x80;
    msgSYNC.LEN = 0;
    msgSYNC.MSGTYPE = MSGTYPE_STANDARD;
    msgSYNC.DATA[0] = 0x0;
    msgSYNC.DATA[1] = 0x0;
    msgSYNC.DATA[2] = 0x0;
    msgSYNC.DATA[3] = 0x0;
    msgSYNC.DATA[4] = 0x0;
    msgSYNC.DATA[5] = 0x0;
    msgSYNC.DATA[6] = 0x0;
    msgSYNC.DATA[7] = 0x0;

    msgSDO.ID = 0x601;
    msgSDO.LEN = 8;
    msgSDO.MSGTYPE = MSGTYPE_STANDARD;
    msgSDO.DATA[0] = 0x0;
    msgSDO.DATA[1] = 0x0;
    msgSDO.DATA[2] = 0x0;
    msgSDO.DATA[3] = 0x0;
    msgSDO.DATA[4] = 0x0;
    msgSDO.DATA[5] = 0x0;
    msgSDO.DATA[6] = 0x0;
    msgSDO.DATA[7] = 0x0;

    msgPDO.ID = 0x181;
    msgPDO.LEN = 8;
    msgPDO.MSGTYPE = MSGTYPE_STANDARD;
    msgPDO.DATA[0] = 0x0;
    msgPDO.DATA[1] = 0x0;
    msgPDO.DATA[2] = 0x0;
    msgPDO.DATA[3] = 0x0;
    msgPDO.DATA[4] = 0x0;
    msgPDO.DATA[5] = 0x0;
    msgPDO.DATA[6] = 0x0;
    msgPDO.DATA[7] = 0x0;
}

TPCANMsg *CanOpenBus::sendSDO(unsigned char id, bool write, unsigned short index, unsigned char subindex, unsigned char length, unsigned char *data)
{
    if (length > 4)
        return NULL;
    if ((id < 1) || (id > 0x7f))
        return NULL;

    msgSDO.ID = 0x600 + id;
    msgSDO.LEN = 4 + length;
    if (write == true)
        msgSDO.DATA[0] = 0x23 + ((4 - length) << 2);
    else
        msgSDO.DATA[0] = 0x40 + ((4 - length) << 2);
    msgSDO.DATA[1] = (index & 0xff);
    msgSDO.DATA[2] = (index >> 8);
    msgSDO.DATA[3] = subindex;
    for (int i = 0; i < (length); ++i)
    {
        msgSDO.DATA[i + 4] = data[i];
    }

    if (!can->sendFrame(msgSDO))
    {
        return NULL;
    }

    unsigned short rightRecvID = 0x580 + id;
    recvMsg.ID = 0;
    while (recvMsg.ID != rightRecvID)
    {
        can->readFrame(recvMsg);
    }

    return &recvMsg;
}

#endif
