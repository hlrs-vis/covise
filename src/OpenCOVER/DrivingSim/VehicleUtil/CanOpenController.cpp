/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CanOpenController.h"

CanOpenController::CanOpenController(const std::string &device)
    : CanController(device)
{
    syncFrame.can_id = 0x80;
    syncFrame.can_dlc = 0;
}

void CanOpenController::startNode(uint8_t nodeid)
{
    can_frame frame;

    frame.can_id = 0x0;
    frame.can_dlc = 2;
    frame.data[0] = 0x1;
    frame.data[1] = nodeid;

    sendFrame(frame);
    std::cerr << "startNode " << (int)nodeid << std::endl;
}

void CanOpenController::stopNode(uint8_t nodeid)
{
    can_frame frame;

    frame.can_id = 0x0;
    frame.can_dlc = 2;
    frame.data[0] = 0x2;
    frame.data[1] = nodeid;

    sendFrame(frame);
    std::cerr << "stopNode " << (int)nodeid << std::endl;
}

void CanOpenController::enterPreOp(uint8_t nodeid)
{
    can_frame frame;

    frame.can_id = 0x0;
    frame.can_dlc = 2;
    frame.data[0] = 0x80;
    frame.data[1] = nodeid;

    sendFrame(frame);
    std::cerr << "enterPreOp " << (int)nodeid << std::endl;
}

void CanOpenController::resetNode(uint8_t nodeid)
{
    can_frame frame;

    frame.can_id = 0x0;
    frame.can_dlc = 2;
    frame.data[0] = 0x81;
    frame.data[1] = nodeid;

    sendFrame(frame);
}

void CanOpenController::resetComm(uint8_t nodeid)
{
    can_frame frame;

    frame.can_id = 0x0;
    frame.can_dlc = 2;
    frame.data[0] = 0x82;
    frame.data[1] = nodeid;

    sendFrame(frame);
    std::cerr << "resetComm " << (int)nodeid << std::endl;
}

void CanOpenController::sendSync()
{
    sendFrame(syncFrame);
}

/*void CanOpenController::sendRTR(uint8_t cob, uint8_t length)
{
   can_frame rtrFrame;
   rtrFrame.can_id = cob;
   rtrFrame.can_dlc = length;

   sendFrame(rtrFrame);
}*/

bool CanOpenController::readSDO(uint8_t nodeid, uint16_t index, uint8_t subindex, uint8_t *data, uint8_t *length)
{
    can_frame frame;
    frame.can_id = 0x600 + nodeid;
    frame.can_dlc = 8;
    frame.data[0] = 0x40;
    memcpy(frame.data + 1, &index, sizeof(index));
    frame.data[3] = subindex;
    frame.data[4] = 0;
    frame.data[5] = 0;
    frame.data[6] = 0;
    frame.data[7] = 0;

    if (!sendFrame(frame))
    {
        return false;
    }

    //unsigned int numReadTrials = 0;
    setRecvTimeout(1000000000);
    /*while(frame.can_id != (0x580 + nodeid)) {
      if(!(numReadTrials < 5)) {
         return false;
      }
   */
    int numTries = 0;
    do
    {
        if (!recvFrame(frame))
        {
            return false;
        }
        if (frame.can_id & CAN_ERR_FLAG)
        {
            std::cerr << "CanOpenController::readSDO received error frame trying to read the message again "
                      << numTries << std::endl;
        }
        numTries++;
    } while (frame.can_id & CAN_ERR_FLAG && (numTries < 100));

    /*
      ++numReadTrials;
   }*/
    setRecvTimeout(RTDM_TIMEOUT_INFINITE);

    if ((frame.data[0] >> 5) == 2)
    {
        memcpy(data, frame.data, 4 * sizeof(uint8_t));

        if (length)
        {
            *length = 4 - ((frame.data[0] - 0x43) / 0x4);
        }

        return true;
    }

    printFrame("sdo read failed: ", frame);
    return false;
}

bool CanOpenController::writeSDO(uint8_t nodeid, uint16_t index, uint8_t subindex, uint8_t *data, uint8_t length)
{
    can_frame frame;
    frame.can_id = 0x600 + nodeid;
    frame.can_dlc = 8;
    frame.data[0] = 0x23 + 0x4 * (4 - length);
    memcpy(frame.data + 1, &index, sizeof(index));
    frame.data[3] = subindex;
    memcpy(frame.data + 4, data, length * sizeof(uint8_t));

    //std::cerr << "sdo write: " << frame << std::endl;

    if (!sendFrame(frame))
    {
        return false;
    }

    unsigned int numReadTrials = 0;
    setRecvTimeout(500000000);
    while (numReadTrials < 5)
    {

        if (!recvFrame(frame))
        {
            return false;
        }
        if (frame.can_id == (0x580 + nodeid))
            break;
        printFrame("received other message than expected, retrying TODO buffer this message for later use: ", frame);
        numReadTrials++;
    }
    setRecvTimeout(RTDM_TIMEOUT_INFINITE);

    if ((frame.data[0] == 0x60) && (frame.data[1] == (index & 0xff)) && (frame.data[2] == (index >> 8)))
    {
        return true;
    }
    printFrame("sdo write failed: ", frame);
    return false;
}
