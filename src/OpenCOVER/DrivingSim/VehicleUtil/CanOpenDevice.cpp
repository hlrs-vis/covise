/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CanOpenDevice.h"

#include <cstring>
#include <native/timer.h>

CanOpenDevice::CanOpenDevice(CanOpenController &setCon, uint8_t setId)
    : controller(&setCon)
    , nodeid(setId)
{
    if ((nodeid < 1) || (nodeid > 0x7f))
    {
        std::cerr << "CanOpenDevice::CanOpenDevice(): Warning: CanOpen identifier " << std::hex << nodeid << " is not in valid range (0x1 - 0x7f)" << std::endl;
        std::cerr << " -> undefined behaviour of device expected!" << std::endl;
    }
}

void CanOpenDevice::initCANOpenDevice()
{
    startNode();
}
void CanOpenDevice::shutdownCANOpenDevice()
{
    stopNode();
}

void CanOpenDevice::startNode()
{
    controller->startNode(nodeid);
}

void CanOpenDevice::stopNode()
{
    controller->stopNode(nodeid);
}

void CanOpenDevice::enterPreOp()
{
    controller->enterPreOp(nodeid);
}

void CanOpenDevice::resetNode()
{
    controller->resetNode(nodeid);
}

void CanOpenDevice::resetComm()
{
    controller->resetComm(nodeid);
}

/*void CanOpenDevice::sendRTR(uint8_t cob, uint8_t length)
{
   can_frame rtrFrame;
   rtrFrame.can_id = cob;
   rtrFrame.can_dlc = length;

   can->sendFrame(rtrFrame);
}*/

bool CanOpenDevice::readSDO(uint16_t index, uint8_t subindex, uint8_t *data)
{
    return controller->readSDO(nodeid, index, subindex, data);
}

bool CanOpenDevice::writeSDO(uint16_t index, uint8_t subindex, uint8_t *data, uint8_t length)
{
    return controller->writeSDO(nodeid, index, subindex, data, length);
}

uint8_t *CanOpenDevice::readTPDO(uint8_t pdo)
{
    return controller->readTPDO(nodeid, pdo);
}

void CanOpenDevice::writeRPDO(uint8_t pdo, uint8_t *data, uint8_t numData)
{
    return controller->writeRPDO(nodeid, pdo, data, numData);
}
