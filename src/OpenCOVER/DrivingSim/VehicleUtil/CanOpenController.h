/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CanOpenController_h
#define __CanOpenController_h

#include "CanController.h"

#include <cstring>

class CanOpenController : public CanController
{
public:
    CanOpenController(const std::string &);

    //NMT service
    void startNode(uint8_t = 0);
    void stopNode(uint8_t = 0);
    void enterPreOp(uint8_t = 0);
    void resetNode(uint8_t = 0);
    void resetComm(uint8_t = 0);

    //RTR service
    //void sendRTR(uint8_t cob, uint8_t length);

    //SDO service
    bool readSDO(uint8_t nodeid, uint16_t index, uint8_t subindex, uint8_t *data, uint8_t * = NULL);
    bool writeSDO(uint8_t nodeid, uint16_t index, uint8_t subindex, uint8_t *data, uint8_t length = 4);

    //PDO service
    uint8_t *readTPDO(uint8_t nodeid, uint8_t pdo);
    void writeRPDO(uint8_t nodeid, uint8_t pdo, uint8_t *data, uint8_t numData = 8);

    //SYNC service
    void sendSync();

protected:
    can_frame syncFrame;
};

inline uint8_t *CanOpenController::readTPDO(uint8_t nodeid, uint8_t pdo)
{
    //std::cerr << "CanOpenController::readTPDO: Nodeid :" << (int)nodeid << ", frame: " << TPDOMap[pdo*0x100 + 0x80 + nodeid] << std::endl;

    return TPDOMap[pdo * 0x100 + 0x80 + nodeid].data;
}

inline void CanOpenController::writeRPDO(uint8_t nodeid, uint8_t pdo, uint8_t *data, uint8_t numData)
{
    can_id_t canId = pdo * 0x100 + 0x100 + nodeid;
    can_frame *frame = &(RPDOMap[canId]);
    frame->can_id = canId;
    frame->can_dlc = numData;

    memcpy(frame->data, data, numData);
}

#endif
