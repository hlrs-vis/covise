/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CanOpenDevice_h
#define __CanOpenDevice_h

#include "CanOpenController.h"

#include <vector>

class VEHICLEUTILEXPORT CanOpenDevice
{
public:
    CanOpenDevice(CanOpenController &, uint8_t);
    virtual ~CanOpenDevice(){};

    virtual void initCANOpenDevice();
    virtual void shutdownCANOpenDevice();

    //NMT service
    void startNode();
    void stopNode();
    void enterPreOp();
    void resetNode();
    void resetComm();

    //RTR service
    //void sendRTR(uint8_t cob, uint8_t length);

    //SDO service
    bool readSDO(uint16_t index, uint8_t subindex, uint8_t *data);
    bool writeSDO(uint16_t index, uint8_t subindex, uint8_t *data, uint8_t length = 4);

    //PDO service
    uint8_t *readTPDO(uint8_t pdo);
    void writeRPDO(uint8_t pdo, uint8_t *data, uint8_t numData = 8);

protected:
    CanOpenController *controller;
    uint8_t nodeid;
};

#endif
