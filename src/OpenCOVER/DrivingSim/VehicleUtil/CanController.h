/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CanController_h
#define __CanController_h

#include "XenomaiSocketCan.h"

#include <map>

class CanController : public XenomaiSocketCan
{
public:
    CanController(const std::string &device)
        : XenomaiSocketCan(device)
    {
    }

    //PDO buffer service
    const can_frame &readPDO(can_id_t);
    void writePDO(const can_frame &);

    //PDO comm service
    void recvPDO(uint8_t);
    bool handleFrame(can_frame &frame);
    void sendPDO();

protected:
    std::map<can_id_t, can_frame> TPDOMap;
    std::map<can_id_t, can_frame> RPDOMap;
};

inline const can_frame &CanController::readPDO(can_id_t id)
{
    return TPDOMap[id];
}

inline void CanController::writePDO(const can_frame &pdoFrame)
{
    RPDOMap[pdoFrame.can_id] = pdoFrame;
}

inline void CanController::recvPDO(uint8_t num)
{
    can_frame frame;
    for (int recvIt = 0; recvIt < num; ++recvIt)
    {
        int counter = 0;
        do
        {
            recvFrame(frame);
            //printFrame("recvPDO:",frame);
            counter++;
            if (counter > 10)
            {
                fprintf(stderr, "no valid PDO frame in time");
                return;
            }
        } while (!(frame.can_id & 0x700)); // skip sync messages and others
        TPDOMap[frame.can_id] = frame;
        //std::cerr << "recv: " << frame << std::endl;
    }
}

inline bool CanController::handleFrame(can_frame &frame)
{
    //std::map<can_id_t, can_frame>::iterator iter = TPDOMap.begin();

    TPDOMap[frame.can_id] = frame;
    /*if(frame.can_id == 0x18b) {
      std::cerr << "CanController::handleFrame(): " << frame << std::endl;
   }*/

    //    // Check if frame is in map
    /*iter = TPDOMap.find(frame.can_id);
 
      if(iter != TPDOMap.end())
      {
          //TPDOMap[frame.can_id] = frame;
          if (frame.can_id == 0x195) std::cout << "Received 195: " << frame << std::endl;
      }
//    else
//        return false;*/
    return true;
}

inline void CanController::sendPDO()
{
    for (std::map<can_id_t, can_frame>::iterator mapIt = RPDOMap.begin(); mapIt != RPDOMap.end(); ++mapIt)
    {
        sendFrame(mapIt->second);
        //std::cerr << "send: " << mapIt->second << std::endl;
    }
}

#endif
