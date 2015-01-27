/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CanUsb.h"

CanUsb::CanUsb(char *serial, int nSpeed)
{
    std::cout << "Setting VID(0x403) and PID(0xffa8)... ";
    if (FT_SetVIDPID(0x403, 0xffa8) != FT_OK)
        std::cout << "failed!" << std::endl;
    else
        std::cout << "done!" << std::endl;

    std::cout << "Connecting to CANUSB... ";
    FT_STATUS ftStatus = FT_OpenEx(serial, FT_OPEN_BY_SERIAL_NUMBER, &ftHandle);
    if (ftStatus != FT_OK)
    {
        /* 
      	This can fail if the ftdi_sio driver is loaded
    	   use lsmod to check this and rmmod ftdi_sio to remove
			also rmmod usbserial
    		*/
        printf("FT_Open(%s) failed. rv=%d\n", serial, ftStatus);
    }
    std::cout << "done!" << std::endl;

    FT_SetTimeouts(ftHandle, 3000, 3000); // 3 second read timeout

    //---Open channel---
    char buf[80];
    unsigned long size;
    unsigned long retLen;

    // Set baudrate
    FT_Purge(ftHandle, FT_PURGE_RX);

    sprintf(buf, "S%d\r", nSpeed);
    size = 3;
    if (!(FT_OK == FT_Write(ftHandle, buf, size, &retLen)))
    {
        printf("Write failed\n");
    }

    // Open device
    FT_Purge(ftHandle, FT_PURGE_RX);
    strcpy(buf, "O\r");
    size = 2;
    if (!(FT_OK == FT_Write(ftHandle, buf, size, &retLen)))
    {
        printf("Write failed\n");
    }
}

CanUsb::~CanUsb()
{
    //---Close channel---
    char buf[80];
    unsigned long size;
    unsigned long retLen;

    // Close device
    FT_Purge(ftHandle, FT_PURGE_RX | FT_PURGE_TX);
    strcpy(buf, "C\r");
    size = 2;
    if (!(FT_OK == FT_Write(ftHandle, buf, size, &retLen)))
    {
    }

    std::cout << "Disconnecting!" << std::endl;
    FT_Close(ftHandle);
}

bool CanUsb::sendFrame(TPCANMsg &pmsg)
{
    CanMsg cmsg;
    TPCANMsgToCanMsg(pmsg, cmsg);
    return sendFrame(cmsg);
}

BOOL CanUsb::sendFrame(CanMsg &msg)
{
    int i;
    char txbuf[80];
    unsigned long size;
    unsigned long retLen;

    if (msg.flags & CANMSG_EXTENDED)
    {
        if (msg.flags & CANMSG_RTR)
        {
            sprintf(txbuf, "R%08.8lX%i", msg.id, msg.len);
            msg.len = 0;
        }
        else
        {
            sprintf(txbuf, "T%08.8lX%i", msg.id, msg.len);
        }
    }
    else
    {
        if (msg.flags & CANMSG_RTR)
        {
            sprintf(txbuf, "r%03.3lX%i", msg.id, msg.len);
            msg.len = 0; // Just dlc no data for RTR
        }
        else
        {
            sprintf(txbuf, "t%03.3lX%i", msg.id, msg.len);
        }
    }

    if (msg.len)
    {
        char hex[5];

        for (i = 0; i < msg.len; i++)
        {
            sprintf(hex, "%02.2X", msg.data[i]);
            strcat(txbuf, hex);
        }
    }

    // Add CR
    strcat(txbuf, "\r");

    size = strlen(txbuf);

    // Transmit fram
    if (!(FT_OK == FT_Write(ftHandle, txbuf, size, &retLen)))
    {
        return FALSE;
    }

    return TRUE;
}

bool CanUsb::readFrame(TPCANMsg &pmsg)
{
    CanMsg cmsg;
    if (readFrame(cmsg))
    {
        CanMsgToTPCANMsg(cmsg, pmsg);
        return true;
    }
    else
        return false;
}

BOOL CanUsb::readFrame(CanMsg &msg)
{

    char gbufferRx[BUF_SIZE];
    bool gnReceivedFrames = FALSE;
    int i;
    unsigned long nRxCnt;
    unsigned long nTxCnt;
    unsigned long eventStatus;
    unsigned long nRcvCnt;
    char c;

    static char msgReceiveBuf[80];
    static int cntMsgRcv = 0;
    static int state = CANUSB_STATE_NONE;

    // Check if there is something to receive
    if (FT_OK == FT_GetStatus(ftHandle, &nRxCnt, &nTxCnt, &eventStatus))
    {

        // If there are characters to receive
        if (nRxCnt)
        {

            // Must fit to buffer
            if (nRxCnt > sizeof(gbufferRx))
            {
                nRxCnt = sizeof(gbufferRx);
            }

            // Read data
            if ((FT_OK == FT_Read(ftHandle, gbufferRx, nRxCnt, &nRcvCnt)) && (nRcvCnt > 0))
            {
                for (i = 0; i < nRcvCnt; i++)
                {

                    // Get character
                    c = gbufferRx[i];

                    if (CANUSB_STATE_NONE == state)
                    {

                        if (('t' == c) || ('T' == c) || ('r' == c) || ('R' == c))
                        {
                            state = CANUSB_STATE_MSG;
                            memset(msgReceiveBuf, 0, sizeof(msgReceiveBuf));
                            msgReceiveBuf[0] = c;
                            cntMsgRcv = 1;
                        }
                    }

                    else if (CANUSB_STATE_MSG == state)
                    {

                        msgReceiveBuf[cntMsgRcv++] = c;

                        if (0x0d == c)
                        {

                            //printf("Raw Msg = %s\n", msgReceiveBuf );
                            if (!canusbToCanMsg(msgReceiveBuf, &msg))
                            {
                                printf("Message conversion failed!\n");
                                state = CANUSB_STATE_NONE;
                                return FALSE;
                            }

                            gnReceivedFrames = TRUE;

                            state = CANUSB_STATE_NONE;

                        } // full message

                    } // STATE_MSG

                } // for each char

            } // Read data

        } // characters to receive

    } // getstatus

    return gnReceivedFrames;
}

void CanUsb::printMsg(CanMsg &msg)
{
    if (msg.flags & CANMSG_EXTENDED)
    {
        std::cout << "extended ";
    }
    else
    {
        std::cout << "standard ";
    }

    std::cout << "message: id=" << std::hex << msg.id << " len=" << (int)msg.len << " timestamp=" << std::hex << msg.timestamp;

    if (msg.len)
    {
        std::cout << " data=";

        for (int j = 0; j < msg.len; j++)
        {
            std::cout << std::hex << (int)msg.data[j] << " ";
        }
    }
    std::cout << std::endl;
}

BOOL CanUsb::canusbToCanMsg(char *p, CanMsg *pMsg)
{
    int val;
    int i;
    short data_offset; // Offset to dlc byte
    char save;

    if ('t' == *p)
    {
        // Standard frame
        pMsg->flags = 0;
        data_offset = 5;
        pMsg->len = p[4] - '0';
        p[4] = 0;
        sscanf(p + 1, "%lx", &pMsg->id);
    }
    else if ('r' == *p)
    {
        // Standard remote  frame
        pMsg->len = p[4] - '0';
        pMsg->flags = CANMSG_RTR;
        //data_offset = 5 - 1;// To make timestamp work
        data_offset = 5;
        //save = p[ 4 ];
        p[4] = 0;
        sscanf(p + 1, "%lx", &pMsg->id);
        //p[ 4 ] = save;
    }
    else if ('T' == *p)
    {
        // Extended frame
        pMsg->flags = CANMSG_EXTENDED;
        data_offset = 10;
        pMsg->len = p[9] - '0';
        p[9] = 0;
        sscanf(p + 1, "%lx", &pMsg->id);
    }
    else if ('R' == *p)
    {
        // Extended remote frame
        pMsg->flags = CANMSG_EXTENDED | CANMSG_RTR;
        //data_offset = 10 - 1;// To make timestamp work
        data_offset = 10;
        pMsg->len = p[9] - '0';
        //save = p[ 9 ];
        p[9] = 0;
        sscanf(p + 1, "%lx", &pMsg->id);
        //p[ 9 ] = save;
    }

    save = *(p + data_offset + 2 * pMsg->len);

    // Fill in data
    if (!(pMsg->flags & CANMSG_RTR))
    {
        for (i = MIN(pMsg->len, 8); i > 0; i--)
        {
            *(p + data_offset + 2 * (i - 1) + 2) = 0;
            sscanf(p + data_offset + 2 * (i - 1), "%x", &val);
            pMsg->data[i - 1] = val;
        }
    }

    *(p + data_offset + 2 * pMsg->len) = save;

    if (!(pMsg->flags & CANMSG_RTR))
    {
        // If timestamp is active - fetch it
        if (0x0d != *(p + data_offset + 2 * pMsg->len))
        {
            p[data_offset + 2 * (pMsg->len) + 4] = 0;
            sscanf((p + data_offset + 2 * (pMsg->len)), "%x", &val);
            pMsg->timestamp = val;
        }
        else
        {
            pMsg->timestamp = 0;
        }
    }
    else
    {
        if (0x0d != *(p + data_offset))
        {
            p[data_offset + 4] = 0;
            sscanf((p + data_offset), "%x", &val);
            pMsg->timestamp = val;
        }
        else
        {
            pMsg->timestamp = 0;
        }
    }
    return TRUE;
}

void CanUsb::TPCANMsgToCanMsg(TPCANMsg &pmsg, CanMsg &cmsg)
{
    cmsg.flags = 0;
    if (pmsg.MSGTYPE == MSGTYPE_RTR)
        cmsg.flags = CANMSG_RTR;
    if (pmsg.MSGTYPE == MSGTYPE_EXTENDED)
        cmsg.flags += CANMSG_EXTENDED;

    cmsg.len = pmsg.LEN;

    memcpy(cmsg.data, pmsg.DATA, 8 * sizeof(unsigned char));
}

void CanUsb::CanMsgToTPCANMsg(CanMsg &cmsg, TPCANMsg &pmsg)
{
    pmsg.ID = cmsg.id;
    if (cmsg.flags & CANMSG_RTR)
        pmsg.MSGTYPE = MSGTYPE_RTR;
    else if (cmsg.flags & CANMSG_EXTENDED)
        pmsg.MSGTYPE = MSGTYPE_EXTENDED;
    else
        pmsg.MSGTYPE = MSGTYPE_STANDARD;
    pmsg.LEN = cmsg.len;
    memcpy(pmsg.DATA, cmsg.data, 8 * sizeof(unsigned char));
}
