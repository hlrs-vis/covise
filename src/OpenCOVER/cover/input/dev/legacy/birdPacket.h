/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__BIRDPACKET_H)
#define __BIRDPACKET_H

class INPUT_LEGACY_EXPORT birdPacket
{
protected:
    // header
    unsigned short int sequence;
    unsigned short int milliseconds;
    unsigned int time;
    unsigned char type;
    unsigned char xtype;
    unsigned char protocol;
    unsigned char error_code;
    unsigned short int error_code_extension;
    unsigned short int number_bytes;
    char data[2048];

    // allocate mem for the string and copy s into it, then return a pointer
    char *newString(char *s);

public:
    enum command
    {
        MSG_WAKE_UP = 10,
        MSG_SHUT_DOWN = 11,
        MSG_GET_STATUS = 101,
        MSG_SEND_SETUP = 102,
        MSG_SINGLE_SHOT = 103,
        MSG_RUN_CONTINUOUS = 104,
        MSG_STOP_DATA = 105,
        MSG_SEND_DATA = 106,
        MSG_SYNC_SEQUENCE = 30,
        RSP_WAKE_UP = 20,
        RSP_SHUT_DOWN = 21,
        RSP_GET_STATUS = 201,
        RSP_SEND_SETUP = 202,
        RSP_RUN_CONTINUOUS = 204,
        RSP_STOP_DATA = 205,
        RSP_ILLEGAL = 40,
        RSP_UNKNOWN = 50,
        RSP_SYNC_SEQUENCE = 31,
        DATA_PACKET_MULTI = 210,
        DATA_PACKET_ACK = 211,
        DATA_PACKET_SINGLE = 212,
        MSG_SLEEP = 71
    };

    birdPacket();
    ~birdPacket();

    // set the packet-type and destination fbb-address
    void setType(command cmd, int fbb = 0);
    char *getType();

    // packet (header+data-field)
    unsigned getSize();
    void *getPtr();

    // data-field
    void setDataSize(unsigned num);
    unsigned getDataSize();
    void *getData();

    // header
    void setSequence(unsigned short int seq);
    unsigned short int getSequence();

    void setTimeStamp(unsigned int seconds, unsigned short int millisec);
    void getTimeStamp(unsigned int *seconds, unsigned short int *millisec);

    // debugging
    void dump();
    void dumpByte(unsigned char b);
};
#endif
