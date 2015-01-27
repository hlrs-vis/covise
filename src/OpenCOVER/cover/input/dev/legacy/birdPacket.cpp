/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/common.h>

#include "birdPacket.h"

void swap_int(unsigned int *d, int num)
{
    unsigned int *data = (unsigned int *)d;
    int i;

    for (i = 0; i < num; i++)
    {

        *data = (((*data) & 0xff000000) >> 24)
                | (((*data) & 0x00ff0000) >> 8)
                | (((*data) & 0x0000ff00) << 8)
                | (((*data) & 0x000000ff) << 24);

        data++;
    }
}

void swap_shortint(unsigned short int *d, int num)
{
    unsigned short int *data = (unsigned short int *)d;
    int i;

    for (i = 0; i < num; i++)
    {

        *data = (((*data) & 0xff00) >> 8)
                | (((*data) & 0x00ff) << 8);

        data++;
    }
}

birdPacket::birdPacket()
{
    // we expect protocol level 3
    protocol = 3;

    // done
    return;
}

birdPacket::~birdPacket()
{
    // dummy
}

void birdPacket::setType(command cmd, int fbb)
{
    // set type / xtype of packet
    type = (unsigned char)cmd;
    xtype = (unsigned char)fbb;

    // we report no errors
    error_code = 0;
    error_code_extension = 0;

    return;
}

char *birdPacket::getType()
{
    char *r = NULL;
    //command cmd;

    // we want the enum again
    //cmd = (command)type;

    switch (type)
    {
    case birdPacket::MSG_WAKE_UP:
        r = newString((char *)"MSG_WAKE_UP");
        break;
    case birdPacket::MSG_SHUT_DOWN:
        r = newString((char *)"MSG_SHUT_DOWN");
        break;
    case birdPacket::MSG_GET_STATUS:
        r = newString((char *)"MSG_GET_STATUS");
        break;
    case birdPacket::MSG_SEND_SETUP:
        r = newString((char *)"MSG_SEND_SETUP");
        break;
    case birdPacket::MSG_SINGLE_SHOT:
        r = newString((char *)"MSG_SINGLE_SHOT");
        break;
    case birdPacket::MSG_RUN_CONTINUOUS:
        r = newString((char *)"MSG_RUN_CONTINUOUS");
        break;
    case birdPacket::MSG_STOP_DATA:
        r = newString((char *)"MSG_STOP_DATA");
        break;
    case birdPacket::MSG_SEND_DATA:
        r = newString((char *)"MSG_SEND_DATA");
        break;
    case birdPacket::MSG_SYNC_SEQUENCE:
        r = newString((char *)"MSG_SYNC_SEQUENCE");
        break;
    case birdPacket::RSP_WAKE_UP:
        r = newString((char *)"RSP_WAKE_UP");
        break;
    case birdPacket::RSP_SHUT_DOWN:
        r = newString((char *)"RSP_SHUT_DOWN");
        break;
    case birdPacket::RSP_GET_STATUS:
        r = newString((char *)"RSP_GET_STATUS");
        break;
    case birdPacket::RSP_SEND_SETUP:
        r = newString((char *)"RSP_SEND_SETUP");
        break;
    case birdPacket::RSP_RUN_CONTINUOUS:
        r = newString((char *)"RSP_RUN_CONTINUOUS");
        break;
    case birdPacket::RSP_STOP_DATA:
        r = newString((char *)"RSP_STOP_DATA");
        break;
    case birdPacket::RSP_ILLEGAL:
        r = newString((char *)"RSP_ILLEGAL");
        break;
    case birdPacket::RSP_UNKNOWN:
        r = newString((char *)"RSP_UNKNOWN");
        break;
    case birdPacket::RSP_SYNC_SEQUENCE:
        r = newString((char *)"RSP_SYNC_SEQUENCE");
        break;
    case birdPacket::DATA_PACKET_MULTI:
        r = newString((char *)"DATA_PACKET_MULTI");
        break;
    case birdPacket::DATA_PACKET_ACK:
        r = newString((char *)"DATA_PACKET_ACK");
        break;
    case birdPacket::DATA_PACKET_SINGLE:
        r = newString((char *)"DATA_PACKET_SINGLE");
        break;
    default:
        r = newString((char *)"***UNDEFINED***");
    }

    // return a pointer to the packet-type-string, remember to
    // delete[] it later !!
    return (r);
}

char *birdPacket::newString(char *s)
{
    char *t = NULL;

    int i;
    for (i = 0; s[i]; i++)
        ;

    t = new char[i + 1];

    for (i = 0; s[i]; i++)
        t[i] = s[i];
    t[i] = '\0';

    return t;
}

void birdPacket::setDataSize(unsigned num)
{
// set the size in bytes of the following data-field
#ifdef BYTESWAP
    unsigned short int size = (unsigned short int)num;
    swap_shortint(&size, 1);
    number_bytes = size;
#else
    number_bytes = (unsigned char)num;
#endif
    return;
}

void *birdPacket::getPtr()
{
    // return a pointer to the beginning of the packet (the header)
    return ((void *)&sequence);
}

unsigned birdPacket::getSize()
{
// return total number of bytes in this packet
#ifdef BYTESWAP
    unsigned short int size = number_bytes;
    swap_shortint(&size, 1);
    return (size + 16);
#else
    return (number_bytes + 16);
#endif
}

unsigned birdPacket::getDataSize()
{
// return the size in bytes of the following data-field
#ifdef BYTESWAP
    unsigned short int size = number_bytes;
    swap_shortint(&size, 1);
    return (size);
#else
    return (number_bytes);
#endif
}

void *birdPacket::getData()
{
    // get a pointer to the data-field of the packet
    return ((void *)data);
}

void birdPacket::setSequence(unsigned short int seq)
{
// set the sequence of this packet
#ifdef BYTESWAP
    unsigned short int tmpi = seq;
    swap_shortint(&tmpi, 1);
    sequence = tmpi;
#else
    sequence = seq;
#endif

    return;
}

unsigned short int birdPacket::getSequence()
{
// get the sequence of this packet
#ifdef BYTESWAP
    unsigned short int tmpi = sequence;
    swap_shortint(&tmpi, 1);
    return (tmpi);
#else
    return (sequence);
#endif
}

void birdPacket::setTimeStamp(unsigned int seconds, unsigned short int millisec)
{
// set the timestamp of current packet
#ifdef BYTESWAP
    unsigned int tmpi = seconds;
    swap_int(&tmpi, 1);
    time = tmpi;
#else
    time = seconds;
#endif
#ifdef BYTESWAP
    unsigned short int tmpsi = millisec;
    swap_shortint(&tmpsi, 1);
    milliseconds = tmpsi;
#else
    milliseconds = millisec;
#endif

    return;
}

void birdPacket::getTimeStamp(unsigned int *seconds, unsigned short int *millisec)
{
// get the timestamp

#ifdef BYTESWAP
    unsigned int tmpi = time;
    swap_int(&tmpi, 1);
    *seconds = tmpi;
#else
    *seconds = time;
#endif
#ifdef BYTESWAP
    unsigned short int tmpsi = milliseconds;
    swap_shortint(&tmpsi, 1);
    *millisec = tmpsi;
#else
    *millisec = milliseconds;
#endif

    return;
}

void birdPacket::dump()
{
    //unsigned char *ptr;
    //char *t;
    unsigned int i;

    fprintf(stderr, "\n\n====== birdPacket::dump() ======\n");

    // dump header first
    //ptr = (unsigned char *)getPtr();
    unsigned int sec;
    unsigned short int milli;
    getTimeStamp(&sec, &milli);
    fprintf(stderr, "sequence: %d\n", getSequence());
    fprintf(stderr, "millisec: %d\n", milli);
    fprintf(stderr, "time    : %d\n", sec);
    fprintf(stderr, "type    : %s\n", getType());
    fprintf(stderr, "ext.type: %d\n", xtype);
    fprintf(stderr, "protocol: %d\n", protocol);
    fprintf(stderr, "error   : %d\n", error_code);
    fprintf(stderr, "errorext: %d\n", error_code_extension);
    fprintf(stderr, "numbytes: %d\n", getDataSize());

    // separator
    fprintf(stderr, "--------------------------------\n");

    // switch( type )
    // {
    //    default:
    for (i = 0; i < getDataSize(); i++)
    {
        fprintf(stderr, "%2d:   %02x   ", i, data[i]);
        dumpByte(data[i]);
        fprintf(stderr, "   %d\n", data[i]);
    }

    //       break;
    // }

    // done
    fprintf(stderr, "================================\n");

    return;
}

void birdPacket::dumpByte(unsigned char b)
{
    int i;
    int f;

    for (i = 7; i > -1; i--)
    {
        if (b & (1 << i))
            f = 1;
        else
            f = 0;
        fprintf(stderr, "%d", f);
    }

    return;
}
