/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <iostream>
#include <net/covise_host.h>
#include <net/covise_connect.h>
#include <net/covise_socket.h>
#include <string.h>

#ifndef WIN32
#include <unistd.h>
#endif
using namespace std;
using namespace covise;

unsigned char commandBuffer[100];


unsigned char generateChecksum(unsigned char *buf,int len)
{
    int sum = 0;
    for (int i = 1; i < len; i++) // skip first byte
    {
        sum += buf[i];
    }
    return (unsigned int) (sum % 256); 
}
int genCommand(unsigned char c1, unsigned char c2, unsigned char c3)
{
            commandBuffer[0] = 0xFE;
            commandBuffer[1] = 0x00; // device address
            commandBuffer[2] = 0x00; // answer prefix (1)
            commandBuffer[3] = 0x03; // answer prefix (2)
            commandBuffer[4] = 0x02; // answer prefix data (send result)
            commandBuffer[5] = c1; // command byte (1)
            commandBuffer[6] = c2; // command byte (2)
            commandBuffer[7] = c3; // data byte (0x01 = switch lamps on)
            commandBuffer[8] = generateChecksum(commandBuffer,8);
            commandBuffer[9] = 0xFF;

            return 10;
}
int main(int argc, char **argv)
{
    // check arguments
    if (argc != 3)
    {
        cerr << "Wrong argument number" << endl;
        cerr << "Usage: as3dctrl projector command " << endl;
        return 1;
    }
    Host *pro = new Host(argv[1]);

    SimpleClientConnection *projector = new SimpleClientConnection(pro, 9999);
    if (projector->is_connected())
    {
            int commandLength=0;
            if(strcmp(argv[1],"on") == 0)
            {
                commandLength = genCommand(0x76,0x1A,0x01);
            }
            else if(strcmp(argv[1],"off") == 0)
            {
                commandLength = genCommand(0x76,0x1A,0x00);
            }
            if (commandLength > 0)
            {
                int numWritten = projector->getSocket()->write(commandBuffer, commandLength);
                if (numWritten < commandLength)
                {
                    cerr << "could not send all bytes, only" << numWritten << " of " << commandLength << endl;
                }
                else
                {
                    unsigned char replyBuffer[1000];
                    projector->getSocket()->setNonBlocking(true);
                    time_t startZeit = time(NULL);
                    do
                    {
                        int numRead;
                        do
                        {
                            errno = 0;
#ifdef _WIN32
                            numRead = ::recv(projector->getSocket()->get_id(), (char *)replyBuffer, 1000, 0);
                        } while (((projector->getSocket()->getErrno() == WSAEINPROGRESS) || (projector->getSocket()->getErrno() == WSAEINTR)));
#else
                            numRead = ::read(projector->getSocket()->get_id(), replyBuffer, 1000);
                        } while ((errno == EAGAIN || errno == EINTR));
#endif

                        if (numRead > 0)
                        {
                            if(numRead == 5)
                            {
                                if(replyBuffer[0] == 0xFE)
                                    return replyBuffer[4];
                                else
                                return -1;
                            }
                            else
                            {
                                return -1;
                            }
                            break;
                        }
                    } while (time(NULL) < startZeit + 1);
                }
            }
    }
    else
    {
        cerr << "could not connect to projector " << argv[1] << " on port 0xAAA0" << endl;
    }
}
