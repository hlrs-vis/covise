/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <iostream>
#include <net/covise_host.h>
#include <net/covise_connect.h>
#include <net/covise_socket.h>
using namespace std;
using namespace covise;

enum
{
    END = 0,
    DVI = 4,
    VGA = 5,
    SYNCON = 7,
    SYNCOFF = 8,
    POWERON = 9,
    POWEROFF = 10,
    GAMMA = 11,
    UP = 12,
    DOWN = 13,
    RIGHT = 14,
    LEFT = 15,
    OK = 16,
    MENU = 17,
    STEREO = 32,
    SELECTRIGHT = 33,
    SELECTLEFT = 34,
    FORWARD = 30,
    PROJLEFT = 25,
    PROJRIGHT = 27,
    PROJBOTH = 26,
    FILENAME = 127
};

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

    SimpleClientConnection *projector = new SimpleClientConnection(pro, 1025);
    if (projector->is_connected())
    {
        char filename[1000];
        sprintf(filename, "eOps/%s", argv[2]);
        int file = open(filename, O_RDONLY);
        char commandBuffer[1000];
        if (file > 0)
        {
            int commandLength = read(file, commandBuffer, 1000);
            if (commandLength > 0)
            {
                int numWritten = projector->getSocket()->write(commandBuffer, commandLength);
                if (numWritten < commandLength)
                {
                    cerr << "could not send all bytes, only" << numWritten << " of " << commandLength << endl;
                }
            }
            close(file);
        }
        else
        {
            //cerr << "could not open file eOps/" <<  argv[2]  << endl;
            //assume ascii command
            sprintf(commandBuffer, ": %s\r", argv[2]);
            int commandLength = strlen(commandBuffer);
            if (commandLength > 0)
            {
                int numWritten = projector->getSocket()->write(commandBuffer, commandLength);
                if (numWritten < commandLength)
                {
                    cerr << "could not send all bytes, only" << numWritten << " of " << commandLength << endl;
                }
                else
                {
                    char replyBuffer[1000];
                    projector->getSocket()->setNonBlocking(true);
                    time_t startZeit = time(NULL);
                    do
                    {
                        int numRead;
                        do
                        {
                            errno = 0;
#ifdef _WIN32
                            numRead = ::recv(projector->getSocket()->get_id(), replyBuffer, 1000, 0);
                        } while (((projector->getSocket()->getErrno() == WSAEINPROGRESS) || (projector->getSocket()->getErrno() == WSAEINTR)));
#else
                            numRead = ::read(projector->getSocket()->get_id(), replyBuffer, 1000);
                        } while ((errno == EAGAIN || errno == EINTR));
#endif

                        if (numRead > 0)
                        {
                            cerr << replyBuffer;
                            break;
                        }
                    } while (time(NULL) < startZeit + 1);
                }
            }
        }
    }
    else
    {
        cerr << "could not connect to projector " << argv[1] << " on port 1025" << endl;
    }
}
