/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ----------------------------------------------------------------------------
// A description of the Pulse API and a complete list of JSON-RPC 2.0 commands
// for the Barco F-Series can be found in:
//   Barco End User Reference Guid
//   RS232 and Network Command Catalog
//   For JSON RPC/Pulse Based Projectors For F70
// ----------------------------------------------------------------------------

#include <stdio.h>
#include <iostream>
#include <net/covise_host.h>
#include <net/covise_connect.h>
#include <net/covise_socket.h>
#include <string.h>
#include <algorithm>

#ifndef WIN32
#include <unistd.h>
#endif

using namespace std;
using namespace covise;

#define PROJECTOR_CMD_PORT 9090 

const char* CMD_ON = "{\"jsonrpc\": \"2.0\",\"method\": \"system.poweron\",\"params\": {\"property\": \"system.state\"  },\"id\": 3}\n";
const char* CMD_OFF = "{\"jsonrpc\": \"2.0\",\"method\": \"system.poweroff\",\"params\": {\"property\": \"system.state\"  },\"id\": 3}\n";
const char* CMD_LASER40 = "{\"jsonrpc\": \"2.0\",\"method\": \"property.set\",\"params\": {\"property\": \"illumination.sources.laser.power\",\"value\": 40  },\"id\": 5}\n";
const char* CMD_LASER90 = "{\"jsonrpc\": \"2.0\",\"method\": \"property.set\",\"params\": {\"property\": \"illumination.sources.laser.power\",\"value\": 90  },\"id\": 5}\n";
const char* CMD_LASER100 = "{\"jsonrpc\": \"2.0\",\"method\": \"property.set\",\"params\": {\"property\": \"illumination.sources.laser.power\",\"value\": 100  },\"id\": 5}\n";

unsigned char commandBuffer[100];

// ----------------------------------------------------------------------------
// main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
    
    unsigned char commandBuffer[256];
    size_t commandLength = 0;
    
    // check arguments
    if (argc != 3)
    {
        cerr << "f80ctrl - basic barco f-series projector control" << endl;
        cerr << "error: wrong number of arguments" << endl << endl;
        cerr << "Usage: f80ctrl [IP/ADDRESS OF PROJECTOR] [COMMAND]" << endl;
        cerr << "" << endl;
        cerr << "commands     json object" << endl;
        cerr << "  on          " << CMD_ON;
        cerr << "  off         " << CMD_OFF;
        cerr << "  laser40     " << CMD_LASER40;
        cerr << "  laser90     " << CMD_LASER90;
        cerr << "  laser100    " << CMD_LASER100;
        return 1;
    }
    
    Host *pro = new Host(argv[1]);
    SimpleClientConnection *projector = new SimpleClientConnection(pro, PROJECTOR_CMD_PORT);
    
    if (projector->is_connected())
    {
        // cout << "connected to " << argv[1] << endl;

        if(strcmp(argv[2],"on") == 0)
        {
            strcpy((char*) commandBuffer, "{\"jsonrpc\": \"2.0\",\"method\": \"system.poweron\",\"params\": {\"property\": \"system.state\"  },\"id\": 3}\n");
        }
        else if(strcmp(argv[2],"off") == 0)
        {
            strcpy((char*) commandBuffer, "{\"jsonrpc\": \"2.0\",\"method\": \"system.poweroff\",\"params\": {\"property\": \"system.state\"  },\"id\": 3}\n");
        }
        else if(strcmp(argv[2],"laser40") == 0)
        {
            strcpy((char*) commandBuffer, "{\"jsonrpc\": \"2.0\",\"method\": \"property.set\",\"params\": {\"property\": \"illumination.sources.laser.power\",\"value\": 40  },\"id\": 5}");
        }
        else if(strcmp(argv[2],"laser90") == 0)
        {
            strcpy((char*) commandBuffer, "{\"jsonrpc\": \"2.0\",\"method\": \"property.set\",\"params\": {\"property\": \"illumination.sources.laser.power\",\"value\": 90  },\"id\": 5}");
        }
        else if(strcmp(argv[2],"laser100") == 0)
        {
            strcpy((char*) commandBuffer, "{\"jsonrpc\": \"2.0\",\"method\": \"property.set\",\"params\": {\"property\": \"illumination.sources.laser.power\",\"value\": 100  },\"id\": 5}");
        }
        else
        {
            commandBuffer[0] = '\n';
        }
        
        // cout << "cmd: " << commandBuffer << endl;
        // cout << "len: " << strlen((char*)commandBuffer) << endl;

        commandLength = strlen((char*)commandBuffer);

        if (commandLength > 1)
        {
            int numWritten = projector->getSocket()->write(commandBuffer, commandLength);
            if (numWritten < commandLength)
            {
                cerr << "error: could not send all bytes, only" << numWritten << " of " << commandLength << endl;
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
                    }
                    while (((projector->getSocket()->getErrno() == WSAEINPROGRESS) || (projector->getSocket()->getErrno() == WSAEINTR)));
#else
                        numRead = ::read(projector->getSocket()->get_id(), replyBuffer, 1000);
                    }
                    while ((errno == EAGAIN || errno == EINTR));
#endif
                
                    if (numRead > 0)
                    {
                        //cout << "recv: " << replyBuffer << endl;
                        return 0;
                    }
                    
                }
                while (time(NULL) < startZeit + 1);
                cerr << "error: timeout" << endl;
            }
        }
        else
        {
            cerr << "error: unknown command" << endl;
        }
    }
    else
    {
        cerr << "error: could not connect to projector " << argv[1] << " on port " << PROJECTOR_CMD_PORT << endl;
    }

    return -1;
}

// ----------------------------------------------------------------------------
