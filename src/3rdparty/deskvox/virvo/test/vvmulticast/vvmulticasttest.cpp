// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 2010 University of Cologne
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <pthread.h>

#include "vvmulticast.h"
#include "vvtcpserver.h"
#include "vvtcpsocket.h"
#include "vvclock.h"
#include "vvtoolshed.h"

using namespace std;

//-------------------------------------------------------------------
// Threads for parallel TCP-Communication
//-------------------------------------------------------------------
class tcpConnection
{
public:
  tcpConnection(string addr, uchar* data, int size)
  {
    _addr = addr;
    _data = data;
    _size = size;
    _done = false;

    // socket connection
    int delim = addr.find_first_of(":");
    string port = addr.substr(delim+1);
    addr = addr.substr(0, delim);
    _socket = new vvTcpSocket();

    if(vvSocket::VV_OK == _socket->connectToHost(addr.c_str(), atoi(port.c_str())))
    {
      _status = true;
      _return = pthread_create( &_thread, NULL, threadMain, this);
    }
    else
    {
      _status = false;
    }
  }

  ~tcpConnection()
  {
    delete _socket;
  }

  static void* threadMain(void* attrib )
  {
    tcpConnection *obj = static_cast<tcpConnection*>(attrib);

    // send data via socket
    if(vvSocket::VV_OK != obj->_socket->write32(obj->_size))
    {
      cout << "Error occured in transfer to " << obj->_addr << "!" << endl;
      return NULL;
    }

    if(vvSocket::VV_OK == obj->_socket->writeData(obj->_data, obj->_size))
    {
      cout << "transfer to " << obj->_addr << " complete!" << endl;
    }
    else
    {
      cout << "Error occured in transfer to " << obj->_addr << endl;
    }

    int node;
    node = obj->_socket->read32();
    {
      obj->_node = (NormNodeId)node;
      cout << "Reading NodeId "<< node << endl;
    }

    obj->_done = true;
    return NULL;
  }

  NormNodeId   _node;
  bool         _status;
  bool         _done;
  string       _addr;
  uchar       *_data;
  uint         _size;
  int          _return;
  pthread_t    _thread;
  vvTcpSocket *_socket;
};

uchar* generateData(const int size)
{
  // init data to be sent
  cout << "Prepare " << size << " Bytes of random data to be send/checked..." << flush;
  uchar* bar = new uchar[size];

  srand(123456);
  for(int i=0; i<size; i++)
  {
    uchar num = rand() % 256;
    bar[i] = num;
  }
  cout << "done!" << endl;
  return bar;
}

int main(int argc, char** argv)
{
  // don't forget arguments
  if(argc < 2)
  {
    cout << "Start with -sender or -receiver" << endl;
    return 0;
  }

  // ----------------------------------------------------------------
  // Sender
  // ----------------------------------------------------------------
  if(strcmp("-sender", argv[1])== 0)
  {
    cout << "Sender-Mode" << endl;
    cout << "###########" << endl;
    cout << endl;

    // Number of Bytes to be send/received
    uint count = 1024;
    if(NULL != argv[2] && 3 <= argc)
    {
      count = atoi(argv[2]);
    }
    else
    {
      cout << "No number of bytes given. (Default: " << count << ")" << endl;
    }

    uchar* bar = generateData(count);

    // timeout
    double sendTimeout;
    if(NULL != argv[3] && 4 <= argc)
    {
      sendTimeout = atof(argv[3]);
      cout << "Timeout set to " << sendTimeout << endl;
    }
    else
    {
      cout << "No timeout given. (Default: no timeout)" << endl;
      sendTimeout = -1.0;
    }

    double startTime = vvClock::getTime();

    // Send to all servers
    cout << "Try sending to receivers via TCP-Connection!" << endl;
    std::vector<tcpConnection*> servers;
    for(int i=0; i<argc; i++)
    {
      if(0 == strcmp("-server", argv[i]))
      {
        servers.push_back(new tcpConnection(argv[i+1], bar, count));
        if(false == servers.back()->_status)
        {
          return 1;
        }
        i++;
      }
    }
    bool done = false;
    while(!done)
    {
      vvToolshed::sleep(100);

      done = true;
      for(unsigned int i = 0;i<servers.size(); i++)
      {
        if(false == servers[i]->_done)
        {
          done = false;
          break;
        }
      }
    }
    cout << "TCP-Transfers complete!" << endl << endl;
    cout << "Time needed: " << vvClock::getTime() - startTime << endl;

    cout << endl << endl;

    cout << "Receiver ready?" << endl;
    string tmp;
    cin >> tmp;

    // init Multicaster
    vvMulticast foo = vvMulticast(vvMulticast::VV_SENDER, vvMulticast::VV_NORM, "224.1.2.3", 50096);

    for(unsigned int i = 0;i<servers.size(); i++)
    {
      foo._nodes.push_back(servers[i]->_node);
    }

    cout << "Sending " << count << " Bytes of random numbers..." << flush;
    startTime = vvClock::getTime();
    int sendBytes = foo.write(reinterpret_cast<const unsigned char*>(bar), count, sendTimeout);
    cout << "sendBytes = " << sendBytes << endl;
    cout << "sendTimeout = " << sendTimeout << endl;
    for(unsigned int i = 0;i<servers.size() && sendBytes>0; i++)
    {
      char *multidone = new char[5];
      servers[i]->_socket->readString(multidone, 5);
      if(strcmp("done!", multidone) == 0)
        continue;
      else
        cout << "Server did not finish!"<< endl;

      delete[] multidone;
    }
    cout << "done!" << endl;

    if(sendBytes == -1 || sendBytes == 0)
      cout << "Error occured! (No Norm found?) " << sendBytes << endl;
    else
      cout << "Successfully sent " << sendBytes << " Bytes!" << endl;

    cout << "Time needed: " << vvClock::getTime() - startTime << endl;
    cout << endl;

    delete[] bar;

    return 0;
  }

  // ----------------------------------------------------------------
  // Receiver
  // ----------------------------------------------------------------
  if(strcmp("-receiver", argv[1])== 0)
  {
    cout << "Receiver-Mode" << endl;
    cout << "#############" << endl;
    cout << endl;

    // timeout
    double receiveTimeout;
    if(NULL != argv[3] && 4 <= argc)
    {
      receiveTimeout = atof(argv[3]);
    }
    else
    {
      cout << "No timeout given. (Default: no timeout)" << endl;
      receiveTimeout = -1.0;
    }

    vvMulticast foo = vvMulticast(vvMulticast::VV_RECEIVER, vvMulticast::VV_NORM, "224.1.2.3", 50096);

    cout << "Waiting for incoming data on TCP..." << endl;

    vvTcpServer server = vvTcpServer(31050);

    vvSocket *recSocket = server.nextConnection();
    uint tcpSize = recSocket->read32();
    cout << "Expecting " << tcpSize << "Byes of data." << endl;
    uchar* bartcp = new uchar[tcpSize];
    if(vvSocket::VV_OK == recSocket->readData(bartcp, tcpSize))
    {
      cout << "Successfully received " << tcpSize << "Bytes. (Node: " << foo._nodes[0] << ")" << endl;
      recSocket->write32(foo._nodes[0]);
    }
    else
    {
      cout << "Error for TCP-transfer!" << endl;
    }
    cout << endl;

    uchar* bar = generateData(tcpSize);

    cout << "Waiting for incoming data over Multicasting..." << endl;

    uchar* bartext = new uchar[tcpSize];
    int receivedBytes = foo.read(bartext, tcpSize, receiveTimeout);

    // Tell sender, that we are done
    recSocket->writeString("done!");

    cout << "Received: " << receivedBytes << endl;
    if(0 == receivedBytes)
      cout << "Timeout reached and no data received!" << endl;
    cout << endl;

    cout << "Check data for differences...    ";
    for(int i=0; i<receivedBytes;i++)
    {
      if(bar[i] != bartext[i])
      {
        cout << "Failed: Differences found!" << endl;
        cout << "i:         " << i           << endl
             << "checkdata: " << bar[i]      << endl
             << "multicast: " << bartext[i]  << endl
             << "tcpdata:   " << bartcp[i]   << endl;
        break;
      }
      else if(i % 1024 == 0)
      {
        cout << "\r" << flush;
        cout << "Check data for differences..." << int(100 * float(i)/float(tcpSize)) << "%" << flush;
      }
    }
    cout << endl;

    delete[] bar;
    delete[] bartext;
    delete[] bartcp;
    delete recSocket;
    return 0;
  }

  cout << "Nothing done..." << endl;

  return 1;
}

/*

Build-Notes:

build with libraries: virvo, norm, Protokit

Attention!
If Protokit and NORM are build seperately, then use libProto.a instead of libProtokit.a! (both libs are build automatically)
If Protokit is build together with norm (included subdirectory) then use libProtokit.a (only this lib is build)
(Same thing for windows and .dlls)
The reason for this issue is, that norm-developers use an older/different version of Protokit and are too lazy to fix this.
NORM is generally build with the old version instead.

*/

