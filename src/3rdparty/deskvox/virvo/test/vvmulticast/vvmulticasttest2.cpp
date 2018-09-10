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

#include "vvtcpserver.h"
#include "vvtcpsocket.h"
#include "vvmulticast.h"
#include "vvclock.h"
#include "vvtoolshed.h"

#include <pthread.h>

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

    if(vvSocket::VV_OK != obj->_socket->writeData(obj->_data, obj->_size))
    {
      cout << "Error occured in transfer to " << obj->_addr << endl;
    }

    obj->_done = true;
    return NULL;
  }

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
    ssize_t count = 1024;
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
      sendTimeout = -1.0;
    }

    double startTime = vvClock::getTime();

    // Send to all servers
    cout << "Sending to receivers via TCP...";
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
      vvToolshed::sleep(1);

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
    cout << "ok." << endl << endl;
    cout << "Time needed: " << vvClock::getTime() - startTime << endl;

    cout << endl;

    cout << "Receivers ready?" << endl;
    string tmp;
    cin >> tmp;

    // init Multicaster
    vvMulticast foo = vvMulticast(vvMulticast::VV_SENDER, vvMulticast::VV_VVSOCKET, "224.1.2.3", 50096);

    cout << "Sending to receivers via multicast..." << flush;
    startTime = vvClock::getTime();
    int written = foo.write(reinterpret_cast<const unsigned char*>(bar), count, sendTimeout);
    if(written != count)
      cout << "error!" << endl;
    else
      cout << "ok." << endl;

    cout << "Wait for Collectors answers..." << flush;
    for(unsigned int i = 0;i<servers.size() && count>0; i++)
    {
      char *multidone = new char[6];
      servers[i]->_socket->readString(multidone, 6);
      if(strcmp("done!", multidone) == 0)
        continue;
      else
        cout << "unexpected message from receiver" ;

      delete[] multidone;
    }
    cout << "done" << endl;

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
      cout << "Timeout: " << receiveTimeout << endl << endl;
    }
    else
    {
      receiveTimeout = -1.0;
    }

    vvMulticast foo = vvMulticast(vvMulticast::VV_RECEIVER, vvMulticast::VV_VVSOCKET, "224.1.2.3", 50096);

    cout << "Waiting for incoming data on TCP..." << endl;

    vvTcpServer server = vvTcpServer(31050);
    vvTcpSocket *recSocket = server.nextConnection();

    uint tcpSize = recSocket->read32();
    cout << "Expecting " << tcpSize << "Byes...";
    uchar* bartcp = new uchar[tcpSize];
    if(vvSocket::VV_OK == recSocket->readData(bartcp, tcpSize))
      cout << "ok." << endl;
    else
      cout << "error!" << endl;
    cout << endl;

    uchar* bar = generateData(tcpSize);

    cout << "Waiting for incoming data over Multicasting..." << flush;

    uchar* bartext = new uchar[tcpSize];
    int read = foo.read(bartext, tcpSize);

    // Tell sender, that we are done
    recSocket->writeString("done!");

    if(read >= 0)
    {
      cout << "ok." << endl;
      cout << endl;
      cout << "Check data for differences...    ";
      for(uint i=0; i<tcpSize;i++)
      {
        if(bar[i] != bartext[i])
        {
          cout << "Failed: Differences found on byte " << i << endl;
          break;
        }
        else if(i % 1024 == 0)
        {
          cout << "\r" << flush;
          cout << "Check data for differences..." << int(100 * float(i)/float(tcpSize)) << "%" << flush;
        }
      }
      cout << endl;
    }
    else
      cout << "error!" << endl;

    delete[] bar;
    delete[] bartext;
    delete[] bartcp;
    delete recSocket;
    return 0;
  }

  cout << "Nothing done..." << endl;

  return 1;
}
