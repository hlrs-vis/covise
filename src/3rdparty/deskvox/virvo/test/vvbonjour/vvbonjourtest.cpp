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

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef HAVE_BONJOUR

#include "vvbonjour/vvbonjour.h"
#include "vvbonjour/vvbonjourregistrar.h"
#include "vvbonjour/vvbonjourbrowser.h"
#include "vvbonjour/vvbonjourresolver.h"
#include "vvsocketio.h"
#include "vvtcpserver.h"

#include <sstream>

using namespace std;

int main(int argc, char **argv)
{
  // don't forget arguments
  if(argc < 2)
  {
    cerr << "Start with option -server or -client" << endl;
    return 0;
  }

  // --------------------------------------
  // # Server #
  // --------------------------------------
  if(strcmp("-server", argv[1])== 0)
  {
    ushort port;
    if(argc == 3)
    {
      port = atoi(argv[2]);
    }
    else
    {
      port = rand();
    }

    while(true)
    {
      vvTcpServer tcpServer = vvTcpServer(port);
      //vvSocket sock = vvSocket(port, vvSocket::VV_TCP);


      cerr << "Register service on port " << port << "..." << flush;
      vvBonjourEntry entr = vvBonjourEntry("vvBonjourServer", "_bonjourtest._tcp", "");
      vvBonjourRegistrar registrar;
      registrar.registerService(*&entr, port);
      cerr << "done." << endl;


      cerr << "Wait for connection on port " << port << "...";
      vvTcpSocket* sock;
      if((sock = tcpServer.nextConnection()) != NULL)
      {
        cerr << "done." << endl;
      }
      else
      {
        cerr << "error." << endl;
        return 1;
      }

      cerr << "Unregister service...";
      registrar.unregisterService();
      cerr << "done." << endl;

      vvSocketIO sockio = vvSocketIO(reinterpret_cast<vvSocket*>(sock));
      bool bla = false;
      sockio.getBool(bla);

      cerr << "message read: " << bla << endl;
    }

    return 0;
  }
  // --------------------------------------
  // # Client #
  // --------------------------------------
  else if(strcmp("-client", argv[1])== 0)
  {
    cerr << "Browse for services..." << flush;
    vvBonjourBrowser br;
    br.browseForServiceType("_bonjourtest._tcp");
    cerr << "done." << endl;
    std::vector<vvBonjourEntry> entries = br.getBonjourEntries();
    cerr << entries.size() << " entries found." << endl;

    for (std::vector<vvBonjourEntry>::const_iterator it = entries.begin(); it != entries.end(); ++it)
    {
      cerr << "Resolve entry..." << flush;
      vvBonjourResolver resol;
      if(vvBonjour::VV_OK == resol.resolveBonjourEntry(*it))
      {
        cerr << "done." << endl;

        cerr << "Connecting with " << resol._hostname.c_str() << resol._port << "...";
        vvTcpSocket sock = vvTcpSocket();
        sock.connectToHost(resol._hostname.c_str(), resol._port);
        cerr << "done." << endl;

        cerr << "Send bool 1...";
        vvSocketIO sockio = vvSocketIO(reinterpret_cast<vvSocket*>(&sock));
        sockio.putBool(true);
        cerr << "done." << endl;
      }
      else
      {
        cerr << "error." << endl;
      }
    }

    // OR USE Wrapper instead:

    /*
    vvBonjour bonj;

    std::vector<vvSocket*> sockets = bonj.getSocketsFor("_bonjourtest._tcp");

    cerr << "Socket-count" << sockets.size() << endl; */

    return 0;
  }
  else
  {
    cerr << "Unknown parameter" << endl;
    return 1;
  }
  return 1;
}

#else
#include <iostream>
int main(int, char **)
{
  std::cerr << "Could not compile with Bonjour. Libs not found?" << std::endl;
}
#endif
