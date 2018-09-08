// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
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

#ifndef _VV_BONJOUREVENTLOOP_H_
#define _VV_BONJOUREVENTLOOP_H_

#include "vvbonjour.h"
#include "vvexport.h"

/**
  Class to handle Bonjourevents in a blocking or nonblocking/threaded loop.
  In general this class is never needed directly and only used by the other Bonjour-classes.

  @author Stavros Delisavas (stavros.delisavas@uni-koeln.de
  */
class VIRVOEXPORT vvBonjourEventLoop
{
public:
  vvBonjourEventLoop(void *service);
  ~vvBonjourEventLoop();

  /**
    Start the eventloop.
    @param inThread indicates to use a pthread or block until all bonjourevents are handled.
    @param timeout  maximum timeout in seconds.
    */
  void run(bool inThread = false, double timeout = 1.0);
  /**
    Do not call. Use run() instead. Static function necessary for thread-creation only.
    */
  static void * loop(void * attrib);
  /**
    Cut short the eventloop eventhough not done. E.g. if timeout not reached yet but already enough services handled anyway.
    */
  void stop();

  double            _timeout;
  bool              _run;
  bool              _noMoreFlags;
private:
  struct Thread;
  struct BonjourData;

  Thread      *_thread;
  BonjourData *_bonjourData;
};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
