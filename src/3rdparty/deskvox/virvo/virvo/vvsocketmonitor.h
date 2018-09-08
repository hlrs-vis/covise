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

#ifndef VV_SOCKETMONITOR_H
#define VV_SOCKETMONITOR_H

#include "vvsocket.h"

#include <vector>

/** Class handling non-blocking file-describtors with select

  For usage, use setters ((setReadFds(), setWriteFds(),
  setErrorFds()) to fill the filedescribtor-sets and call wait(),
  which will handle all sets at once and provide the first socket
  with any events to handle.

  @author Stefan Zellmann (zellmans@uni-koeln.de)
  @author Stavros Delisavas (stavros.delisavas@uni-koeln.de)
 */
class vvSocketMonitor
{
public:

  enum ErrorType
  {
    VV_OK,
    VV_TIMEOUT,
    VV_ERROR
  };

  /** Constructor creating a socketmonitor */
  vvSocketMonitor();
  ~vvSocketMonitor();

  /** Add a List of filedescribtors for reading
    \param readfds vector of vvSockets
    */
  void setReadFds (const std::vector<vvSocket*>& readfds);
  /** Add a List of filedescribtors for writing
    \param writefds vector of vvSockets
    */
  void setWriteFds(const std::vector<vvSocket*>& writefds);
  /** Add a List of filedescribtors for errors
    \param errorfds vector of vvSockets
    */
  void setErrorFds(const std::vector<vvSocket*>& errorfds);

  /** Wait until one of all filedescribtors throws an event

    \param socket pointer set to the first socket ready or NULL if an error occured
    \param timeout pointer to timeout, decreased while time goes by. NULL for no timeout.

    \return VV_OK if select returned normally and socket is set
    \return VV_TIMEOUT if timeout reached. socket is NULL and timeout (if given) is set to 0
    \return VV_ERROR on any error.
    */
  ErrorType wait(vvSocket** socket, double* timeout = NULL);
  /** Delete all Sockets and emtpy the filedescribtor-sets
    */
  void clear();
private:
  fd_set _readsockfds;
  fd_set _writesockfds;
  fd_set _errorsockfds;

  std::vector<vvSocket*> _readSockets;
  std::vector<vvSocket*> _writeSockets;
  std::vector<vvSocket*> _errorSockets;

  vvsock_t _highestSocketNum;
};

#endif

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
