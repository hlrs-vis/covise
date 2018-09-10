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

#include "vvclock.h"

using namespace std;

int main(int, char**)
{
  vvStopwatch* watch;
  char input[128];

  watch = new vvStopwatch();

  cerr << "Current time in seconds: " << watch->getTime() << endl;
  cerr << "Input something to start stopwatch: " << endl;
  cin >> input;
  watch->start();

  cerr << "Input something to get time since start: " << endl;
  cin >> input;
  cerr << "Time since start: " << watch->getTime() << endl;

  cerr << "Input something to get Time since last call: " << endl;
  cin >> input;
  cerr << "Time difference: " << watch->getDiff() << endl;

  cerr << "Input something to get total time: " << endl;
  cin >> input;
  cerr << "Total time: " << watch->getTime() << endl;

  delete watch;

  return 0;
}
