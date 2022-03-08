/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/environment.h>
#include "handler.h"

int main(int argc, char **argv)
{
  covise::setupEnvironment(argc, argv);
  covise::controller::CTRLHandler(argc, argv);
  return 0;
}
