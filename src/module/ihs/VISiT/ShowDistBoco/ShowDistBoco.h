#ifndef _READ_ELMER_H
#define _READ_ELMER_H
#define RNUM 16

#include <stdlib.h>
#include <stdio.h>

#ifndef WIN32
#include <unistd.h>
#endif

#include <math.h>

#include <api/coModule.h>
using namespace covise;

class ShowDistBoco: public coModule
{

   private:

      //  member functions
      virtual int compute(const char *port);

      //  in- and ouput ports
      coInputPort *gridInPort;
      coInputPort *distbocoInputPort;
      coInputPort *cellInPort;

      coOutputPort *velocityOutPort;

   public:

      ShowDistBoco(int argc, char *argv[]);

};
#endif
