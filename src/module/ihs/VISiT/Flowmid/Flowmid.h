#ifndef _READ_ELMER_H
#define _READ_ELMER_H
#define RNUM 16

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "api/coModule.h"

class Flowmid: public coModule
{

   private:

      //  member functions
      virtual int compute(const char *port);

      //  in- and ouput ports
      coInputPort *distbocoDSInPort;
      coInputPort *outletcellUSInPort;
      coInputPort *velocityUSInPort;

      coInputPort *distbocoUSInPort;
      coInputPort *inletcellDSInPort;
      coInputPort *pressureDSInPort;

      coOutputPort *distbocoDSOutPort;
      coOutputPort *distbocoUSOutPort;

      // parameters
      coChoiceParam *interpolationTypeChoice;
      coBooleanParam *downStream;
      coBooleanParam *upStream;

      map<string,int> choiceMap;
      map<int,string> sortedChoiceMap;
      char **interp_choice;

   public:

      Flowmid(int argc, char *argv[]);
      void setInterpolatorList(char *);

};
#endif
