#ifndef _READ_VATECH_H
#define _READ_VATECH_H

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <api/coModule.h>
using namespace covise;

class ReadVATECH: public coModule
{

   private:

      //  member functions
      virtual int compute(const char *port);
      virtual void postInst();

      coOutputPort *gridOutPort;
      coOutputPort *velocityOutPort;
      coOutputPort *relVelocityOutPort;
      coOutputPort *pressureOutPort;

      coFileBrowserParam *filenameGrid;
      coFileBrowserParam *filenameEuler;
      coFloatParam *omega;
      coFloatParam *gridnorm;

   public:

      ReadVATECH();
};
#endif                                            // _READ_VATECH_H
