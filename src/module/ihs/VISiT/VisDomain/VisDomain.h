#ifndef _VisDomain_H
#define _VisDomain_H

#include <appl/ApplInterface.h>
using namespace covise;
#include <api/coModule.h>
using namespace covise;
#include <stdlib.h>
#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#endif

#define BC_SELECTS   7
#define UD_SELECTS   7
#ifndef WIN32
#define CM_MAX    5
#else
#define CM_MAX    6
#endif
#define SA_MAX    2

class VisDomain : public coModule
{

   private:

      //////////  member functions

      virtual int compute(const char *port);
      coInputPort            *p_grid, *p_boco, *p_boco2, *p_in_bcin;
      coOutputPort           *p_blocknum;

      char *bcSelects[BC_SELECTS];
      char *udSelects[UD_SELECTS];

   public:

      VisDomain(int argc, char *argv[]);

      virtual ~VisDomain() {}

};
#endif                                            // _VisDomain_H
