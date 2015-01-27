#ifndef _HELLO_H
#define _HELLO_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: "Hello, world!" in COVISE API                          ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Andreas Werner                           ++
// ++               Computer Center University of Stuttgart               ++
// ++                            Allmandring 30                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  10.01.2000  V2.0                                             ++
// ++**********************************************************************/

#include <api/coModule.h>

class Hello : public coModule
{

   private:

      //////////  member functions

      /// this module has only the compute call-back
      virtual int compute(const char *port);

#if defined(E1) || defined(E2)
      coStringParam * stringParam;
#endif
#ifdef E2
      coBooleanParam * boolParam;
      virtual void param(const char * paramName, bool inMapLoading);
#endif

#if defined(E3) || defined(E5)
      coOutputPort * geoOutPort;
#endif
#ifdef E4
      coOutputPort * dataOutPort;
#endif
   public:

      Hello(int argc, char *argv[]);

};
#endif
