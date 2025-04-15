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

using namespace covise;

class ReadIlpoe : public coModule
{

   private:

      //////////  member functions

      /// this module has only the compute call-back
      virtual int compute(const char*);

      //////// Parameters:
      coFileBrowserParam *filenameParam;

      coOutputPort *p_polyOut;

   public:
     ReadIlpoe(int argc, char** argv);
};
#endif
