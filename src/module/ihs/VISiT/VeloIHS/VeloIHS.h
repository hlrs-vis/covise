/**************************************************************************\ 
 **                                                              2002      **
 **                                                                        **
 ** Description:  COVISE VeloIHS     New application module               **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:  M. Becker                                                     **
 **                                                                        **
 **                                                                        **
 ** Date:  01.07.02  V1.0                                                  **
\**************************************************************************/
#include <util/coviseCompat.h>
#include <api/coSimpleModule.h>
using namespace covise;

class VeloIHS : public coSimpleModule
{
   COMODULE

   private:

      virtual int compute(const char *port);

      // parameters

      coChoiceParam *p_rotaxis;

      // ports
      coInputPort *p_grid;
      coInputPort *p_velo_in;

      coOutputPort *p_vu_vector_out;
      coOutputPort *p_vr_vector_out;
      coOutputPort *p_vm_vector_out;

      coOutputPort *p_v_scalar_out;
      coOutputPort *p_vu_scalar_out;
      coOutputPort *p_vr_scalar_out;
      coOutputPort *p_vm_scalar_out;
      coOutputPort *p_rvu_scalar_out;

      enum RotAxisChoice
      {
         RotX = 0x00,
         RotY = 0x01,
         RotZ = 0x02
      };

      char *s_rotaxis[3];

      // private data

   public:

      VeloIHS(int argc, char *argv[]);

};
