#ifndef _READ_CALCULIX_H
#define _READ_CALCULIX_H
/**************************************************************************\
**                                               	  (C)2015 Stellba **
**                                                                        **
** Description: READ Calculix FEM                                         **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
** Author: Martin Becker                                                  **
**                                                                        **
\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <api/coModule.h>
using namespace covise;


class ReadCalculix: public coModule
{

   private:

      //  member functions
      virtual int compute(const char *port);

      //  member data
      coOutputPort *p_mesh;
      coOutputPort *p_normalStress;
      coOutputPort *p_displacement;
      coOutputPort *p_vonMises;
      coOutputPort *p_strain;
      
      coFileBrowserParam *p_inpFile;
      //coBooleanParam	 *p_readFrd;
      coFileBrowserParam *p_frdFile;
      coIntScalarParam	 *p_data_step_to_read;
      
      bool containsLetters(string t);

   public:

      ReadCalculix(int argc, char *argv[]);
      virtual ~ReadCalculix();

};

#endif

