#ifndef _GG_SET_H
#define _GG_SET_H

#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>
#include <api/coSimpleModule.h>

using namespace covise;

class DomainDecomposition : public coSimpleModule
{
   COMODULE

   private:

      //  member functions
      virtual int compute(const char *port);
      virtual void  postInst();

      coInputPort *p_boco;
      coInputPort *p_grid;
      //    coOutputPort *p_ogrid;

      coIntScalarParam *p_numPart;
      coStringParam    *p_dir;
      coBooleanParam   *p_writeFiles;
	  coIntScalarParam *p_zerbuflen;
      coDoUnstructuredGrid *MemoGrid;

#ifdef YAC
      virtual void paramChanged(coParam *param);
#endif

   public:

      DomainDecomposition(int argc, char *argv[]);

};
#endif
