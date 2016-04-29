/**************************************************************************\ 
 **                                                              2002      **
 **                                                                        **
 ** Description:  COVISE ReadIHS2     New application module               **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:  M. Becker                                                     **
 **                                                                        **
 **                                                                        **
 ** Date:  01.07.02  V1.0                                                  **
\**************************************************************************/

#include <api/coModule.h>
using namespace covise;
#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>
#include "./include/geosimrb.h"

#define COL_NODE 2
#define COL_ELEM 2
#define COL_DIRICLET 2
#define COL_WALL 7
#define COL_BALANCE 6
#define COL_PRESS 6

class ReadIHS2 : public coModule
{

   private:

      virtual int compute(const char *port);
      virtual void  postInst();
      virtual void  quit();

      int ReadGeoSimRB(struct geometry *geo);
      void Check_geo_sim_rb(struct geometry *geo);
      int Data2Covise(struct geometry *geo, coDoVec3 *velocity, coDoFloat *press,
         coDoFloat *k, coDoFloat *eps);
      int CreateBocoObject(coDistributedObject **partObj, struct geometry *geo, const char *basename);
      int checkfornewnr (int *nrlist, int *n_nr, char **names, int nr, int last_nr, char *name);
      int countnrs (int n_nrs, int *nr_list, int nr);

      struct geometry   *geo;

      // parameters
      coFileBrowserParam *p_geoFile;
      coFileBrowserParam *p_rbFile;
      coFileBrowserParam *p_simFile;

      coBooleanParam *p_readsim;
      coIntScalarParam *p_dimension;       // if geo-file does not contain dim
      coFloatParam *p_scalingfactor;
      coBooleanParam *p_numbered;          // numbered connectivity list?
      coBooleanParam *p_showallbilas;
      coBooleanParam *p_showallwalls;
      coBooleanParam  *p_create_boco_obj;
      coBooleanParam  *p_generate_inlet_boco;
      coBooleanParam  *p_abs2rel;
      coFloatParam  *p_n;
      coChoiceParam  *p_RotAxis;

      enum RotAxisChoice
      {
         RotX = 0x00,
         RotY = 0x01,
         RotZ = 0x02
      };

      coIntScalarParam *p_bilanr1;
      coIntScalarParam *p_bilanr2;
      coIntScalarParam *p_bilanr3;
      coIntScalarParam *p_bilanr4;
      coIntScalarParam *p_bilanr5;

      coIntScalarParam *p_wallnr1;
      coIntScalarParam *p_wallnr2;
      coIntScalarParam *p_wallnr3;
      coIntScalarParam *p_wallnr4;
      coIntScalarParam *p_wallnr5;

      coIntScalarParam *p_pressnr1;
      coIntScalarParam *p_pressnr2;
      coIntScalarParam *p_pressnr3;
      coIntScalarParam *p_pressnr4;
      coIntScalarParam *p_pressnr5;

      coIntScalarParam *p_bila_in;
      coIntScalarParam *p_bila_out;
      coIntScalarParam *p_periodic_1;
      coIntScalarParam *p_periodic_2;

      // ports
      coOutputPort *port_grid;
      coOutputPort *port_velocity;
      coOutputPort *port_pressure;
      coOutputPort *port_k;
      coOutputPort *port_eps;
      coOutputPort *port_wall;
      coOutputPort *port_pressrb;
      coOutputPort *port_bila;
      coOutputPort *port_boco;
	  coOutputPort *port_bcin;

      char *s_RotAxis[3];

   public:

      ReadIHS2(int argc, char *argv[]);

};
