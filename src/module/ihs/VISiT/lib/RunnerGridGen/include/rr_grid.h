#ifndef RR_GRID_INCLUDE
#define RR_GRID_INCLUDE

#include "../../General/include/curve.h"
#include "../../General/include/points.h"
#include "../../General/include/flist.h"
#include "../../General/include/ilist.h"
#include "../../General/include/nodes.h"
#include "../../General/include/elements.h"

#define INPUT_FILE_ERROR  1
#define GRID_TYPE_ERROR   2
#define PSEDIS_ERROR      3
#define NO_GEOMETRY_ERROR 4
#define IMPLEMENT_ERROR   5
#define TEDISCRETE_ERROR  6

#define  RG_COL_NODE        2
#define  RG_COL_ELEM        2
#define  RG_COL_DIRICLET    2
#define  RG_COL_WALL        7
#define  RG_COL_BALANCE     7
#define  RG_COL_PRESS       7

#define CLASSIC  1
#define MODIFIED 2
#define ISOMESH  3

#ifndef NPOIN_EXT
#define NPOIN_EXT 10
#endif

#ifndef NPE_BC
#define NPE_BC 4
#endif

struct region {
    int   numl;                          // number of lines
    struct Point **line;                 // region consists of border lines
    struct Flist **arc;                  // arc length, radius * circumf. angle
    struct Flist **para;                 // line point parameters
    struct Ilist **nodes;                // global node indices of region's nodes
};

struct cgrid {
    int reg_num;                         // number of regions
    struct Point *cl;                    // center line
    struct Flist *clarc;                 // arc length on circle
    struct curve *ps;                    // pressure side
    struct curve *ss;                    // suction side
    struct region **reg;                 // grid regions
};

struct ge {
    float para;                          // meridian parameter
    struct curve *ml;                    // meridian curve
    struct Point *cl;                    // center line (interpolated)
    struct Point *ps;                    // pres. side surf. (intpol.)
    struct Point *ss;                    // suct. side surf. (intpol.)
};

struct bc {
    float bcQ;                           // discharge
    float bcH;                           // head
    float bcN;                           // revs
    float bcAlpha;                       // flow angle
    float vratio;                        // velocity ratio from geometry module
    float cm;                            // meridian velocity
    int useAlpha;                        // get flow angle from Euler (0), input(1)
    int useQ;                            // use Q or given velocities (0)
};

struct rr_grid {
    int type;                            // grid type
    int     write_grid;                  // write grid (0/1) button.
    int     rot_clock;                   // rotation clockwise (0/1)
    int     create_inbc;                 // create inlet bcs.
    int     alpha_const;                 // constant inlet angle?
    int     turb_prof;                   // turbulent inlet profile?
    int     mesh_ext;                    // mesh inlet extension or not.
    int     rot_ext;                     // inlet ext. rotating (1) or not!
    int     iinlet;                      // index of runner inlet point on meridian.
    int     ioutlet;                     // points on outlet region
    int     le_dis;                      // number of points on blade in stagn. pt. area
    int     jadd;                        // add points to inlet
    int     skew_runin;                  // get phi[0] for each ge-plane.
    float phi_scale[2];                  // scaling factor for phi (inlet/outlet)
    float phi_skew[2];                   // skew parameters (hub/shroud)
    float phi_skewout[2];                // skew parameters outlet (hub/shroud)
	
    float phi0_ext;                      // scaling factor for phi0 for ext.
    float angle_ext[2];                  // tangent angle for spline vector at inlet, 1.1
    float bl_scale[2];                   // scale factor, b.l. thickness
    float   v14_angle[2];                // tangent angles for spline 1.4 vectors
    float   bl_v14_part[2];              // partition ratios, where b.l. starts on 1.4
    float ss_part[2];                    // ss partition factor hub/shroud (0/1)
    float ps_part[2];                    // ps envelope ext. part. factors
    float ssle_part[2];                  // ss leading edge ratio hub/shroud
    float psle_part[2];                  // ps leading edge ratio
    float out_part[2];                   // partition, outlet
    int     ge_num;                      // number of grid elements
    float   ge_bias;                     // grid elem. bias
    int     ge_type;                     // bias type
#ifdef GAP
    int     gp_num;                      // number of grid elements in gap
    float   gp_bias;                     // bias
    int     gp_type;                     // bias type
    int     gpreg_num;                   // number of addtl. regions for gap
#endif
    int     extdis;                      // discretization, inlet extension
    float extbias;                       // bias factor for this edge
    int      extbias_type;               // bias type
    int      cdis;                       // circumferential discretization, inlet edge
    float cbias;                         // bias factor for this edge
    int      cbias_type;                 // bias type
    int      cledis;                     // circumferential discretization, leading edge
    float clebias;                       // bias factor for this edge
    int      clebias_type;               // bias type
    int      ssmdis;                     // inlet region, meridional discretization
    float ssmbias;                       // bias factor
    int      ssmbias_type;               // bias type
    int      psdis;                      // pressure side, along ps
    float psbias;                        // bias factor
    int      psbias_type;                // bias type
    int      psedis;                     // pressure side, envelope
    float psebias;                       // bias factor
    int      psebias_type;               // bias type
    int      ssdis;                      // suction side, along ss
    float ssbias;                        // bias factor
    int      ssbias_type;                // bias type
    float midbias;                       // bias factor ss-ps-connection
    int      midbias_type;               // bias type
    int      lowdis;                     // lower part, suction side
    float lowbias;                       // bias factor
    int      lowbias_type;               // bias type
    int      lowindis;                   // lower part, inner curve
    float lowinbias;                     // bias factor
    int      lowin_type;                 // bias type
	int      ssxdis;                     // outlet, suction side
    float ssxbias;                       // bias factor ss-exit, outlet
    int      ssxbias_type;               // bias type for this line
	int      psxdis;                     // outlet, pressure side
    float psxbias;                       // bias factor ps-exit, outlet
    int      psxbias_type;               // bias type for this line
    float cxbias;                        // bias factor center outlet (x..exit)
    int      cxbias_type;                // bias type for this line
    int      reg_num;                    // number or regions
    int      numl;                       // standard number of lines per region
    struct ge **ge;                      // grid element
    struct cgrid **cge;                  // circumferential grid in meridional plane
    struct Nodelist *n;                  // all nodes
    struct Element  *e;                  // all elements
    struct Ilist *inlet;                 // inlet nodes
    struct Ilist *psle;                  // leading edge, ps, periodic
    struct Ilist *ssle;                  // leading edge, ss, periodic
    struct Ilist *psnod;                 // blade surface, pressure side
    struct Ilist *ssnod;                 // blade surface, suction side
    struct Ilist *pste;                  // trailing edge, ps, periodic
    struct Ilist *sste;                  // trailing edge, ss, periodic
    struct Ilist *outlet;                // outlet nodes
    struct Element *wall;                // solid wall elements
    struct Element *frictless;           // frictionless wall
    struct Element *shroud;              // shroud elements
    struct Element *shroudext;           // shroud extension
    struct Element *psblade;             // blade surface elements
    struct Element *ssblade;             // blade surface elements
    struct Element *ssleperiodic;        // periodic elements, le suct. side
    struct Element *psleperiodic;        // periodic elements, le pres. side
    struct Element *ssteperiodic;        // periodic elements, te suct. side
    struct Element *psteperiodic;        // periodic elements, te pres. side
    struct Element *einlet;              // inlet elements
    struct Element *eoutlet;             // outlet elements
    struct Element *rrinlet;             // runner inlet
    struct Element *rroutlet;            // runner outlet
    ///all hub elements including frictless elements
    struct Element *shroudAll;
    ///all shroud elements
    struct Element *hubAll;
    struct bc *inbc;                     // values for inlet bcs.
    float  **bcval;                      // bcvalues for nodes;

};

struct region **AllocRRGridRegions(int reg_num, int numl);
struct ge **AllocRRGridElements(int ge_num);
struct cgrid **AllocRRCGridElements(int ge_num);
struct rr_grid *AllocRRGrid();
void FreeRRGridRegions(int reg_num, struct region **reg);
void FreeRRCGridElements(int ge_num, struct cgrid **cge);
void FreeRRGridElements(int ge_num, struct ge **ge);
int FreeRRGrid(struct rr_grid *grid);
int FreeRRGridMesh(struct rr_grid *grid);
#ifdef GRID4COV
int CreateRR_Grid4Covise(struct Nodelist *n, float *x, float *y, float *z,
			 int istart, int iend);
int CreateRR_BClist4Covise(struct Element *e, int *corners, int *poly_list,
						   int *i_corn, int *i_poly,
						   int num_corn, int num_poly, char *buf);
void FreeElemSetPtr(void);
void **Create_ElemSet(int *num, ...);
int CreateRR_BClistbyElemset(struct Nodelist *n,
							 struct Element **e, int num,
							 float *xc, float *yc, float *zc,
							 int *corners, int *poly_list,
							 int num_corn, int num_poly, int num_node,
							 char *buf);
int SetElement(int **data, struct Element *e, int flag);

#endif
#endif                                            // RR_GRID_INCLUDE
