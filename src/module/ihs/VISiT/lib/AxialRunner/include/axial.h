#ifndef AXIAL_H_INCLUDED
#define AXIAL_H_INCLUDED

#include "../../General/include/geo.h"
#include "../../General/include/points.h"
#include "../../General/include/curve.h"
#include "../../General/include/flist.h"
#include "../../General/include/parameter.h"
#include "../../General/include/profile.h"

#define RAD(x)          (float((x)*M_PI/180.0))
#define GRAD(x)         (float((x)*180.0/M_PI))
#define ABS2REL(x,y)    ((x)/(y))
#define REL2ABS(x,y)    ((x)*(y))

#define INIT_PORTION 25
#define BSPLN_DEGREE 3

#define DESIGN_DATA_ERR 1
#define BLADE_CONTOUR_INTERSECT_ERR 2
#define EULER_ANGLE_ERR 3
#define INLET_RADIUS_ERR 4
#define INLET_HEIGHT_ERR 5
#define BLADE_MERID_ERR 6
#define PUT_BLADEDATA_ERR 7

// index of lock parameters in axial->parlock
#define LOCK_INANGLE 0
#define LOCK_OUTANGLE 1
#define LOCK_INMOD 2
#define LOCK_OUTMOD 3
#define LOCK_PTHICK 4
#define LOCK_TETHICK 5
#define LOCK_MAXCAMB 6
#define LOCK_CAMBPOS 7
#define LOCK_BPSHIFT 8
#define NUM_PARLOCK 9                             // highest index + 1!!!

struct be
{
   float  para;                                   // rel. Radius [0..1]
   float  le_part[3];                             // leading edge polygon partition
   float  te_part[2];                             // trailing edge polygon partition
   float  angle[2];                               // inlet[0] and outlet [1] blade angle
   float  rot_abs[2];                             // cu, rot. part of abs. vel.
   float  mer_vel[2];                             // meridional velocity
   float  cir_vel[2];                             // circumf. velocity
   float  mod_angle[2];                           // inlet[0] and outlet [1] blade angle modification
   float  con_area[2];                            // inlet[0] and outlet [1] conduit area
   float  p_thick;                                // abs. profile thickness
   float  te_thick;                               // abs. trailing edge thickness
   float  camb;                                   // centre line camber
   float  camb_pos;                               // centre line camber position
   float  te_wrap;                                // trailing edge wrap angle
   float  bl_wrap;                                // blade element wrap angle
   float  bp_shift;                               // blade profile shift
   float  cl_len;                                 // centre line arc length
   struct profile *bp;                            // profile data
   struct Point *clg;                             // centre line gradients
   struct Point *cl;                              // centre line coordinates
   struct Point *ps;                              // pressure side coordinates
   struct Point *ss;                              // suction side coordinates
   struct Point *cl_cart;                         // centre line, cartesian coords
   struct Point *ps_cart;                         // pressure side, cartesian coords
   struct Point *ss_cart;                         // suctions side, cartesina coords
   float  pivot;                                  // rel. pivot location from le
   float  rad;                                    // abs. radius of blade element
   float  lec;                                    // le constriction
   float  tec;                                    // te constriction
};

struct meridian
{
   float  para;                                   // rel. conduit parameter [0..1]
   float  ngle[2];                                // inlet[0] and outlet[1] blade angle
   float  con_area[2];                            // inlet[0] and outlet [1] conduit area
   float  mer_vel[2];
   struct curve *ml;                              // meridional line coordinates
   struct Point *cl;                              // centre line coordinates
   struct Point *ps;                              // pressure side coordinates
   struct Point *ss;                              // suction side coordinates
   struct Flist *area;                            // conduit area
};

struct edge
{
   float  con[2];                                 // inner[0] and outer[1] constriction
   float  nocon;                                  // location of unconstricted blade element
};

struct design
{
   float  dis;                                    // nominal design discharge
   float  head;                                   // nominal design head
   float  revs;                                   // nominal design revolutions
   float  vratio;                                 // ratio, v_hub/v_shroud at inlet
   float  spec_revs;                              // specific revolutions
};

struct margin
{
   float  dr;                                     // radius difference
   struct Point *cl_ext;                          // centre line extension vector
   struct Point *cl_int;                          // centre line intersection
   struct Point *ps_ext;                          // pressure side extension vector
   struct Point *ps_int;                          // pressure side intersection
   struct Point *ss_ext;                          // suction side extension vector
   struct Point *ss_int;                          // suction side intersection
};

struct model
{
   int inl;                                       // inlet extension
   int bend;                                      // bend region
   int outl;                                      // outlet extension
   int arbitrary;                                 // arbitrary inlet region
};

struct axial
{
   int    clspline;                               // 1 or 2 splines for cl
   int    vratio_flag;                            // use given vel. ratio at inlet
   int    rot_clockwise;                          // clockwise rotation
   int    nob;                                    // number of blades
   float  nED;                                    // dimensionless rpm
   float  QED;                                    // dimensionless flow
   float  H;                                      // head
   float  Q;                                      // flow rate
   float  n;                                      // rpm
   float  D1;                                     // diameter, prototype
   float  alpha;                                  // ??? fuer Euler
   float  bangle;                                 // blade angle (operational point)

   float  enlace;                                 // blade enlacement
   float  piv;                                    // pivot location on chord
   float  ref;                                    // reference diameter = abs. outer runner diameter
   float  le_part[3];                             // leading edge polygon partition
   float  te_part[2];                             // trailing edge polygon partition
   float  diam[2];                                // hub[0] and shroud[1] runner diameter
   float  h_inl_ext;                              // inlet height
   float  d_inl_ext;                              // inlet extension diameter
   float  arb_angle;                              // inlet pitch angle, arbitrary inlet
   float  arb_part[2];                            // inlet spline partition ratios
   float  a_hub;                                  // semi-vertical axis hub ellipse
   float  b_hub;                                  // semi-horizontal axis hub ellipse
   float  h_run;                                  // runner height below inlet
   float  d_hub_sphere;                           // hub sphere diameter
   int    hub_sphere;                             // flag: hub sphere
   float  d_shroud_sphere;                        // shroud sphere diameter
   int    shroud_sphere;                          // flag: shroud sphere
   int    shroud_hemi;                            // flag: shroud hemisphere
   int    shroud_counter_rad;                     // flag: shroud sphere counter radius
   int    counter_nos;                            // number of counter radius sections
   int    hub_nos;                                // number of hub bend sections
   int    hub_bmodpoints;                         // modify hub bend points directly
   struct Point *p_hbpoints;                      // hub bend points (direct assignement)
   float  h_outl_ext;                             // outlet extension height
   float  d_outl_ext[2];                          // hub[0] and shroud[1] outlet extension diameter
   float  r_shroud[2];                            // radius shroud corner start[0] and end[1] arc
   float  ang_shroud;                             // angle shroud corner start arc
   int    cap_nop;                                // number of cap points
   struct Point *p_hubcap;                        // hub cap points
   float  h_draft;                                // draft tube height below runner
   float  d_draft;                                // draft tube inlet diameter
   float  ang_draft;                              // draft tube opening angle
   int    euler;                                  // use euler's turb. eqn. (1), no (0)
   struct model *mod;                             // flags for modelling options
   struct edge *le;                               // leading edge data
   struct edge *te;                               // trailing edge data
   struct design *des;                            // machine nominal design data
   struct parameter *iang;                        // inlet blade angle distribution
   struct parameter *mod_iang;                    // inlet angle modification distribution
   struct parameter *oang;                        // outlet blade angle distribution
   struct parameter *mod_oang;                    // outlet angle modification distribution
   struct parameter *t;                           // profile thickness distribution
   struct parameter *tet;                         // trailing edge thickness distribution
   struct parameter *camb;                        // centre line camber
   struct parameter *camb_pos;                    // centre line camber distribution
   struct parameter *bps;                         // blade profile shift distribution
   struct profile *bp;                            // blade profile data
   struct Point *p_hub;                           // hub staging points
   struct Point *p_shroud;                        // shroud staging points
   struct Point *p_hinlet;                        // inlet extension hub points
   struct Point *p_sinlet;                        // inlet extension shroud points
   struct Point *p_hbend;                         // bend region hub points
   struct Point *p_sbend;                         // bend region shroud points
   struct Point *p_hcore;                         // core region hub points
   struct Point *p_score;                         // core region shroud points
   struct Point *p_houtlet;                       // outlet extension hub points
   struct Point *p_soutlet;                       // outlet extension shroud points
   struct Flist *area;                            // conduit areas
   int    be_num;                                 // number of blade elements
   float  be_bias;                                // blade element bias
   int    be_type;                                // bias type: equidistant[0], one-way[1], two-way[2]
   int    extrapol;                               // allow parameter extrapolation[1] or not[0]
   int    be_single;                              // flag, whether a single be is modified or the entire blade
   int    parlock[NUM_PARLOCK];                   // lock parameter for quadratic evaluation
   struct margin *mhub;                           // hub margin intersection
   struct margin *mshroud;                        // shroud margin intersection
   struct be **be;                                // blade element data for runner construction (cylindrical)
   struct meridian **me;                          // meridinonal data for grid generation (meridional)
};

struct axial *AllocAxialRunner(void);
int CheckAR_EdgeConstriction(struct axial *ar);
struct Flist *CalcBladeElementBias(int nodes, float t1, float t2, int type, float ratio);
int ReadAxialRunner(struct geometry *g, const char *fn);
int WriteAxialRunner(struct axial *ar, FILE *fp);
void PlotAR_BladeEdges(struct axial *ar);
void WriteGNU_AR(struct axial *ar);
void DumpAR(struct axial *ar);
#endif                                            // AXIAL_H_INCLUDED
