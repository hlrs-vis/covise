#ifndef RADIAL_H_INCLUDED
#define RADIAL_H_INCLUDED

#include "../../General/include/geo.h"
#include "../../General/include/curve.h"

#define NPOIN_EDGE   20                           // number of points blade edge curves
#define NPOIN_MERIDIAN  50                        // number of points casing curves
#define NPOIN_EXT 10                              // number of points casing extension curves

#define NPOIN_SPLINE 10                           // number of points to define splines (meridian->sphere)
#define NPOIN_SPHERE (NPOIN_MERIDIAN - 2*(NPOIN_SPLINE+1))  // number of points to define sphere

#define CONF_ERR 1
#define DESIGN_DATA_ERR 2
#define EULER_ERR 3
#define POLYGON_ERROR 4
#define PUT_BLADEDATA_ERR 5
#define BLMER_INTERSECT_ERR 6

#define CAMB_FUNC_TEFIX 2
#define CAMB_FUNC_LEFIX 3
#define CAMB_FUNC_LEN   4

struct edge {
    float para[2];                                 // start[0] and end[1] meridian parameter
    float angle[2];                                // start[0] and end[1] off-contour angle
    float h_norm[3];                               // off-contour tangent at hub
    float s_norm[3];                               // off-contour tangent at shroud
    float   spara[2];                              // spline point params.
    struct curve *c;                               // edge curve data (meridional plane)
    struct Point *bmint;                           // blade edge-meridian intersection points
    struct Flist *bmpar;                           // parameter value of intersection on meridian curve
};

struct be {                                        // in meridional planes {
    float para;                                    // meridian parameter value
    float le_para;                                 // spline params, le
    float te_para;                                 // spline params, te
    float angle[2];                                // inlet[0] and outlet[1] blade angle
    float mod_angle[2];                            // inlet[0]/outlet[1] angle modification
    float con_area[2];                             // conduit areas at inlet[0] and outlet[1]
    float mer_vel[2];                              // meridional velocities
    float cir_vel[2];                              // cicumferential velocities
    float rot_abs[2];                              // rotational part of absolute vel.
    float p_thick;                                 // abs. profile thickness
    float te_thick;                                // abs. trailing edge thickness
    float camb;                                    // max.camber of centre line
    float camb_pos;                                // max. camber position, relative on chord
    float camb_para;                               // parameter for camber function
    float bl_lenpara;                              // blade length (para*2*pi/nob*D1)
    float te_wrap;                                 // trailing edge wrap angle, rel. to shroud te
    float bl_wrap;                                 // blade wrap angle
    float bp_shift;                                // blade profile shift
    float cl_len;                                  // centre line arc length (from cl_cart)
    struct profile *bp;                            // blade profile
    struct curve *ml;                              // meridional line data (r,0.0,z)
    struct Point *clg;                             // centre line gradients
    struct Point *cl;                              // centre line coordinates (r,phi,z)
    struct Point *ps;                              // pressure side coordinates
    struct Point *ss;                              // suction side coordinates
    struct Point *cl_cart;                         // centre line, cartesian coords
    struct Point *ps_cart;                         // pressure side, cartesian coords
    struct Point *ss_cart;                         // suction side, cartesian coords
    struct Flist *area;                            // conduit areas
};

struct design {
    float dis;                                     // nominal design discharge
    float head;                                    // nominal design head
    float revs;                                    // nominal design revolutions
    float vratio;                                  // ratio, v_hub/v_shroud at inlet
    float spec_revs;                               // specific revolutions
};

struct radial {
    int     camb2surf;            // create blade from given camber (0/1)
    int     be_single;                             // only single modified.
    int     euler;                                 // use eulerian turbine eqn. (1), no (0)
    int     db2merid;                              // create contour from database
    int     rot_clockwise;                         // rotate clockwise
    int     straight_cont[2];                      // straight contour
    int     camb_flag;                             // force max. camber point (1), no (0)
    int     vratio_flag;                           // use defined velocity profile at le
    int     showExt;                               // compute polygons for extended geometry (1/0)
    int     nob;                                   // number of blades
	int     diagonal;                              // diagonal runner with sphere contour
	int     pump;                                  // make a pump
    float   ref;                                   // reference dimension = abs. shroud outlet diameter
    float   diam[2];                               // shroud inlet[0] and outlet[1] diameters
    float   ext_diam[2];                           // extension shroud inlet/outlet (0,1) diam.
    float   height;                                // shroud inlet/outlet height difference
    float   ext_height[2];                         // extension height difference (inlet, outlet)
    float   cond[2];                               // inlet[0] and outlet[1] conduit width
    float   ext_cond[2];                           // extension conduit width (in, out)
    float   angle[2];                              // inlet[0] and outlet[1] contour angle
    float   ext_iangle;                            // inlet contour angle, extension
    float   iop_angle[2];                          // inlet hub[0] and shroud[1] opening angles
    float   oop_angle[2];                          // outlet hub[0] and shroud[1] opening angles
    float   hspara[2];                             // spline params., hub
    float   sspara[2];                             // spline params., shroud
    float   sstparam[2];                           // shroud, straight lines, parametre
    float   hstparam[2];                           // hub, straight lines, parametre
    float   hspara_inext[2];                       // spline params., hub, inlet extension
    float   sspara_inext[2];                       // spline params., shroud, inlet extension
	float   sphdiam[2];                            // rel. sphere inlet/outlet (0/1) diam.
	float   sphcond;                               // rel. sphere conduit width (const.!)
	float   spheight;                              // rel. sphere height diff. (shroud)
	float   ospheight;                             // rel. sph. outlet height coord. (shroud)
	float   stpara[2];                             // stretch parameter for hub sphere section
    struct edge *le;                               // leading edge data
    struct edge *te;                               // trailing edge data
    struct design *des;                            // machine nominal design data
    struct parameter *cambpara;                    // camber pos. rel. on chord
    struct parameter *bl_lenpara;                  // blade length param
    struct parameter *camb_pos;                    // camber pos. rel. on chord
    struct parameter *iang;                        // inlet angle distribution
    struct parameter *mod_iang;                    // inlet angle modification distribution
    struct parameter *oang;                        // outlet angle distribution
    struct parameter *mod_oang;                    // outlet angle mod. distrib.
    struct parameter *orot_abs;                    // remaining rotational vel., outlet
    struct parameter *t;                           // profile thickness dirtibution
    struct parameter *tet;                         // trailing edge thickness distribution
    struct parameter *camb;                        // centre line camber distribution
    struct parameter *tewr;                        // trailing edge wrap angle distribution
    struct parameter *blwr;                        // blade wrap angle distribution
    struct parameter *bps;                         // blade profile shift distribution
    struct parameter *le_para;                     // blade spline paras, le
    struct parameter *te_para;                     // blade spline paras, te
    struct profile *bp;                            // blade profile data
    int      be_num;                               // number of blade elements
    float    be_bias;                              // blade element bias
    int      be_type;                              // bias type: equidistant[0], one-way[1], two-way[2]
    int      extrapol;                             // allow parameter extrapolation[1] or not[0]
    struct be **be;                                // blade element data
#ifdef GAP
    float gap;                                     // relative gap width
    struct be *gp;                                 // shroud contour
#endif
};

// in rr_io.c
struct radial *AllocRadialRunner(void);
int ReadRadialRunner(struct geometry *g, const char *fn);
int WriteRadialRunner(struct radial *rr, FILE *fp);
void DumpRR(struct radial *rr);
void WriteGNU_RR(struct radial *rr);
int WriteRadialRunner(struct radial *rr, FILE *fp);

int CreateRR_BladeElements(struct radial *rr);
int CalcRR_BladeAngles(struct radial *rr);
#endif                                            // RADIAL_H_INCLUDED
