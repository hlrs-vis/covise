#ifndef GATE_H_INCLUDED
#define GATE_H_INCLUDED

#include <General/include/geo.h>

#define NPOIN_SHROUD_AB  10

#define BLADE  1
#define HUB    2
#define SHROUD 3

struct gate
{
   int      geofromfile;                          // read blade and meridian contour from files
   int   radial;                                  // axial (=0) or radial (=1) geometry
   char  cfgfile[255];                            // path and name of the cfg-file
   float Q;                                       // flow rate [m3/s], calculated automatically
   float H;                                       // head [m]
   float n;                                       // rotation speed [1/s]
   float Qopt;                                    // head in design-(best-)point [m]
   float nopt;                                    // rotation speed in design-(best-)point
   float beta;                                    // angle at blade exit
   float qmax;                                    // maximum flow rate (if blade rotated), filled in automatically
   float a0_max;                                  // maximum ratio of a0 / a0opt
   int      nob;                                  // number of blades
   float bangle;                                  // blade angle (open & close)
   float pivot_rad;                               // radius of blade axis
   float chord;                                   // chord length
   float pivot;                                   // location of pivot on chord [m] from TE
   float angle;                                   // chord angle
   float    p_thick;                              // abs. profile thickness
   float maxcamb;                                 // max. abs. centre line camber
   float bp_shift;                                // blade profile shift
   struct   parameter *camb;                      // camber distribution data
   struct   parameter *prof;                      // blade profile data
   struct   profile *bp;                          // profile data
   struct   Point *cl;                            // centre line coordinates
   struct   Point *clg;                           // centre line gradients
   struct   Point *ps;                            // pressure side coordinates
   struct   Point *ss;                            // suction side coordinates
   struct   Point *phub;                          // runner hub points
   struct   Point *pshroud;                       // runner shroud points
   struct   Point *phub_n;                        // runner hub points for normals
   struct   Point *p_pivot;                       // pivot coordinates (only if geofromfile)
   struct   curve *chub;                          // gate hub contour, spline
   struct   curve *cshroud;                       // gate shroud contour, spline
   float in_height;                               // height of inlet (z hub - z_shroud)
   float in_rad;                                  // radius of inlet
   float in_z;                                    // z of inlet
   float    out_rad1;                             // outlet inner diameter
   float out_rad2;                                // outlet outer diameter
   float out_z;                                   // outlet z
   float shroud_ab[2];                            // shroud ellipse radius (a: horizontal [0], b: vertical [1])
   float hub_ab[2];                               // hub ellipse radius (a: horizontal [0], b: vertical [1])
   int      num_hub_arc;                          // number of points at hub arc (quarter ellipse)
   float a0_beta[20];                             // relationship between beta and a0/a0opt
   float beta_min;                                // angle for closing position (Q=0)
   float beta_max;                                // angle for 100% open position (Q=qmax)
   int   close;                                   // does it close? (1=yes, 0=no)
   struct   ggrid_para *gr;                       // pointer to grid data

};

struct ggrid_para
{

   // grid parameters

   int      savegrid;                             // generate geo-file and rb-file?

   // border position, boundary layer thickness
   int      edge_ps;
   int      edge_ss;
   float bound_layer;

   // number of points
   int      n_rad;
   int     n_bound;
   int      n_out;
   int      n_in;
   int      n_blade_ps_back;
   int      n_blade_ps_front;
   int      n_blade_ss_back;
   int      n_blade_ss_front;

   // lengths
   int      len_start_out_hub;                    // % between trailing edge and outlet
   int      len_start_out_shroud;                 // % between trailing edge and outlet
   float len_expand_in;
   float len_expand_out;

   // compressions
   float comp_ps_back;
   float comp_ps_front;
   float comp_ss_back;
   float comp_ss_front;
   float comp_trail;
   float comp_out;
   float comp_in;
   float comp_bound;
   float comp_middle;
   float comp_rad;

   // shifts
   float shift_out;
};

/*
struct margin{
   float	dr;				// radius difference
   struct Point *ps_ext;	// pressure side extension vector
   struct Point *ps_int;	// pressure side intersection
   struct Point *ss_ext;	// suction side extension vector
   struct Point *ss_int;	// suction side intersection
};
*/

// in ga_io.c
struct gate *AllocGate(void);
int ReadGate(struct geometry *g, const char *fn);
int WriteGate(struct gate *ga, FILE *fp);
void DumpGA(struct gate *ga);
int ReadPointStruct(struct Point *p, const char *sec, const char *fn);
//void WriteGNU_GA(struct gate *ga);

// in ga_comp.c
int InitGA_BladeElements(struct gate *ga);
//int ModifyGA_BladeElements4Covise(struct gate *ga);
//void DetermineCoefficients(float *x, float *y, float *a);
//float EvaluateParameter(float x, float *a);
int CreateGA_Contours(struct gate *ga);
void CreateGAContourWireframe(struct curve *c);
int CreateGA_BladeElements(struct gate *ga);
int SurfacesGA_BladeElement(struct gate *ga);
int ReadProfileFromFile(struct gate *ga, const char *fn);
int CreateShell(struct gate *ga, int n_circles, int steps, float *xpl, float *ypl);
int CreateIsoAngleLines(struct gate *ga, int position, int n_isolines, float xwmin, float xwmax, float *xpl, float *ypl);
int RotateBlade(struct gate *ga, float angle, float x_piv, float y_piv);
float get_a0(struct gate *ga, float sign);
int cubic_equation(float a, float b, float c, float d, int *num, float res[3]);
#endif                                            // GATE_H_INCLUDED
