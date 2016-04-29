#ifndef  TUBE_INCLUDED
#define  TUBE_INCLUDED

#include "../../General/include/geo.h"

struct pe
{
   int p_start_cs;
   int p_end_cs;
   int p_type_dist;
   float p_start_dist;
   float p_end_dist;
   float p_nose_length;
   float p_nose_rad;

   struct Point *p;
};

struct cs
{
   // the next parameters we get all from the startup file
   // d_ means default ...
   float d_m_x;
   float d_m_y;
   float d_m_z;
   float d_width;
   float d_height;
   float d_a[4];
   float d_b[4];
   int   d_angletype;
   float d_angle;
   float d_part[8];
   float d_linfact;
   int   d_nume;

   // here starts the section, were we store all actual values,
   // c_ means changed ...
   float c_m_x;
   float c_m_y;
   float c_m_z;
   float c_width;
   float c_height;
   float c_a[4];
   float c_b[4];
   int   c_angletype;
   float c_angle;
   float c_part[8];
   float c_linfact;
   int   c_nume;

   // these are the 8 points, which divide the 8 edges of every cross-section
   int cov_ind[8];                                // index for x,y,z

   float T[3][3];
   struct Point *p;
};

struct tube
{
   int cs_num;
   int cs_max;
   struct cs **cs;

   int pe_num;
   int pe_max;
   int pe_orient;
   struct pe **pe;

   int d_el[4];
   int c_el[4];
   int d_el_o;
   int c_el_o;
};

static const char *sectornames[] = { "E", "NE", "N", "NW", "W", "SW", "S", "SE" };

// type of angle
#define  T_ANGLE_ABSOLUTE  1
#define  T_ANGLE_RELATIV      2

// index for cs[x]->d_a[...]
#define  T_RT_AB  0
#define  T_LT_AB  1
#define  T_LB_AB  2
#define  T_RB_AB  3

// Return values for all interpolation functions
#define  TIP_OK            0
#define  TIP_CD_PUT_OF_RANGE  10

// Peer
#define  PE_VERTICAL    1
#define  PE_HORIZONTAL  2
#define  PE_TYPE_DIST_LE         0
#define  PE_TYPE_DIST_PERCENT 1

extern  void FreeT_CS(struct cs *cs);
extern  void FreeTube(struct tube *tu);
extern  void AllocT_CS(struct tube *t);
extern  struct tube *AllocTube(void);
extern  struct tube* ReadTube(const char *fn);
extern  void AllocT_PE(struct tube *t);
extern  void FreeT_PE(struct pe *pe);
extern  int WriteTube2File(struct tube *t, char *fn);
extern  int WriteTube(struct tube *tu, FILE *fp);
extern  struct covise_info *Tube2Covise(struct tube *tu);
extern  int CalcCSGeometry(struct tube *t, int csi, float n[3]);
extern  int CalcTubeGeometry(struct tube *tu);
extern  int t_lin_ip(struct tube *t, int s_cs, int e_cs);
extern  float CalcOneCSArea(struct cs *cs);
extern  void DumpTube(struct tube *t);
extern  void DumpT_CS(struct cs *cs, char *fname);
#ifdef   DEBUG
void Tube2CoviseDump(struct covise_info *ci);
#endif                                            // DEBUG
#endif                                            // TUBE_INCLUDED
