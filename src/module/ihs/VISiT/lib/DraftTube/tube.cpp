#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif
#include <math.h>
#include "include/tube.h"
#include "../General/include/cfg.h"
#include "../General/include/geo.h"
#include "../General/include/points.h"
#include "../General/include/common.h"
#include "../General/include/log.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"

#define  G_PEER            "[peer]"
#define  G_PEER_ORIENT     "orientation"
#define  T_PEER            "[peer_%d]"
#define  P_START_SEC       "start cross section"
#define  P_END_SEC         "end cross section"
#define  P_TYPE_DIST       "type of distance"
#define  P_START_DIST      "distance at start cross section"
#define  P_END_DIST        "distance at end cross section"
#define  P_NOSE_LENGTH     "length of peer nose"
#define  P_NOSE_RAD        "radius of peer nose"
#define  G_PEER_VERTICAL      "vertical"
#define  G_PEER_HORIZONTAL "horizontal"
#define  G_SECTION         "[grid parameters]"
#define  G_ELEMS_LV        "num elems left vertical"
#define  G_ELEMS_RV        "num elems right vertical"
#define  G_ELEMS_TH        "num elems top horicontal"
#define  G_ELEMS_BH        "num elems bottom horicontal"
#define  G_ELEMS_O         "num rows in outer sections"
#define  G_ELEMS           "num elems"
#define  G_LINFACT         "linear factor"
#define  G_PART            "part %s"
#define  T_SECTION         "[c_section_%d]"
#define  T_MIDDLE       "middle point"
#define  T_HEIGHT       "height"
#define  T_WIDTH           "width"
#define  T_ANGLETYPE       "type angle"
#define  T_ANGLE           "angle"

#define  PE_MAX_NUM  2

static void CalcEllipsePoints(struct cs *cs, float a, float b, float dx, float dy, float as, float ae);
static void CalcTrafo(struct tube *tu, int ics, float n[3]);

static const char *angletypes[] =
{
   NULL,
   "absolute",
   "relativ",
};

static const char *abtypes[] =
{
   "RightTop ab",
   "LeftTop ab",
   "LeftBottom ab",
   "RightBottom ab",
};

int WriteTube2File(struct tube *t, char *fn)
{
   FILE *fp;

   if ((fp = fopen(fn, "w")) != NULL)
   {
      WriteTube(t, fp);
      fclose(fp);
      return 1;
   }
   return 0;
}


#define  P_LEN 35
int WriteTube(struct tube *tu, FILE *fp)
{
   int i, j;
   char section[100];
   char buf[100];

   // Peer
   fprintf(fp, "\n%s", G_PEER);
   fprintf(fp, "%*s = %s\n", P_LEN, G_PEER_ORIENT,
      (tu->pe_orient == PE_VERTICAL ? G_PEER_VERTICAL : G_PEER_HORIZONTAL));
   for (i = 0; i < tu->pe_num; i++)
   {
      sprintf(section, T_PEER, i);
      fprintf(fp, "\n%s\n", section);
      fprintf(fp, "%*s = %d\n", P_LEN, P_START_SEC, tu->pe[i]->p_start_cs);
      fprintf(fp, "%*s = %d\n", P_LEN, P_END_SEC, tu->pe[i]->p_end_cs);
      fprintf(fp, "%*s = %s\n", P_LEN, P_TYPE_DIST,
         (tu->pe[i]->p_type_dist == PE_TYPE_DIST_PERCENT ? "%" : "LE"));
      fprintf(fp, "%*s = %f\n", P_LEN, P_START_DIST, tu->pe[i]->p_start_dist);
      fprintf(fp, "%*s = %f\n", P_LEN, P_END_DIST, tu->pe[i]->p_end_dist);
      fprintf(fp, "%*s = %f\n", P_LEN, P_NOSE_LENGTH, tu->pe[i]->p_nose_length);
      fprintf(fp, "%*s = %f\n", P_LEN, P_NOSE_RAD, tu->pe[i]->p_nose_rad);
   }
   //
   fprintf(fp, "\n%s\n", G_SECTION);
   fprintf(fp, "%*s = %d\n", P_LEN, G_ELEMS_RV, tu->c_el[0]);
   fprintf(fp, "%*s = %d\n", P_LEN, G_ELEMS_TH, tu->c_el[1]);
   fprintf(fp, "%*s = %d\n", P_LEN, G_ELEMS_LV, tu->c_el[2]);
   fprintf(fp, "%*s = %d\n", P_LEN, G_ELEMS_BH, tu->c_el[3]);
   fprintf(fp, "%*s = %d\n", P_LEN, G_ELEMS_O,  tu->c_el_o);
   for (i = 0; i < tu->cs_num; i++)
   {
      sprintf(section, T_SECTION, i);
      fprintf(fp, "\n%s\n", section);
      fprintf(fp, "%*s = %f, %f, %f\n", P_LEN, T_MIDDLE, tu->cs[i]->c_m_x, tu->cs[i]->c_m_y, tu->cs[i]->c_m_z);
      fprintf(fp, "%*s = %f\n", P_LEN, T_HEIGHT,    tu->cs[i]->c_height);
      fprintf(fp, "%*s = %f\n", P_LEN, T_WIDTH ,    tu->cs[i]->c_width);
      fprintf(fp, "%*s = %s\n", P_LEN, T_ANGLETYPE, angletypes[tu->cs[i]->c_angletype]);
      fprintf(fp, "%*s = %f\n", P_LEN, T_ANGLE,     tu->cs[i]->c_angle);
      fprintf(fp, "%*s = %d\n", P_LEN, G_ELEMS,     tu->cs[i]->c_nume);
      for (j = 0; j < 8; j++)
      {
         sprintf(buf, G_PART, sectornames[j]);
         fprintf(fp, "%*s = %f\n", P_LEN, buf, tu->cs[i]->c_part[j]);
      }
      fprintf(fp, "%*s = %f\n", P_LEN, G_LINFACT, tu->cs[i]->c_linfact);
      for (j = 0; j < 4; j++)
      {
         fprintf(fp, "%*s = %f, %f\n", P_LEN, abtypes[j], tu->cs[i]->c_a[j], tu->cs[i]->c_b[j]);
      }
   }
   return 1;
}


struct tube *ReadTube(const char *fn)
{
   int i, j;
   char *tmp;
   char **dp;
   char buf[200];
   char sector[100];
   float x, y, z, height, width;
   int num_elems;
   float part;
   struct tube *tu;

   dprintf(1, "Reading tube configuration");
   tu = (struct tube *)AllocTube();

   dp = dump_cfg();
   for (i = 0; dp[i] && *dp[i]; i++)
   {
      dprintf(5, "%3d: #%s#\n", i, dp[i]);
      free(dp[i]);
   }
   if (dp)  free(dp);

   // peer
   strcpy(buf, G_PEER);
   tu->pe_orient = -1;
   if ((tmp = IHS_GetCFGValue(fn, buf, G_PEER_ORIENT)) != NULL)
   {
      if (!strcmp(tmp, G_PEER_VERTICAL))
         tu->pe_orient = PE_VERTICAL;
      else if (!strcmp(tmp, G_PEER_HORIZONTAL))
         tu->pe_orient = PE_HORIZONTAL;
      free(tmp);
   }
   for (i = 0; ; i++)
   {
      int start_cs, end_cs;

      sprintf(buf, T_PEER, i);
      dprintf(3, "Scanning for %s in %s", buf, fn);

      if ((tmp = IHS_GetCFGValue(fn, buf, P_START_SEC)) != NULL)
      {
         int num;
         sscanf(tmp, "%d", &num);
         start_cs = num;
         free(tmp);
      }
      else
         start_cs = 0;
      if ((tmp = IHS_GetCFGValue(fn, buf, P_END_SEC)) != NULL)
      {
         int num;
         sscanf(tmp, "%d", &num);
         end_cs = num;
         free(tmp);
      }
      else
         end_cs = 0;
      if (start_cs == end_cs || end_cs == 0 || start_cs > end_cs)
      {
         dprintf(2, "Einlesen peer abgebrochen (i=%d)", i);
         break;
      }
      AllocT_PE(tu);
      tu->pe[i]->p_start_cs = start_cs;
      tu->pe[i]->p_end_cs = end_cs;

      if ((tmp = IHS_GetCFGValue(fn, buf, P_TYPE_DIST)) != NULL)
      {
         int num;
         sscanf(tmp, "%d", &num);
         tu->pe[i]->p_type_dist = num;
         free(tmp);
      }
      if ((tmp = IHS_GetCFGValue(fn, buf, P_START_DIST)) != NULL)
      {
         float num;
         sscanf(tmp, "%f", &num);
         tu->pe[i]->p_start_dist = num;
         free(tmp);
      }
      if ((tmp = IHS_GetCFGValue(fn, buf, P_END_DIST)) != NULL)
      {
         float num;
         sscanf(tmp, "%f", &num);
         tu->pe[i]->p_end_dist = num;
         free(tmp);
      }
      if ((tmp = IHS_GetCFGValue(fn, buf, P_NOSE_LENGTH)) != NULL)
      {
         float num;
         sscanf(tmp, "%f", &num);
         tu->pe[i]->p_nose_length = num;
         free(tmp);
      }
      if ((tmp = IHS_GetCFGValue(fn, buf, P_NOSE_RAD)) != NULL)
      {
         float num;
         sscanf(tmp, "%f", &num);
         tu->pe[i]->p_nose_rad = num;
         free(tmp);
      }
   }

   strcpy(buf, G_SECTION);
   for (i = 0; i < 4; i++)
      tu->d_el[i] = 1;
   tu->d_el_o = 1;
   if ((tmp = IHS_GetCFGValue(fn, buf, G_ELEMS_RV)) != NULL)
   {
      sscanf(tmp, "%d", &num_elems);
      if (num_elems > 0)
         tu->d_el[0] = num_elems;
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, buf, G_ELEMS_TH)) != NULL)
   {
      sscanf(tmp, "%d", &num_elems);
      if (num_elems > 0)
         tu->d_el[1] = num_elems;
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, buf, G_ELEMS_LV)) != NULL)
   {
      sscanf(tmp, "%d", &num_elems);
      if (num_elems > 0)
         tu->d_el[2] = num_elems;
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, buf, G_ELEMS_BH)) != NULL)
   {
      sscanf(tmp, "%d", &num_elems);
      if (num_elems > 0)
         tu->d_el[3] = num_elems;
      free(tmp);
   }
   if ((tmp = IHS_GetCFGValue(fn, buf, G_ELEMS_O)) != NULL)
   {
      sscanf(tmp, "%d", &num_elems);
      if (num_elems > 0)
         tu->d_el_o = num_elems;
      free(tmp);
   }

   for (i = 0; ; i++)
   {
      x = y = z = height = width = 0.0;
      sprintf(buf, T_SECTION, i);
      dprintf(2, "Scanning for %s in %s", buf, fn);

      if ((tmp = IHS_GetCFGValue(fn, buf, T_MIDDLE)) != NULL)
      {
         sscanf(tmp, "%f,%f,%f", &x, &y, &z);
         free(tmp);

         num_elems = 1;
         AllocT_CS(tu);
         for (j = 0; j < 8; j++)
         {
            tu->cs[i]->d_part[j] = 0.8f;
            sprintf(sector, G_PART, sectornames[j]);
            if ((tmp = IHS_GetCFGValue(fn, buf, sector)) != NULL)
            {
               sscanf(tmp, "%f", &part);
               if (part > 0.0 && part < 1.0)
                  tu->cs[i]->d_part[j] = part;
               free(tmp);
            }
            else
            {
               dprintf(5, "Error during reading: %d, %s, %s, using default value\n", j, sectornames[j], sector);
            }
         }
         tu->cs[i]->d_linfact = 1.0;
         if ((tmp = IHS_GetCFGValue(fn, buf, G_LINFACT)) != NULL)
         {
            sscanf(tmp, "%f", &part);
            if (part > -50.0 && part < 50.0)
               tu->cs[i]->d_linfact = part;
            free(tmp);
         }
         if ((tmp = IHS_GetCFGValue(fn, buf, G_ELEMS)) != NULL)
         {
            sscanf(tmp, "%d", &num_elems);
            free(tmp);
         }
         if ((tmp = IHS_GetCFGValue(fn, buf, T_HEIGHT)) != NULL)
         {
            sscanf(tmp, "%f", &height);
            free(tmp);
         }
         if ((tmp = IHS_GetCFGValue(fn, buf, T_WIDTH)) != NULL)
         {
            sscanf(tmp, "%f", &width);
            free(tmp);
         }
         if ((tmp = IHS_GetCFGValue(fn, buf, T_ANGLE)) != NULL)
         {
            sscanf(tmp, "%f", (float *)&(tu->cs[tu->cs_num-1]->d_angle));
            free(tmp);
            if ((tmp = IHS_GetCFGValue(fn, buf, T_ANGLETYPE)) != NULL)
            {
               if (!strcmp(angletypes[T_ANGLE_ABSOLUTE], tmp))
                  tu->cs[tu->cs_num-1]->d_angletype = T_ANGLE_ABSOLUTE;
               else if (!strcmp(angletypes[T_ANGLE_RELATIV], tmp))
                  tu->cs[tu->cs_num-1]->d_angletype = T_ANGLE_RELATIV;
               else
                  tu->cs[tu->cs_num-1]->d_angletype = 0;
               free(tmp);
            }
            else
            {
               tu->cs[tu->cs_num-1]->d_angletype = T_ANGLE_RELATIV;
            }
         }
         for (j = 0; j < 4; j++)
         {
            float a, b;
            if ((tmp = IHS_GetCFGValue(fn, buf, abtypes[j])) != NULL)
            {
               if (2 != sscanf(tmp, "%f,%f", &a, &b))
               {
                  tu->cs[tu->cs_num-1]->d_a[j] = 0.0;
                  tu->cs[tu->cs_num-1]->d_b[j] = 0.0;
               }
               else
               {
                  tu->cs[tu->cs_num-1]->d_a[j] = a;
                  tu->cs[tu->cs_num-1]->d_b[j] = b;
               }
               free(tmp);
            }
         }
         tu->cs[tu->cs_num-1]->d_m_x = x;
         tu->cs[tu->cs_num-1]->d_m_y = y;
         tu->cs[tu->cs_num-1]->d_m_z = z;
         tu->cs[tu->cs_num-1]->d_width = width;
         tu->cs[tu->cs_num-1]->d_height = height;
         tu->cs[tu->cs_num-1]->d_nume = num_elems;

         // for the first time, we have to set the c_ section ...
         tu->cs[tu->cs_num-1]->c_m_x = tu->cs[tu->cs_num-1]->d_m_x;
         tu->cs[tu->cs_num-1]->c_m_y = tu->cs[tu->cs_num-1]->d_m_y;
         tu->cs[tu->cs_num-1]->c_m_z = tu->cs[tu->cs_num-1]->d_m_z;
         tu->cs[tu->cs_num-1]->c_width = tu->cs[tu->cs_num-1]->d_width;
         tu->cs[tu->cs_num-1]->c_height = tu->cs[tu->cs_num-1]->d_height;
         for (j = 0; j < 4; j++)
         {
            tu->cs[tu->cs_num-1]->c_a[j] = tu->cs[tu->cs_num-1]->d_a[j];
            tu->cs[tu->cs_num-1]->c_b[j] = tu->cs[tu->cs_num-1]->d_b[j];
         }
         tu->cs[tu->cs_num-1]->c_angletype = tu->cs[tu->cs_num-1]->d_angletype;
         tu->cs[tu->cs_num-1]->c_angle = tu->cs[tu->cs_num-1]->d_angle;
         // and now grid parameters ...
         tu->cs[tu->cs_num-1]->c_nume = tu->cs[tu->cs_num-1]->d_nume;
         for (j = 0; j < 8; j++)
            tu->cs[tu->cs_num-1]->c_part[j] = tu->cs[tu->cs_num-1]->d_part[j];
         tu->cs[tu->cs_num-1]->c_linfact = tu->cs[tu->cs_num-1]->d_linfact;
      }
      else
      {
         dprintf(1, "End reading tube configuration");
         break;
      }
   }
   tu->c_el_o  = tu->d_el_o;
   for (i = 0; i < 4; i++)
      tu->c_el[i] = tu->d_el[i];
   DumpTube(tu);

   return tu;
}


void FreeTube(struct tube *tu)
{
   int i;

   if (tu)
   {
      for (i = 0; i < tu->cs_num; i++)
         FreeT_CS(tu->cs[i]);
      free(tu->cs);
      for (i = 0; i < tu->pe_num; i++)
         FreeT_PE(tu->pe[i]);
      free(tu->pe);
      free(tu);
   }
}


struct tube *AllocTube(void)
{
   struct tube *t;

   if ((t = (struct tube *)calloc(1, sizeof(struct tube))) != NULL)
   {
      t->cs_max = 10;
      if ((t->cs = (struct cs **)calloc(t->cs_max, sizeof(struct cs *))) == NULL)
         fatal("Not enough space");
      t->pe_max = PE_MAX_NUM;
      if ((t->pe = (struct pe **)calloc(t->pe_max, sizeof(struct pe *))) == NULL)
         fatal("Not enough space");
   }
   else
   {
      fatal("Not enough space ...");
   }

   return t;
}


void AllocT_PE(struct tube *t)
{
   int nnum;

   nnum = t->pe_num + 1;

   if (t->pe_num >= t->pe_max)
      fatal("Maximum are 2 peers !!");
   if ((t->pe[nnum-1] = (struct pe *)calloc(1, sizeof(struct pe))) == NULL)
      fatal("Space");;
   t->pe[nnum-1]->p = AllocPointStruct();
   t->pe_num++;
}


void FreeT_PE(struct pe *pe)
{
   if (pe)
   {
      FreePointStruct(pe->p);
      free(pe);
   }
}


void FreeT_CS(struct cs *cs)
{
   if (cs)
   {
      FreePointStruct(cs->p);
      free(cs);
   }
}


void AllocT_CS(struct tube *t)
{
   int nnum;

   nnum = t->cs_num + 1;

   if (t->cs_num >= t->cs_max)
   {
      t->cs_max += 10;
      if ((t->cs = (struct cs **)realloc(t->cs,
         t->cs_max*sizeof(struct cs*))) == NULL)
         fatal("Space");
   }
   if ((t->cs[nnum-1] = (struct cs *)calloc(1, sizeof(struct cs))) == NULL)
      fatal("Space");;
   t->cs[nnum-1]->p = AllocPointStruct();
   t->cs_num++;
}


static void CalcTrafo(struct tube *tu, int ics, float n[3])
{
   float m[3], lm[3], nm[3];
   float erz[3], ez[3];
   float lv[3], nv[3], lnv[3];
   float rot, srot, gra;

   ez[0] = 0.0;
   ez[1] = 0.0;
   ez[2] = 1.0;

   // vector to the last CS
   if (ics)
   {
      lm[0] = tu->cs[ics-1]->c_m_x;
      lm[1] = tu->cs[ics-1]->c_m_y;
      lm[2] = tu->cs[ics-1]->c_m_z;
   }
   else                                           // special case: the first CS
   {
      lm[0] = tu->cs[ics]->c_m_x;
      lm[1] = tu->cs[ics]->c_m_y;
      lm[2] = tu->cs[ics]->c_m_z;
   }
   // vector to the actual CS
   m[0] = tu->cs[ics]->c_m_x;
   m[1] = tu->cs[ics]->c_m_y;
   m[2] = tu->cs[ics]->c_m_z;
   // vector to the following CS
   if (ics < tu->cs_num-1 || n)
   {
      if (n)                                      // we need that bullshit for VR
      {
         dprintf(5, "CalcTrafo(): USING n-vector !!!\n");
         nm[0] = n[0];                            //tu->cs[ics+1]->c_m_x;
         nm[1] = n[1];                            //tu->cs[ics+1]->c_m_y;
         nm[2] = n[2];                            //tu->cs[ics+1]->c_m_z;
      }
      else
      {
         dprintf(5, "CalcTrafo(): !!!! DONT using n-vector !!!\n");
         nm[0] = tu->cs[ics+1]->c_m_x;
         nm[1] = tu->cs[ics+1]->c_m_y;
         nm[2] = tu->cs[ics+1]->c_m_z;
      }
   }
   else                                           // special case: the last CS
   {
      nm[0] = tu->cs[ics]->c_m_x;
      nm[1] = tu->cs[ics]->c_m_y;
      nm[2] = tu->cs[ics]->c_m_z;
   }
   // vector from the last CS to the actual
   V_Sub(m,  lm,  lv);
   // vector from the actual CS to the next
   V_Sub(nm, m, nv);
   // vector from the last CS to the next CS
   V_Add(lv, nv, lnv);
   // this orientation is our Z-direction !!

   // vector erz
   V_Copy(erz, lnv);
   V_Norm(erz);
   V_MultScal(erz, -1);

   srot = 1.0;
   if (tu->cs[ics]->d_angletype == T_ANGLE_ABSOLUTE)
   {
      gra = tu->cs[ics]->d_angle;
      if (tu->cs[ics]->d_angle >= 180.0f)
      {
         gra = tu->cs[ics]->d_angle -180.0f;
         srot = -1.0;
      }
      rot = float(cos(gra/180.0*M_PI));
   }
   else if (tu->cs[ics]->d_angletype == T_ANGLE_RELATIV)
      rot = float(V_Angle(erz, ez) + cos(tu->cs[ics]->d_angle/180.0*M_PI));
   // REI : NICHT getestet !!!!
   else
      rot = V_Angle(erz, ez);
   dprintf(6, "CalcTrafo(): ics = %d, angle = %f, rot = %f, srot = %f, cos(rot) = %f, sin(rot) = %f\n",
      ics, tu->cs[ics]->d_angle, rot, srot, cos(rot), sin(rot));

   tu->cs[ics]->T[0][0] = 1.0;
   tu->cs[ics]->T[1][0] = 0.0;
   tu->cs[ics]->T[2][0] = 0.0;
   tu->cs[ics]->T[0][1] = 0.0;
   tu->cs[ics]->T[1][1] = srot*rot;
   tu->cs[ics]->T[2][1] = srot*float(sqrt(1-rot*rot));
   tu->cs[ics]->T[0][2] = 0.0;
   tu->cs[ics]->T[1][2] = srot*-float(sqrt(1-rot*rot));
   tu->cs[ics]->T[2][2] = srot*rot;
}


static void CalcEllipsePoints(struct cs *cs, float a, float b, float dx, float dy, float as, float ae)
{
   float phi;

   for (phi = as+(float)M_PI/20; phi < ae; phi += (float)M_PI/20)
   {
      AddPoint(cs->p, float(a*cos(phi) + dx), float(b*sin(phi) + dy), 0.0f);
   }
}


int CalcCSGeometry(struct tube *tu, int csi, float n[3])
{
   struct cs *cs;
   int j, ind;

   static float sx[] = {1, -1, -1,  1};
   static float sy[] = {1,  1, -1, -1};
   float dx, dy, a, b;

   cs = tu->cs[csi];

   // when we come the second time ...
   FreePointStruct(cs->p);
   cs->p = AllocPointStruct();
   // first we get Trafo from the relativ(crossection) to the absolute system
   CalcTrafo(tu, csi, n);

   // all corners, starting with the right top, next is left top, ...
   for (j = 0; j < 4; j += 2)
   {

      //////////////////////////////////////////////
      // the right top || the left bottom corner ...
      ind = j;
      a  = cs->c_a[ind];
      b  = cs->c_b[ind];
      dx = cs->c_width/2  - a;
      dy = cs->c_height/2 - b;

      // First point of corner
      AddPoint(cs->p, sx[ind]*(dx+a), sy[ind]*dy, 0.0);
      cs->cov_ind[ind*2] = cs->p->nump-1;

      // second point of corner and the ellipse points ...
      if (!IS_0(a))
      {
         CalcEllipsePoints(cs, a, b, sx[ind]*dx, sy[ind]*dy, float(M_PI/2)*(ind), float((M_PI/2)*(ind)+M_PI/2));
         AddPoint(cs->p, sx[ind]*dx, sy[ind]*(dy+b), 0.0);
      }
      cs->cov_ind[ind*2+1] = cs->p->nump-1;

      //////////////////////////////////////////////
      // the left top || the right bottom corner ...
      ind = j+1;
      a  = cs->c_a[ind];
      b  = cs->c_b[ind];
      dx = cs->c_width/2  - a;
      dy = cs->c_height/2 - b;
      AddPoint(cs->p, sx[ind] * dx, sy[ind]*(dy+b), 0.0);
      cs->cov_ind[ind*2] = cs->p->nump-1;
      if (!IS_0(b))
      {
         CalcEllipsePoints(cs, a, b, sx[ind]*dx, sy[ind]*dy, float((M_PI/2)*(ind)), float((M_PI/2)*(ind)+M_PI/2));
         AddPoint(cs->p, sx[ind]*(dx+a), sy[ind]*dy, 0.0);
      }
      cs->cov_ind[ind*2+1] = cs->p->nump-1;
   }
   return 1;
}


int CalcTubeGeometry(struct tube *tu)
{
   int i;

   if (tu && tu->cs_num)
   {
      for (i = 0; i < tu->cs_num; i++)
         CalcCSGeometry(tu, i, NULL);
      DumpTube(tu);
      return 1;
   }
   return 0;
}


void DumpTube(struct tube *t)
{
   int i;
   char fn[20];

   dprintf(4, "Struct tube:\n");
   dprintf(4, "\tcs_max = %d\n\tcs_num = %d\n", t->cs_max, t->cs_num);
   dprintf(4, "Peer: num=%d, max=%d\n", t->pe_num, t->pe_max);
   for (i = 0; i < t->pe_num; i++)
   {
      dprintf(4, "\tPeer %d:\n", i);
      dprintf(4, "missing info !!!\n");
   }
   for (i = 0; i < 4; i++)
      dprintf(4, "\tel[%d]    = %d\n", i, t->c_el[i]);
   dprintf(4, "\tel_o     = %d\n", t->c_el_o);
   for (i = 0; i < t->cs_num; i++)
   {
      dprintf(4, "\tSection %d:\n", i);
      sprintf(fn, "cs_%03d.gp", i);
      DumpT_CS(t->cs[i], fn);
   }
}


void DumpT_CS(struct cs *cs, char *fnbuf)
{
   int j;
   FILE *fp=NULL;
   char *fn;

   dprintf(5, "\tGeometry-Parameters ...\n");
   dprintf(5, "\t\tx           = %10.6f", cs->c_m_x);
   dprintf(5, "\ty           = %10.6f", cs->c_m_y);
   dprintf(5, "\tz           = %10.6f:\n", cs->c_m_z);
   dprintf(5, "\t\theight      = %10.6f", cs->c_height);
   dprintf(5, "\twidth       = %10.6f:\n", cs->c_width);
   dprintf(5, "\telems       = %10d:\n", cs->c_nume);
   dprintf(5, "\tangle       = %10.6f\n", cs->c_angle);
   dprintf(5, "\ttype angle  = %s\n", angletypes[cs->c_angletype]);
   for (j = 0; j < 8; j++)
      dprintf(5, "\tpart[%d]       = %f\n", j, cs->c_part[j]);
   for (j = 0; j < 4; j++)
   {
      dprintf(5, "\t\ta[%d]        = %10.6f", j, cs->c_a[j]);
      dprintf(5, "\tb[%d]        = %10.6f:\n", j, cs->c_b[j]);
   }
   dprintf(5, "\tTrafo:\n");
   for (j = 0; j < 3; j++)
   {
      dprintf(5, "   (%10.6f | %10.6f | %10.6f)\n", cs->T[j][0],
         cs->T[j][1],cs->T[j][2]);
   }
   dprintf(5, "\tPointlist:\n");
   if (fnbuf)
   {
      fn = DebugFilename(fnbuf);
   if(fn)
      fp = fopen(fn, "w");
      DumpPoints(cs->p, fp);
      if (fp && cs->p && cs->p->nump)
         fprintf(fp, "%10.6f %10.6f %10.6f\n", cs->p->x[0],
            cs->p->y[0],cs->p->z[0]);
      if (fp)  fclose(fp);
   }
   for (j = 0; j < 8; j++)
   {
      dprintf(5, "  cov_ind[%d]=%d\n", j, cs->cov_ind[j]);
   }
}


float CalcOneCSArea(struct cs *cs)
{
   float A, a, b;
   int j;

   if (cs->c_height <= 0.0 || cs->c_width <= 0.0)
      A = -1.0;
   else
   {
      A = cs->c_height * cs->c_width;
      for (j = 0; j < 4; j++)
      {
         a = cs->c_a[j];
         b = cs->c_b[j];
         A -= a*b;                                // Sub the corner rectangle
         A += float(M_PI * a * b / 4);                   // Add the part of the ellipse
      }
   }
   return A;
}
