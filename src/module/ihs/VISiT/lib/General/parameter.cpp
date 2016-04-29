#include <stdio.h>
#include <stdlib.h>
#ifdef WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif
#include <math.h>
#include "include/fatal.h"
#include "include/log.h"
#include "include/cfg.h"
#include "include/profile.h"
#include "include/parameter.h"

#define STAT   "stat%d"
#define FSTAT  "fstat%d"

struct parameter *AllocParameterStruct(int portion)
{
   struct parameter *para;

   if ((para = (struct parameter *)calloc(1, sizeof(struct parameter))) != NULL)
   {
      if (portion > 0 && portion < 1024)
         para->portion = portion;
      else
         para->portion = 25;
   }
   return para;
}


int AddParameter(struct parameter *para, float loc, float val)
{
   if ((para->num+1) >= para->max)
   {
      para->max += para->portion;
      if ((para->loc = (float *)realloc(para->loc, para->max*sizeof(float))) == NULL)
      {
         fatal("Memory for para->loc");
      }
      if ((para->val = (float *)realloc(para->val, para->max*sizeof(float))) == NULL)
      {
         fatal("Memory for para->val");
      }
   }
   para->loc[para->num] = loc;
   para->val[para->num] = val;
   para->num++;
   return (para->num-1);
}


int ReadParameterSet(struct parameter *para, const char *sec, const char *fn)
{
   int i, num = 0;
   float loc, val;
   char *tmp = NULL;
   char key[127];

   for (i = 0; ; i++)
   {
      sprintf(key, STAT, i);
      if ((tmp = IHS_GetCFGValue(fn, sec, key)) != NULL)
      {
         sscanf(tmp, "%f, %f", &loc, &val);
         free(tmp);
         num = AddParameter(para, loc, val);
      }
      else
      {
         if(i==0)                                 // missing section
         {
            dprintf(0,"ERROR: keyword missing (%s),section=%s, file=%s\n",
               key,sec,fn);
            //exit (1);
         }
         break;
      }
   }
   return(num);
}


float InterpolateParameterSet(struct parameter *para, float loc, int extrapol)
{
   int i;
   int iinter = 0;
   float x1, x2, y1, y2, val;

   if(!para) return 0.0;
   if(!para->loc) return 0.0;
   // consider parameter as const. if only one is set.
   if (para->num == 1)
   {
      return (para->val[0]);
   }

   // search parameter intervall
   if (para->loc[0] > loc)
      iinter = 1;
   else if (para->loc[para->num-1] < loc)
      iinter = para->num - 1;
   else
      for (i = 1; i < para->num; i++)
   {
      if (para->loc[i] >= loc)
      {
         iinter = i;
         break;
      }
   }
   // calculate parameter value at iinter
   if (!extrapol && (para->loc[0] > loc))
      val = para->val[0];
   else if (!extrapol && (para->loc[para->num-1] < loc))
      val = para->val[para->num-1];
   else
   {
      x1  = para->loc[iinter-1];
      x2  = para->loc[iinter];
      y1  = para->val[iinter-1];
      y2  = para->val[iinter];
      val = (y2 - y1)/(x2 - x1) * (loc - x1) + y1;
   }
   return val;
}


int Parameter2Radians(struct parameter *para)
{
   int i;

   for (i = 0; i < para->num; i++)
      para->val[i] *= ((float) M_PI/180.0f);

   return(0);
}


int Parameter2Profile(struct parameter *para, struct profile *prof)
{
   int i;
   int normalize = 0;
   float maxval = 0.0;

   prof->num = para->num;
   if (prof->num > prof->max)
   {
      if ((prof->c = (float *)realloc(prof->c, prof->num*sizeof(float))) == NULL)
         fatal("memory for (float) prof->c");
      if ((prof->t = (float *)realloc(prof->t, prof->num*sizeof(float))) == NULL)
         fatal("memory for (float) prof->t");
   }
   else
   {
      if (para->loc[para->num-1] != 1.0)
         normalize = 1;
      for (i = 0; i < para->num; i++)
      {
         if (normalize)
         {
            prof->c[i] = para->loc[i] / para->loc[para->num-1];
            prof->t[i] = para->val[i] / para->loc[para->num-1];
         }
         else
         {
            prof->c[i] = para->loc[i];
            prof->t[i] = para->val[i];
         }
         if (para->val[i] > maxval)
         {
            maxval      = para->val[i];
            prof->t_sec = i;
         }
      }
   }
   return(prof->num);
}


void FreeParameterStruct(struct parameter *para)
{
   if (para)
   {
      if (para->num && para->loc)   free(para->loc);
      if (para->num && para->val)   free(para->val);
      free(para);
   }
}


void DumpParameterSet(struct parameter *para)
{
   int i;

   dprintf(5, "para->num     = %d\n", para->num);
   dprintf(5, "para->max     = %d\n", para->max);
   dprintf(5, "para->portion = %d\n", para->portion);
   for (i = 0; i < para->num; i++)
      dprintf(5, "loc[%3d] = %7.4f  val[%3d] = %7.4f\n", i, para->loc[i],
         i, para->val[i]);
}


struct parafield *AllocParameterField(int portion)
{
   struct parafield *paraf;

   if ((paraf = (struct parafield *)calloc(1, sizeof(struct parafield))) == NULL)
      fatal("memory for struct paraf*");
   if (portion > 0 && portion < 1024)
      paraf->portion = portion;
   else
      paraf->portion = 10;
   return paraf;
}


int ReadParameterField(struct parafield *paraf, char *sec, char *subsec, const char *fn)
{
   int j;
   char *tmp;
   char seckey[128], subkey[128];

   for (j = 0; ; j++)
   {
      if ((paraf->num+1) > paraf->max)
      {
         paraf->max += paraf->portion;
         if ((paraf->loc = (float *)realloc(paraf->loc, paraf->max*sizeof(float))) == NULL)
            fatal("memory in parafield for realloc float*");
         if ((paraf->para = (struct parameter **)realloc(paraf->para, paraf->max*sizeof(struct parameter *))) == NULL)
            fatal("memory in parafield for realloc struct parameter*");
      }
      sprintf(seckey, FSTAT, j);
      if ((tmp = IHS_GetCFGValue(fn, sec, seckey)) != NULL)
      {
         sscanf(tmp, "%f", &paraf->loc[j]);
         free(tmp);
         sprintf(subkey, subsec, j);
         paraf->para[paraf->num] = AllocParameterStruct(30);
         ReadParameterSet(paraf->para[paraf->num++], subkey, fn);
      }
      else
         break;
   }
   return(j);
}


struct parameter *InterpolateParameterField(struct parafield *paraf, float loc, int extrapol)
{
   int i, iinter = -1;
   float x1, x2, y1, y2, val;
   struct parameter *para;

   if (paraf->num == 1)
   {
      return (paraf->para[0]);
   }

   para = AllocParameterStruct(30);
   // search parameter field intervall
   if (paraf->loc[0] > loc)
      iinter = 1;
   else if (paraf->loc[paraf->num-1] < loc)
      iinter = paraf->num-1;
   else
      for (i = 1; i < paraf->num; i++)
   {
      if (paraf->loc[i] >= loc)
      {
         iinter = i;
         break;
      }
   }
   if (iinter == -1)
   {
      dprintf(0, "InterpolateParameterField(): iinter nicht gefunden !\n");
      exit(1);
   }
   // calculate parameter value at iinter
   if (!extrapol && (paraf->loc[0] > loc))
      para = paraf->para[0];
   else if (!extrapol && (paraf->loc[paraf->num-1] < loc))
      para = paraf->para[paraf->num-1];
   else
      for (i = 0; i < paraf->para[iinter]->num; i++)
   {
      x1  = paraf->loc[iinter-1];
      x2  = paraf->loc[iinter];
      y1  = paraf->para[iinter-1]->val[i];
      y2  = paraf->para[iinter]->val[i];
      val = (y2 - y1)/(x2 - x1) * (loc - x1) + y1;
      AddParameter(para, paraf->para[iinter]->loc[i], val);
   }
   return(para);
}


void FreeParameterField(struct parafield *paraf)
{
   int i;

   if (paraf)
   {
      if (paraf->num && paraf->loc) free(paraf->loc);
      for (i = 0; i < paraf->num; i++)
         FreeParameterStruct(paraf->para[i]);
   }
   free(paraf);
}


void DumpParameterField(struct parafield *paraf)
{
   int i;

   dprintf(5, "paraf->num     = %d\n", paraf->num);
   dprintf(5, "paraf->max     = %d\n", paraf->max);
   dprintf(5, "paraf->portion = %d\n", paraf->portion);
   for (i = 0; i < paraf->num; i++)
   {
      dprintf(5, "\nparameter set %d: ", i);
      dprintf(5, "paraf->loc[%d] = %7.4f\n", i, paraf->loc[i]);
      DumpParameterSet(paraf->para[i]);
   }
}
