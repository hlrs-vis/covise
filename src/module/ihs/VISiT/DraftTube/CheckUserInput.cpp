// rei, Sun May 14 00:26:43 DST 2000
#include <stdio.h>
#include <General/include/log.h>

#define  MIN_LEN  0.001

#include "DraftTube.h"

void DraftTube::CheckUserInput(const char *portname, struct geometry *geo)
{
   int index;
   char name[255];

   index = -2;
   memset(name, 0, sizeof(name));

   if (portname && p_makeGrid->getName()
      && !strcmp(portname,p_makeGrid->getName()))
   {
      if (!p_makeGrid->getValue()) return;

      // push in : create Grids in compute() Call-Back
      selfExec();
   }
   else if (portname && p_ip_start->getName()
      && !strcmp(portname,p_ip_start->getName()))
   {
      if (!p_ip_start->getValue()) return;

      // push in : interpolate in compute() Call-Back
      selfExec();
   }
   else if (portname && p_absync->getName()
      && !strcmp(portname, p_absync->getName()))
   {
#define  NO_MORE_BUG_IN_COVISE
#ifdef   NO_MORE_BUG_IN_COVISE
      int i, j;

      for ( i = 0; i < MAX_CROSS; i++)
      {
         for (j = 1; j< 4; j++)
         {
            if (p_absync->getValue())
               p_ab[i][j]->disable();
            else
               p_ab[i][j]->enable();
         }
      }
#else
      ;
#endif
   }
   else if (SplitPortname(portname, name, &index))
   {
      if (!strncmp(name, P_AB, strlen(P_AB)))
      {
         char buf[255];
         char *direct;

         strcpy(buf, name);
         direct = strtok(buf, "(");
         strtok(direct, ")");
         // ab sync mode and direction == "NE"
         if (p_absync->getValue() && strcmp(direct, direction[0]))
         {
            // has a changed and the user input is ok ?
            if (geo->tu->cs[index]->c_a[0] != p_ab[index][0]->getValue(0)
               && p_ab[index][0]->getValue(0)*2 < p_hw[index]->getValue(0))
            {
               p_ab[index][1]->setValue(0, p_ab[index][0]->getValue(0));
               p_ab[index][2]->setValue(0, p_ab[index][0]->getValue(0));
               p_ab[index][3]->setValue(0, p_ab[index][0]->getValue(0));
            } else
            p_ab[index][0]->setValue(0, geo->tu->cs[index]->c_a[0]);

            // has b changed and the user input is ok ?
            if (geo->tu->cs[index]->c_b[0] != p_ab[index][0]->getValue(1)
               && p_ab[index][0]->getValue(1)*2 < p_hw[index]->getValue(1))
            {
               p_ab[index][1]->setValue(1, p_ab[index][0]->getValue(1));
               p_ab[index][2]->setValue(1, p_ab[index][0]->getValue(1));
               p_ab[index][3]->setValue(1, p_ab[index][0]->getValue(1));
            } else
            p_ab[index][0]->setValue(1, geo->tu->cs[index]->c_b[0]);
         }
      }
      else if (!strcmp(name, P_CS_AREA))
      {
         float Aold = CalcOneArea(index);
         float fact;
         int j;

         fact = sqrt(p_cs_area[index]->getValue()/Aold);
         p_hw[index]->setValue(0, p_hw[index]->getValue(0) * fact);
         p_hw[index]->setValue(1, p_hw[index]->getValue(1) * fact);
         for (j = 0; j < 4; j++)
         {
            p_ab[index][j]->setValue(0, p_ab[index][j]->getValue(0) * fact);
            p_ab[index][j]->setValue(1, p_ab[index][j]->getValue(1) * fact);
         }
      }
      //selfExec();
   }
   CalcValuesForCSArea();
   dprintf(2, "CheckUserInput(): portname = %s, name = %s, index = %d\n",
      portname, name, index);
}


int DraftTube::SplitPortname(const char *portname, char *name, int *index)
{
   int offs = strlen(portname);

   *name = '\0';
   *index = -1;
   if (portname && offs && portname[--offs] == ']')
   {
      while (strchr("0123456789", portname[--offs]))
         ;
      if (portname[offs] == '[' && portname[offs-1] == '_')
      {
         strncpy(name, portname, offs-1);
         name[offs-1] = '\0';
         *index = atoi(portname+offs+1) - 1;
         return 1;
      }
   }
   return 0;
}


char *DraftTube::IndexedParameterName(const char *name, int index)
{
   char buf[255];
   sprintf(buf, "%s_%d", name, index + 1);
   return strdup(buf);
}
