// rei, Sat May 13 21:29:40 DST 2000

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "DraftTube.h"
#include <General/include/v.h>
#include <General/include/log.h>

void DraftTube::InterpolateAreas()
{
   float len, d_lensum, d_len[MAX_CROSS];
   float m[3], lm[3], d[3];

   int i, j;
   int start = p_ip_S->getValue();
   int end   = p_ip_E->getValue();

   float d_a[4], d_b[4];                          // difference between start and end CS
   float sva[4], svb[4], svh, svw;                // start value of every parameter
   float d_height;                                // difference between start and end CS
   float d_width;                                 // difference between start and end CS

   dprintf(1, "Entering DraftTube::InterpolateAreas()\n");
   if (start > end)                               // stupid user ??
   {
      int tmp;

      tmp   = end;
      end  = start;
      start = tmp;
      p_ip_S->setValue(start);
      p_ip_E->setValue(end);
   }

   dprintf(3, "   start = %d, end = %d\n", start, end);
   // values in correct range ?
   if (end-start < 2 || start < 1 || end > MAX_CROSS)
   {
      p_ip_S->setValue(-999);                     // error message ;-)
      p_ip_E->setValue(-999);
      return;
   }

   // calculation of the complete len between both CS
   // (and between every CS)
   p_m[start-1]->getValue(lm[0], lm[1], lm[2]);
   for (i = start, d_lensum = 0.0; i < end; i++)
   {
      p_m[i]->getValue(m[0], m[1], m[2]);

      V_Sub(lm, m, d);                            // d = lm - m
      d_len[i] = V_Len(d);                        // distance between m and lm
      d_lensum += d_len[i];
      dprintf(5, "   d_len[%d] = %f\n", i, d_lensum);

      V_Copy(lm, m);                              // copy m --> lm
   }
   dprintf(5, "   d_lensum = %f\n", d_lensum);

   // first we calculate all differnces ...
   for (j = 0; j < 4; j++)
   {
      d_a[j] = p_ab[end-1][j]->getValue(0) - p_ab[start-1][j]->getValue(0);
      d_b[j] = p_ab[end-1][j]->getValue(1) - p_ab[start-1][j]->getValue(1);
      dprintf(5, "   d_a[%d] = %f, d_b[%d] = %f\n", j, d_a[j], j, d_b[j]);
   }
   d_height = p_hw[end-1]->getValue(0) - p_hw[start-1]->getValue(0);
   d_width  = p_hw[end-1]->getValue(1) - p_hw[start-1]->getValue(1);
   dprintf(5, "   d_height = %f, d_width = %f\n", d_height, d_width);

   // then we copy the start value (speedup !)
   for (j = 0; j < 4; j++)
   {
      sva[j] = p_ab[start-1][j]->getValue(0);
      svb[j] = p_ab[start-1][j]->getValue(1);
      dprintf(5, "   sva[%d] = %f, svb[%d] = %f\n", j, sva[j], j, svb[j]);
   }
   svh = p_hw[start-1]->getValue(0);
   svw = p_hw[start-1]->getValue(1);
   dprintf(5, "   svh = %f, svw = %f\n", svh, svw);

   // really interpolation
   // now we drive through all CS and do our work ...
   for (i = start, len = 0.0; i < end-1; i++)
   {
      len += d_len[i];
      dprintf(5, "   interpolation of %d\n", i);

      // a,b parameters
      if (p_ip_ab->getValue())
      {
         int j;

         for (j = 0; j < 4; j++)
         {
            p_ab[i][j]->setValue(0, LinearInterpolation(sva[j], d_a[j], len, d_lensum));
            p_ab[i][j]->setValue(1, LinearInterpolation(svb[j], d_b[j], len, d_lensum));
            dprintf(5, "       p_ab[%d][%d]->getValue(0,1) = (%f, %f)\n", i, j,
               p_ab[i][j]->getValue(0),  p_ab[i][j]->getValue(1));
         }
      }

      // height
      if (p_ip_height->getValue())
         p_hw[i]->setValue(0, LinearInterpolation(svh, d_height, len, d_lensum));

      // width
      if (p_ip_width->getValue())
         p_hw[i]->setValue(1, LinearInterpolation(svw, d_width, len, d_lensum));
      dprintf(5, "       p_hw[%d]->getValue(0,1) = (%f, %f)\n", i,
         p_hw[i]->getValue(0),  p_hw[i]->getValue(1));
   }
}


// the selection type is ignored that moment ...
// because there is only one interplation method :-)
float DraftTube::LinearInterpolation(float start, float d, float len, float sumlen)
{
   return start + d * (len/sumlen);
}
