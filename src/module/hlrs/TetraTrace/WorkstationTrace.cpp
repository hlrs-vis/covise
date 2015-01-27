/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "WorkstationTrace.h"
#include "trace.h"

void trace::traceWorkstation()
{
    /*
      int i, l;
      int f, t, g;

      int stopCell;
      float stopX, stopY, stopZ;
      float stopU, stopV, stopW;
      */

    if (numGrids > 1)
    {
        traceTransient();
        /*
         // transient case, so we have to keep track of which grid
         // to trace in
         for( l=0; l<numSteps; l++ )  // cycle through the grids
         {
            for( g=0; g<numGrids; g++ )
       {
          // set to- and from- grid
          f = g;
          if( g==numGrids-1 )
             t = 0;
      else
      t = g+1;

      // continue all traces
      for( i=0; i<numStart; i++ )
      {
      // if trace still running or vel 0 then continue
      fprintf(stderr, "actS[%d]: %d\n", i, actS[i]);

      if( actS[i] )
      {
      actS[i] = continueTrace( actC[i], actX[i], actY[i], actZ[i], actU[i], actV[i], actW[i], \ 
      actDt[i], traceGrid[f], traceGrid[t], \ 
      stopCell, stopX, stopY, stopZ, stopU, stopV, stopW );
      actC[i] = stopCell;
      actX[i] = stopX;
      actY[i] = stopY;
      actZ[i] = stopZ;
      actU[i] = stopU;
      actV[i] = stopV;
      actW[i] = stopW;
      storeActual( i );

      // find cell where we are in the next grid
      if( actS[i] )
      {
      actC[i] = traceGrid[t]->findActCell( actX[i], actY[i], actZ[i], \ 
      actU[i], actV[i], actW[i], actC[i] );
      if( actC[i]==-1 )
      actS[i] = 0;
      }

      }
      }
      }
      }
      */
    }
    else
    {
        traceStationary();
    }

    // done
    return;
}
