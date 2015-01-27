/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "trace.h"
#include "output.h"

#include <appl/ApplInterface.h>

#include <util/coviseCompat.h>
#include <do/coDoPoints.h>
#include <do/coDoSet.h>

coDistributedObject **trace::buildOutput(char **names)
{
    coDistributedObject **returnObject = NULL;
    char bfr[512];

    returnObject = new coDistributedObject *[2];
    returnObject[0] = returnObject[1] = NULL;

    // compute the output
    switch (traceStyle)
    {
    case 2: // lines
        delete[] returnObject;
        returnObject = buildOutput_Lines(names[0], names[1]);
        break;

    case 5: // fader
        delete[] returnObject;
        returnObject = buildOutput_Fader(names[0], names[1]);
        break;

    default: // POINTS or something unsupported
        returnObject[0] = buildOutput_Point(names[0]);
        returnObject[1] = buildOutput_PointData(names[1]);
        break;
    }

    // setup interaction (e.g. with COVER)
    switch (startStyle)
    {
    case 2: // plane
        sprintf(bfr, "P%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
        break;

    default: // line or something unsupported
        sprintf(bfr, "T%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
        break;
    }
    returnObject[0]->addAttribute("FEEDBACK", bfr);

    // done
    return (returnObject);
}

coDistributedObject **trace::buildOutput_Lines(char *lname, char *dname)
{
    (void)lname;
    (void)dname;
    /*
      coDoLines *lines = NULL;
      coDistributedObject **r = NULL;

      int i, k, n;
      //int l;
      int numLines, numVert;

      int *ll, *vl;
      float *x, *y, *z, *u, *v, *w;
      float *lx, *ly, *lz, *lu, *lv, *lw;

   r = new coDistributedObject*[2];
   r[0] = r[1] = NULL;

   // determine maximum size
   numLines = numVert = 0;
   for( i=0; i<numStart; i++ )
   {
   numLines++;
   numVert += wayNum[i];
   }

   // alloc mem
   lx = new float[numVert];
   ly = new float[numVert];
   lz = new float[numVert];
   ll = new int[numLines];
   vl = new int[numVert];

   lu = new float[numVert];
   lv = new float[numVert];
   lw = new float[numVert];

   // compute
   numVert = 0;
   for( i=0; i<numLines; i++ )
   {
   getOutput( i, n, x, y, z, u, v, w );

   ll[i] = numVert;
   for( k=0; k<n; k++ )
   {
   vl[numVert] = numVert;
   lx[numVert] = x[k];
   ly[numVert] = y[k];
   lz[numVert] = z[k];
   numVert++;
   }

   }

   // build output
   r[0] = new coDoLines( lname, numVert, lx, ly, lz, numVert, vl, numLines, ll );

   // clean up
   delete[] lx;
   delete[] ly;
   delete[] lz;
   delete[] ll;
   delete[] vl;
   delete[] lu;
   delete[] lv;
   delete[] lw;

   // done
   return( r );*/
    return (NULL);
}

void trace::getOutput(int tr, int &n, float *x, float *y, float *z, float *u, float *v, float *w)
{
    // we return the way of one single trace
    int i;

    n = wayNum[tr];

    x = new float[n];
    y = new float[n];
    z = new float[n];
    u = new float[n];
    v = new float[n];
    w = new float[n];

    for (i = 0; i < n; i++)
    {
        x[i] = wayX[tr][i];
        y[i] = wayY[tr][i];
        z[i] = wayZ[tr][i];
        u[i] = wayU[tr][i];
        v[i] = wayV[tr][i];
        w[i] = wayW[tr][i];
    }

    // done
    return;
}

coDistributedObject *trace::buildOutput_Point(char *name)
{
    coDistributedObject *returnObject = NULL;
    coDistributedObject **setElems = NULL;
    coDoPoints *points;

    float *x, *y, *z;

    int i, j, n, num;
    int cycles, c;

    char bfr[512];

    // figure out how many steps we'll have to generate
    if (numGrids > 1)
        num = numGrids;
    else
        num = numSteps;

    // alloc mem
    setElems = new coDistributedObject *[num + 1];
    setElems[num] = NULL;

    if (numGrids > 1)
    {
        fprintf(stderr, "numGrids: %d\nnum: %d\n", numGrids, num);
        //for( j=0; j<numStart; j++ )
        //   fprintf(stderr, "%d: %d\n", j, wayNum[j]);

        // build parts of the set
        for (i = 0; i < num; i++)
        {
            sprintf(bfr, "%s_%d", name, i);

            // how many traces are still running ?!
            n = 0;
            for (j = 0; j < numStart; j++)
                if (wayNum[j] > i)
                {
                    // we have this one in the first cycle
                    n++;
                    // and maybe in other cycles ?
                    cycles = (int)(wayNum[j] / (num + 1));
                    n += cycles;
                }

            if (n > 0)
            {
                fprintf(stderr, "%d points in step %d\n", n, i);

                points = new coDoPoints(bfr, n);
                points->getAddresses(&x, &y, &z);
                points->addAttribute("COLOR", "red");
                n = 0;

                for (j = 0; j < numStart; j++)
                {
                    if (wayNum[j] > i)
                    {
                        // we have this at least once
                        x[n] = wayX[j][i];
                        y[n] = wayY[j][i];
                        z[n] = wayZ[j][i];
                        n++;
                        // and maybe sometime later on
                        cycles = (int)(wayNum[j] / (num + 1));
                        for (c = 1; c <= cycles; c++)
                        {
                            x[n] = wayX[j][(c * num) + i];
                            y[n] = wayY[j][(c * num) + i];
                            z[n] = wayZ[j][(c * num) + i];
                            n++;
                        }
                    }
                }
            }
            else
                points = NULL;

            setElems[i] = points;
        }

        // assemble set
        returnObject = new coDoSet(name, setElems);
        sprintf(bfr, "1 %d", num);
        returnObject->addAttribute("TIMESTEP", bfr);

        // clean up
        for (i = 0; i < num; i++)
            if (setElems[i])
                delete setElems[i];
    }
    else
    {
        // build parts of the set
        for (i = 0; i < num; i++)
        {
            sprintf(bfr, "%s_%d", name, i);

            // how many traces are still running ?!
            n = 0;
            for (j = 0; j < numStart; j++)
                if (wayNum[j] > i)
                    n++;

            if (n > 0)
            {
                points = new coDoPoints(bfr, n);
                points->getAddresses(&x, &y, &z);
                points->addAttribute("COLOR", "red");
                n = 0;
                for (j = 0; j < numStart; j++)
                {
                    if (wayNum[j] > i)
                    {
                        x[n] = wayX[j][i];
                        y[n] = wayY[j][i];
                        z[n] = wayZ[j][i];
                        n++;
                    }
                }
            }
            else
                points = NULL;

            setElems[i] = points;
        }

        // assemble set
        returnObject = new coDoSet(name, setElems);
        sprintf(bfr, "1 %d", num);
        returnObject->addAttribute("TIMESTEP", bfr);

        // clean up
        for (i = 0; i < num; i++)
            if (setElems[i])
                delete setElems[i];
    }

    // done
    return (returnObject);
}

coDistributedObject *trace::buildOutput_PointData(char *name)
{
    coDistributedObject *returnObject = NULL;
    coDistributedObject **setElems = NULL;

    coDoFloat *magnitude = NULL;
    coDoVec3 *vel = NULL;

    float *d = NULL;
    float *u = NULL, *v = NULL, *w = NULL;

    int i, j, n, num;
    int cycles, c, c2;

    char bfr[512];

    // figure out how many steps we'll have to generate
    if (numGrids > 1)
        num = numGrids;
    else
        num = numSteps;

    // alloc mem
    setElems = new coDistributedObject *[num + 1];
    setElems[num] = NULL;
    if (numGrids > 1)
    {
        // build parts of the set
        for (i = 0; i < num; i++)
        {
            sprintf(bfr, "%s_%d", name, i);

            // how many traces are still running ?!
            n = 0;
            for (j = 0; j < numStart; j++)
                if (wayNum[j] > i)
                {
                    // we have this one in the first cycle
                    n++;
                    // and maybe in other cycles ?
                    cycles = (int)(wayNum[j] / (num + 1));
                    n += cycles;
                }
            if (n > 0)
            {
                switch (whatOut)
                {
                case 2: // velocity
                    vel = new coDoVec3(bfr, n);
                    vel->getAddresses(&u, &v, &w);
                    break;

                default: // magnitude
                    magnitude = new coDoFloat(bfr, n);
                    magnitude->getAddress(&d);
                    break;
                }
                n = 0;

                for (j = 0; j < numStart; j++)
                {
                    if (wayNum[j] > i)
                    {
                        // we have this at least once
                        switch (whatOut)
                        {
                        case 2:
                            u[n] = wayU[j][i];
                            v[n] = wayV[j][i];
                            w[n] = wayW[j][i];
                            break;

                        default:
                            d[n] = sqrt(wayU[j][i] * wayU[j][i] + wayV[j][i] * wayV[j][i] + wayW[j][i] * wayW[j][i]);
                            break;
                        }
                        n++;
                        // and maybe sometime later on
                        cycles = (int)(wayNum[j] / (num + 1));
                        for (c = 1; c <= cycles; c++)
                        {
                            c2 = (c * num) + i;
                            switch (whatOut)
                            {
                            case 2:
                                u[n] = wayU[j][i];
                                v[n] = wayV[j][i];
                                w[n] = wayW[j][i];
                                break;

                            default:
                                d[n] = sqrt(wayU[j][c2] * wayU[j][c2] + wayV[j][c2] * wayV[j][c2] + wayW[j][c2] * wayW[j][c2]);
                                break;
                            }
                            n++;
                        }
                    }
                }
            }
            else
            {
                magnitude = NULL;
                vel = NULL;
            }

            switch (whatOut)
            {
            case 2:
                setElems[i] = vel;
                break;

            default:
                setElems[i] = magnitude;
                break;
            }
        }

        // assemble set
        returnObject = new coDoSet(name, setElems);
        sprintf(bfr, "1 %d", num);
        returnObject->addAttribute("TIMESTEP", bfr);

        // clean up
        for (i = 0; i < num; i++)
            if (setElems[i])
                delete setElems[i];
    }
    else
    {
        // build parts of the set
        for (i = 0; i < num; i++)
        {
            sprintf(bfr, "%s_%d", name, i);

            // how many traces are still running ?!
            n = 0;
            for (j = 0; j < numStart; j++)
                if (wayNum[j] > i)
                    n++;
            if (n > 0)
            {
                switch (whatOut)
                {
                case 2: // velocity
                    vel = new coDoVec3(bfr, n);
                    vel->getAddresses(&u, &v, &w);
                    break;

                default: // magnitude
                    magnitude = new coDoFloat(bfr, n);
                    magnitude->getAddress(&d);
                    break;
                }
                n = 0;
                for (j = 0; j < numStart; j++)
                {
                    if (wayNum[j] > i)
                    {
                        switch (whatOut)
                        {
                        case 2:
                            u[n] = wayU[j][i];
                            v[n] = wayV[j][i];
                            w[n] = wayW[j][i];
                            break;

                        default:
                            d[n] = sqrt(wayU[j][i] * wayU[j][i] + wayV[j][i] * wayV[j][i] + wayW[j][i] * wayW[j][i]);
                            break;
                        }
                        n++;
                    }
                }
            }
            else
            {
                magnitude = NULL;
                vel = NULL;
            }

            switch (whatOut)
            {
            case 2:
                setElems[i] = vel;
                break;

            default:
                setElems[i] = magnitude;
                break;
            }
        }

        // assemble set
        returnObject = new coDoSet(name, setElems);
        sprintf(bfr, "1 %d", num);
        returnObject->addAttribute("TIMESTEP", bfr);

        // clean up
        for (i = 0; i < num; i++)
            if (setElems[i])
                delete setElems[i];
    }

    // done
    return (returnObject);
}

coDistributedObject **trace::buildOutput_Fader(char *pName, char *dName)
{
    coDistributedObject **returnObjects = NULL;
    coDistributedObject **setElems = NULL;
    coDistributedObject **dataSetElems = NULL;

    coDoPoints *points;
    coDoVec3 *data;

    float *x, *y, *z;
    float *u, *v, *w, scale;

    int i, j, n, num, f, o;
    int cycles, c;

    char bfr[512], bfr2[512];

    // prepare output
    returnObjects = new coDistributedObject *[2];
    returnObjects[0] = NULL;
    returnObjects[1] = NULL;

    // figure out how many steps we'll have to generate
    if (numGrids > 1)
        num = numGrids;
    else
        num = numSteps;

    // alloc mem
    setElems = new coDistributedObject *[num + 1];
    setElems[num] = NULL;
    dataSetElems = new coDistributedObject *[num + 1];
    dataSetElems[num] = NULL;

    if (numGrids > 1)
    {

        fprintf(stderr, "numGrids: %d\nnum: %d\n", numGrids, num);
        //for( j=0; j<numStart; j++ )
        //   fprintf(stderr, "%d: %d\n", j, wayNum[j]);

        // build parts of the set
        for (i = 0; i < num; i++)
        {
            sprintf(bfr, "%s_%d", pName, i);
            sprintf(bfr2, "%s_%d", dName, i);

            // how many traces are still running ?!
            n = 0;
            for (j = 0; j < numStart; j++)
                if (wayNum[j] > i)
                {
                    // we have this one in the first cycle
                    n++;
                    // and maybe in other cycles ?
                    cycles = (int)(wayNum[j] / (num + 1));
                    n += cycles;
                }

            if (n > 0)
            {
                fprintf(stderr, "%d points in step %d\n", n, i);

                points = new coDoPoints(bfr, n * 5);
                points->getAddresses(&x, &y, &z);
                points->addAttribute("COLOR", "red");
                data = new coDoVec3(bfr2, n * 5);
                data->getAddresses(&u, &v, &w);
                n = 0;

                for (j = 0; j < numStart; j++)
                {
                    if (wayNum[j] > i)
                    {
                        scale = 1.0f / 5.0f;
                        for (f = 0; f < 5; f++)
                        {
                            if (i - f > 0)
                                o = i - f;
                            else
                                o = 0;
                            x[n] = wayX[j][o];
                            y[n] = wayY[j][o];
                            z[n] = wayZ[j][o];

                            u[n] = wayU[j][o] * scale * ((float)(5 - f));
                            v[n] = wayV[j][o] * scale * ((float)(5 - f));
                            w[n] = wayW[j][o] * scale * ((float)(5 - f));

                            n++;
                        }

                        // maybe we have it in other cycles ?!
                        cycles = (int)(wayNum[j] / (num + 1));
                        for (c = 1; c <= cycles; c++)
                        {
                            for (f = 0; f < 5; f++)
                            {
                                if ((c * num) + i - f > 0)
                                    o = (c * num) + i - f;
                                else
                                    o = 0;

                                x[n] = wayX[j][o];
                                y[n] = wayY[j][o];
                                z[n] = wayZ[j][o];

                                u[n] = wayU[j][o] * scale * ((float)(5 - f));
                                v[n] = wayV[j][o] * scale * ((float)(5 - f));
                                w[n] = wayW[j][o] * scale * ((float)(5 - f));

                                n++;
                            }
                        }

                        /*
                  // we only support num == 1 !!!

                  // we have this at least once
                  x[n] = wayX[j][i];
                  y[n] = wayY[j][i];
                  z[n] = wayZ[j][i];
                  n++;
                  // and maybe sometime later on
                  cycles = (int)(wayNum[j]/(num+1));
                  for( c=1; c<=cycles; c++ )
                  {
                  x[n] = wayX[j][(c*num)+i];
                  y[n] = wayY[j][(c*num)+i];
                  z[n] = wayZ[j][(c*num)+i];
                  n++;
                  }
                  */
                    }
                }
            }
            else
            {
                points = NULL;
                data = NULL;
            }

            setElems[i] = points;
            dataSetElems[i] = data;
        }

        // assemble set
        returnObjects[0] = new coDoSet(pName, setElems);
        returnObjects[1] = new coDoSet(dName, dataSetElems);
        sprintf(bfr, "1 %d", num);
        returnObjects[0]->addAttribute("TIMESTEP", bfr);

        // clean up
        for (i = 0; i < num; i++)
        {
            if (setElems[i])
                delete setElems[i];
            if (dataSetElems[i])
                delete dataSetElems[i];
        }

        //cerr << "NOT IMPLEMENTED !!!" << endl;
    }
    else
    {
        // build parts of the set
        for (i = 0; i < num; i++)
        {
            sprintf(bfr, "%s_%d", pName, i);
            sprintf(bfr2, "%s_%d", dName, i);

            // how many traces are still running ?!
            n = 0;
            for (j = 0; j < numStart; j++)
                if (wayNum[j] > i)
                    n++;

            if (n > 0)
            {
                points = new coDoPoints(bfr, n * 5);
                points->getAddresses(&x, &y, &z);
                points->addAttribute("COLOR", "red");
                data = new coDoVec3(bfr2, n * 5);
                data->getAddresses(&u, &v, &w);
                n = 0;

                for (j = 0; j < numStart; j++)
                {
                    if (wayNum[j] > i)
                    {
                        scale = 1.0f / 5.0f;
                        for (f = 0; f < 5; f++)
                        {
                            if (i - f > 0)
                                o = i - f;
                            else
                                o = 0;
                            x[n] = wayX[j][o];
                            y[n] = wayY[j][o];
                            z[n] = wayZ[j][o];

                            u[n] = wayU[j][o] * scale * ((float)(5 - f));
                            v[n] = wayV[j][o] * scale * ((float)(5 - f));
                            w[n] = wayW[j][o] * scale * ((float)(5 - f));

                            n++;
                        }

                        //x[n] = wayX[j][i];
                        //y[n] = wayY[j][i];
                        //z[n] = wayZ[j][i];
                        //n++;
                    }
                }
            }
            else
            {
                points = NULL;
                data = NULL;
            }

            setElems[i] = points;
            dataSetElems[i] = data;
        }

        // assemble set
        returnObjects[0] = new coDoSet(pName, setElems);
        returnObjects[1] = new coDoSet(dName, dataSetElems);
        sprintf(bfr, "1 %d", num);
        returnObjects[0]->addAttribute("TIMESTEP", bfr);

        // clean up
        for (i = 0; i < num; i++)
        {
            if (setElems[i])
                delete setElems[i];
            if (dataSetElems[i])
                delete dataSetElems[i];
        }
    }

    // done
    return (returnObjects);
}
