/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Strct.h"

// only for debugging
#include "LTracer.h"

StrctBlock::StrctBlock(GridStep *parent, Grid *owner, int b)
    : GridBlock(parent, owner, b)
{
}

StrctBlock::~StrctBlock()
{
    if (aUData)
        delete[] aUData;
    if (aVData)
        delete[] aVData;
    if (aWData)
        delete[] aWData;
}

int StrctBlock::initialize(coDistributedObject *g, coDistributedObject *d, int sS, int bBa, float attA)
{
    coDoStructuredGrid *grid = (coDoStructuredGrid *)g;
    grid->getGridSize(&iDim, &jDim, &kDim);
    grid->getAddresses(&xCoord, &yCoord, &zCoord);

    saveSearch = sS;
    boundingBoxAlgorithm = bBa;
    attackAngle = attA;

    aUData = NULL;
    aVData = NULL;
    aWData = NULL;

    const char *dataType;
    dataType = d->getType();
    if (!strcmp(dataType, "USTVDT"))
    {
        coDoVec3 *v = (coDoVec3 *)d;
        v->getAddresses(&uData, &vData, &wData);
    }
    else if (!strcmp(dataType, "SETELE"))
    {
        // special case
        const coDistributedObject *const *tObj = ((coDoSet *)d)->getAllElements(&numSteps);
        aUData = new float *[numSteps];
        aVData = new float *[numSteps];
        aWData = new float *[numSteps];
        int i;
        for (i = 0; i < numSteps; i++)
            ((coDoVec3 *)tObj[i])->getAddresses(&aUData[i], &aVData[i], &aWData[i]);
        uData = aUData[0];
        vData = aVData[0];
        wData = aWData[0];
    }
    else
    {
        uData = NULL;
        vData = NULL;
        wData = NULL;
    }
    //   cerr << "DataType: " << dataType << endl;
    //   cerr << "Dim: " << iDim << " " << jDim << " " << kDim << endl;

    // speedup
    iStep = jDim * kDim;
    jStep = kDim;
    kStep = 1;

    // lookup-table for speedup
    cornerTbl[0] = 0;
    cornerTbl[1] = kStep;
    cornerTbl[2] = kStep + iStep;
    cornerTbl[3] = iStep;
    cornerTbl[4] = jStep;
    cornerTbl[5] = jStep + kStep;
    cornerTbl[6] = jStep + kStep + iStep;
    cornerTbl[7] = jStep + iStep;

    // and another one for the tetrahedra-decomposition
    tetraTable[0][0][0] = cornerTbl[0];
    tetraTable[0][0][1] = cornerTbl[2];
    tetraTable[0][0][2] = cornerTbl[5];
    tetraTable[0][0][3] = cornerTbl[1];
    tetraTable[0][1][0] = cornerTbl[0];
    tetraTable[0][1][1] = cornerTbl[5];
    tetraTable[0][1][2] = cornerTbl[7];
    tetraTable[0][1][3] = cornerTbl[4];
    tetraTable[0][2][0] = cornerTbl[0];
    tetraTable[0][2][1] = cornerTbl[2];
    tetraTable[0][2][2] = cornerTbl[7];
    tetraTable[0][2][3] = cornerTbl[5];
    tetraTable[0][3][0] = cornerTbl[0];
    tetraTable[0][3][1] = cornerTbl[7];
    tetraTable[0][3][2] = cornerTbl[2];
    tetraTable[0][3][3] = cornerTbl[3];
    tetraTable[0][4][0] = cornerTbl[2];
    tetraTable[0][4][1] = cornerTbl[7];
    tetraTable[0][4][2] = cornerTbl[5];
    tetraTable[0][4][3] = cornerTbl[6];

    tetraTable[1][0][0] = cornerTbl[4];
    tetraTable[1][0][1] = cornerTbl[3];
    tetraTable[1][0][2] = cornerTbl[1];
    tetraTable[1][0][3] = cornerTbl[0];
    tetraTable[1][1][0] = cornerTbl[3];
    tetraTable[1][1][1] = cornerTbl[4];
    tetraTable[1][1][2] = cornerTbl[6];
    tetraTable[1][1][3] = cornerTbl[7];
    tetraTable[1][2][0] = cornerTbl[3];
    tetraTable[1][2][1] = cornerTbl[6];
    tetraTable[1][2][2] = cornerTbl[1];
    tetraTable[1][2][3] = cornerTbl[2];
    tetraTable[1][3][0] = cornerTbl[1];
    tetraTable[1][3][1] = cornerTbl[6];
    tetraTable[1][3][2] = cornerTbl[4];
    tetraTable[1][3][3] = cornerTbl[5];
    tetraTable[1][4][0] = cornerTbl[4];
    tetraTable[1][4][1] = cornerTbl[6];
    tetraTable[1][4][2] = cornerTbl[1];
    tetraTable[1][4][3] = cornerTbl[3];

    // a table to lookup a specific tetrahedra's neighbour...whoooot
    //   [decompositionOrder][tetra][side][0=di,1=dj,2=dk,3=newTetra]
    neighbourTable[0][0][0][0] = 0;
    neighbourTable[0][0][0][1] = 0;
    neighbourTable[0][0][0][2] = 0;
    neighbourTable[0][0][0][3] = 2;
    neighbourTable[0][0][1][0] = 0;
    neighbourTable[0][0][1][1] = -1;
    neighbourTable[0][0][1][2] = 0;
    neighbourTable[0][0][1][3] = 3;
    neighbourTable[0][0][2][0] = -1;
    neighbourTable[0][0][2][1] = 0;
    neighbourTable[0][0][2][2] = 0;
    neighbourTable[0][0][2][3] = 2;
    neighbourTable[0][0][3][0] = 0;
    neighbourTable[0][0][3][1] = 0;
    neighbourTable[0][0][3][2] = +1;
    neighbourTable[0][0][3][3] = 0;

    neighbourTable[0][1][0][0] = 0;
    neighbourTable[0][1][0][1] = 0;
    neighbourTable[0][1][0][2] = 0;
    neighbourTable[0][1][0][3] = 2;
    neighbourTable[0][1][1][0] = -1;
    neighbourTable[0][1][1][1] = 0;
    neighbourTable[0][1][1][2] = 0;
    neighbourTable[0][1][1][3] = 1;
    neighbourTable[0][1][2][0] = 0;
    neighbourTable[0][1][2][1] = 0;
    neighbourTable[0][1][2][2] = -1;
    neighbourTable[0][1][2][3] = 3;
    neighbourTable[0][1][3][0] = 0;
    neighbourTable[0][1][3][1] = +1;
    neighbourTable[0][1][3][2] = 0;
    neighbourTable[0][1][3][3] = 0;

    neighbourTable[0][2][0][0] = 0;
    neighbourTable[0][2][0][1] = 0;
    neighbourTable[0][2][0][2] = 0;
    neighbourTable[0][2][0][3] = 3;
    neighbourTable[0][2][1][0] = 0;
    neighbourTable[0][2][1][1] = 0;
    neighbourTable[0][2][1][2] = 0;
    neighbourTable[0][2][1][3] = 0;
    neighbourTable[0][2][2][0] = 0;
    neighbourTable[0][2][2][1] = 0;
    neighbourTable[0][2][2][2] = 0;
    neighbourTable[0][2][2][3] = 1;
    neighbourTable[0][2][3][0] = 0;
    neighbourTable[0][2][3][1] = 0;
    neighbourTable[0][2][3][2] = 0;
    neighbourTable[0][2][3][3] = 4;

    neighbourTable[0][3][0][0] = 0;
    neighbourTable[0][3][0][1] = 0;
    neighbourTable[0][3][0][2] = 0;
    neighbourTable[0][3][0][3] = 2;
    neighbourTable[0][3][1][0] = 0;
    neighbourTable[0][3][1][1] = 0;
    neighbourTable[0][3][1][2] = -1;
    neighbourTable[0][3][1][3] = 2;
    neighbourTable[0][3][2][0] = 0;
    neighbourTable[0][3][2][1] = -1;
    neighbourTable[0][3][2][2] = 0;
    neighbourTable[0][3][2][3] = 1;
    neighbourTable[0][3][3][0] = +1;
    neighbourTable[0][3][3][1] = 0;
    neighbourTable[0][3][3][2] = 0;
    neighbourTable[0][3][3][3] = 0;

    neighbourTable[0][4][0][0] = 0;
    neighbourTable[0][4][0][1] = 0;
    neighbourTable[0][4][0][2] = 0;
    neighbourTable[0][4][0][3] = 2;
    neighbourTable[0][4][1][0] = +1;
    neighbourTable[0][4][1][1] = 0;
    neighbourTable[0][4][1][2] = 0;
    neighbourTable[0][4][1][3] = 3;
    neighbourTable[0][4][2][0] = 0;
    neighbourTable[0][4][2][1] = 0;
    neighbourTable[0][4][2][2] = +1;
    neighbourTable[0][4][2][3] = 1;
    neighbourTable[0][4][3][0] = 0;
    neighbourTable[0][4][3][1] = +1;
    neighbourTable[0][4][3][2] = 0;
    neighbourTable[0][4][3][3] = 2;

    neighbourTable[1][0][0][0] = 0;
    neighbourTable[1][0][0][1] = 0;
    neighbourTable[1][0][0][2] = 0;
    neighbourTable[1][0][0][3] = 4;
    neighbourTable[1][0][1][0] = 0;
    neighbourTable[1][0][1][1] = 0;
    neighbourTable[1][0][1][2] = -1;
    neighbourTable[1][0][1][3] = 0;
    neighbourTable[1][0][2][0] = -1;
    neighbourTable[1][0][2][1] = 0;
    neighbourTable[1][0][2][2] = 0;
    neighbourTable[1][0][2][3] = 3;
    neighbourTable[1][0][3][0] = 0;
    neighbourTable[1][0][3][1] = -1;
    neighbourTable[1][0][3][2] = 0;
    neighbourTable[1][0][3][3] = 1;

    neighbourTable[1][1][0][0] = 0;
    neighbourTable[1][1][0][1] = 0;
    neighbourTable[1][1][0][2] = 0;
    neighbourTable[1][1][0][3] = 4;
    neighbourTable[1][1][1][0] = 0;
    neighbourTable[1][1][1][1] = 0;
    neighbourTable[1][1][1][2] = -1;
    neighbourTable[1][1][1][3] = 4;
    neighbourTable[1][1][2][0] = +1;
    neighbourTable[1][1][2][1] = 0;
    neighbourTable[1][1][2][2] = 0;
    neighbourTable[1][1][2][3] = 1;
    neighbourTable[1][1][3][0] = 0;
    neighbourTable[1][1][3][1] = +1;
    neighbourTable[1][1][3][2] = 0;
    neighbourTable[1][1][3][3] = 3;

    neighbourTable[1][2][0][0] = 0;
    neighbourTable[1][2][0][1] = 0;
    neighbourTable[1][2][0][2] = 0;
    neighbourTable[1][2][0][3] = 4;
    neighbourTable[1][2][1][0] = +1;
    neighbourTable[1][2][1][1] = 0;
    neighbourTable[1][2][1][2] = 0;
    neighbourTable[1][2][1][3] = 0;
    neighbourTable[1][2][2][0] = 0;
    neighbourTable[1][2][2][1] = -1;
    neighbourTable[1][2][2][2] = 0;
    neighbourTable[1][2][2][3] = 4;
    neighbourTable[1][2][3][0] = 0;
    neighbourTable[1][2][3][1] = 0;
    neighbourTable[1][2][3][2] = +1;
    neighbourTable[1][2][3][3] = 3;

    neighbourTable[1][3][0][0] = 0;
    neighbourTable[1][3][0][1] = 0;
    neighbourTable[1][3][0][2] = 0;
    neighbourTable[1][3][0][3] = 4;
    neighbourTable[1][3][1][0] = 0;
    neighbourTable[1][3][1][1] = 0;
    neighbourTable[1][3][1][2] = +1;
    neighbourTable[1][3][1][3] = 1;
    neighbourTable[1][3][2][0] = -1;
    neighbourTable[1][3][2][1] = 0;
    neighbourTable[1][3][2][2] = 0;
    neighbourTable[1][3][2][3] = 4;
    neighbourTable[1][3][3][0] = 0;
    neighbourTable[1][3][3][1] = +1;
    neighbourTable[1][3][3][2] = 0;
    neighbourTable[1][3][3][3] = 0;

    neighbourTable[1][4][0][0] = 0;
    neighbourTable[1][4][0][1] = 0;
    neighbourTable[1][4][0][2] = 0;
    neighbourTable[1][4][0][3] = 3;
    neighbourTable[1][4][1][0] = 0;
    neighbourTable[1][4][1][1] = 0;
    neighbourTable[1][4][1][2] = 0;
    neighbourTable[1][4][1][3] = 1;
    neighbourTable[1][4][2][0] = 0;
    neighbourTable[1][4][2][1] = 0;
    neighbourTable[1][4][2][2] = 0;
    neighbourTable[1][4][2][3] = 0;
    neighbourTable[1][4][3][0] = 0;
    neighbourTable[1][4][3][1] = 0;
    neighbourTable[1][4][3][2] = 0;
    neighbourTable[1][4][3][3] = 2;

    // compute bounding-box (of entire block and of largest found cell inside the block)
    int i, j, k, o = 0, m;
    float *xP, *yP, *zP;
    float xCMin, xCMax, yCMin, yCMax, zCMin, zCMax, v;
    xMin = xCoord[o];
    yMin = yCoord[o];
    zMin = zCoord[o];
    xMax = xCoord[o];
    yMax = yCoord[o];
    zMax = zCoord[o];
    searchSizeX = 0.0;
    searchSizeY = 0.0;
    searchSizeZ = 0.0;
    for (i = 0; i < iDim; i++)
    {
        for (j = 0; j < jDim; j++)
        {
            for (k = 0; k < kDim; k++)
            {
                if (xCoord[o] > xMax)
                    xMax = xCoord[o];
                else if (xCoord[o] < xMin)
                    xMin = xCoord[o];

                if (yCoord[o] > yMax)
                    yMax = yCoord[o];
                else if (yCoord[o] < yMin)
                    yMin = yCoord[o];

                if (zCoord[o] > zMax)
                    zMax = zCoord[o];
                else if (zCoord[o] < zMin)
                    zMin = zCoord[o];

                if (i != iDim - 1 && j != jDim - 1 && k != kDim - 1)
                {
                    xP = xCoord + o;
                    yP = yCoord + o;
                    zP = zCoord + o;
                    // first: check boundingbox of this cell with the point
                    xCMin = *xP;
                    yCMin = *yP;
                    zCMin = *zP;
                    xCMax = xCMin;
                    yCMax = yCMin;
                    zCMax = zCMin;
                    for (m = 1; m < 8; m++)
                    {
                        v = *(xP + cornerTbl[m]);
                        if (v > xCMax)
                            xCMax = v;
                        else if (v < xCMin)
                            xCMin = v;

                        v = *(yP + cornerTbl[m]);
                        if (v > yCMax)
                            yCMax = v;
                        else if (v < yCMin)
                            yCMin = v;

                        v = *(zP + cornerTbl[m]);
                        if (v > zCMax)
                            zCMax = v;
                        else if (v < zCMin)
                            zCMin = v;
                    }

                    if (xCMax - xCMin > searchSizeX)
                        searchSizeX = xCMax - xCMin;
                    if (yCMax - yCMin > searchSizeY)
                        searchSizeY = yCMax - yCMin;
                    if (zCMax - zCMin > searchSizeZ)
                        searchSizeZ = zCMax - zCMin;
                }
                o++;
            }
        }
    }

    //searchSizeX *= 2.0;
    //searchSizeY *= 2.0;
    //searchSizeZ *= 2.0;

    //   cerr << "BoundingBox: ( " << xMin << " , " << yMin << " , " << zMin << " )" << endl;
    //   cerr << "                  -     ( " << xMax << " , " << yMax << " , " << zMax << " )" << endl;

    // calculate size of search-box
    /*
   searchSizeX = (xMax-xMin)*(yMax-yMin)*(zMax-zMin);
   searchSizeX = powf(searchSizeX/(iDim*jDim*kDim), 1.0/3.0)*5;

   cerr << "searchSizeX: " << searchSizeX << endl;
   searchSizeY = searchSizeZ = searchSizeX;
   */
    // nachher an BBox angepasste groessen probieren

    /*
   searchSizeX = (xMax-xMin)/powf((iDim*jDim*kDim), 1/3);
   searchSizeY = (yMax-yMin)/powf((iDim*jDim*kDim), 1/3);
   searchSizeZ = (zMax-zMin)/powf((iDim*jDim*kDim), 1/3);
   */
    //cerr << "searchSizeX: " << searchSizeX << "  ";
    //cerr << "searchSizeY: " << searchSizeY << "  ";
    //cerr << "searchSizeZ: " << searchSizeZ << endl;

    /* why wasnt this commented out ? it should be :)

   // NOTE: this might need some extended tweaking...but for now we start off with
   //  this search-box...eventually we might adapt the box in the search-functions
   searchSizeX = (xMax-xMin)/(((float)(iDim+jDim+kDim))/9);
   searchSizeY = (yMax-yMin)/(((float)(iDim+jDim+kDim))/9);
   searchSizeZ = (zMax-zMin)/(((float)(iDim+jDim+kDim))/9);

   // ok...try this (for debugging....)
   */

    // now this really needs to be adaptive (sort of)

    //searchSizeX = 1.0;
    //searchSizeY = 1.0;
    //searchSizeZ = 1.0;

    //;float scale = (xMax-xMin)*(yMax-yMin)*(zMax-zMin)

    /*
   searchSizeX *= 1000;
   searchSizeY *= 1000;
   searchSizeZ *= 1000;
   */

    //   cerr << "searchSizeX: " << searchSizeX << endl;
    //   cerr << "searchSizeY: " << searchSizeY << endl;
    //   cerr << "searchSizeZ: " << searchSizeZ << endl;

    // done
    if (!uData || !vData || !wData)
        return (0);
    return (1);
}

extern long int d_initGBI;

int StrctBlock::initGBI(float x, float y, float z, GridBlockInfo *&gbi)
{
    int i, j, k, iFound, jFound, kFound, tFound;
    //int t, preFlag = 0;
    //float u, v, w, dc;
    float dc;

    d_initGBI++;

    // check boundingbox
    if (x < xMin || x > xMax || y < yMin || y > yMax || z < zMin || z > zMax)
        return (0);

    // something cached ?
    iFound = -1;

    if (gbiCache)
    {
        i = ((StrctBlockInfo *)gbiCache)->i;
        j = ((StrctBlockInfo *)gbiCache)->j;
        k = ((StrctBlockInfo *)gbiCache)->k;
        tFound = ((StrctBlockInfo *)gbiCache)->tetra;

        //cerr << "   cache try" << endl;

        if (this->findNextCell(x, y, z, i, j, k, tFound, dc) == 1)
        {
            //cerr << "   cache hit" << endl;
            //cerr << ".";

            // das bringt praktisch garnichts die Einsparung durch
            //  direkte Geschwindigkeitsberechnung

            iFound = i;
            jFound = j;
            kFound = k;
        }
    }
    //   else
    //      cerr << "BFS " << endl;

    // nothing cached or cache MISS
    if (iFound == -1)
    {
        // try to trace from one of the block-corners towards the requested point
        //   if not found, then perform a brute-force search on the entire block

        //      cerr << "...starting search for particle..." << endl;

        // remember to set the tetra to the correct value

        if (!bruteForceSearch(x, y, z, iFound, jFound, kFound, tFound))
            return (0);
    }

    // found -> store information
    gbi = new StrctBlockInfo();

    gbi->x = x;
    gbi->y = y;
    gbi->z = z;
    gbi->block = this;
    gbi->step = step;
    gbi->stepNo = step->stepNo();
    gbi->blockNo = this->blockNo();
    ((StrctBlockInfo *)gbi)->i = iFound;
    ((StrctBlockInfo *)gbi)->j = jFound;
    ((StrctBlockInfo *)gbi)->k = kFound;
    ((StrctBlockInfo *)gbi)->tetra = tFound;
    ((StrctBlockInfo *)gbi)->iTmp = iFound;
    ((StrctBlockInfo *)gbi)->jTmp = jFound;
    ((StrctBlockInfo *)gbi)->kTmp = kFound;
    ((StrctBlockInfo *)gbi)->tTmp = tFound;

    //   cerr << "...hmmm ?!?..." << endl;

    // store datavalues in gbi (A..H)
    //StrctBlockInfo *rbi = (StrctBlockInfo *)gbi;
    //int o;                                         //, t;

    //   cerr << "...uebernehme daten in cache..." << endl;

    /*
   o = rbi->iTmp*jDim*kDim+rbi->jTmp*kDim+rbi->kTmp;
   t = o+kStep;
   rbi->A[0] = uData[t]; rbi->A[1] = vData[t]; rbi->A[2] = wData[t];
   t += jStep;
   rbi->B[0] = uData[t]; rbi->B[1] = vData[t]; rbi->B[2] = wData[t];
   t += iStep;
   rbi->C[0] = uData[t]; rbi->C[1] = vData[t]; rbi->C[2] = wData[t];
   t -= jStep;
   rbi->D[0] = uData[t]; rbi->D[1] = vData[t]; rbi->D[2] = wData[t];
   t = o;
   rbi->E[0] = uData[t]; rbi->E[1] = vData[t]; rbi->E[2] = wData[t];
   t += jStep;
   rbi->F[0] = uData[t]; rbi->F[1] = vData[t]; rbi->F[2] = wData[t];
   t += iStep;
   rbi->G[0] = uData[t]; rbi->G[1] = vData[t]; rbi->G[2] = wData[t];
   t -= jStep;
   rbi->H[0] = uData[t]; rbi->H[1] = vData[t]; rbi->H[2] = wData[t];
   */

    //   cerr << "...ok..." << endl;

    // get the velocity
    //  wird das nicht schon in bruteForceSearch / findNextCell gemacht ?
    //  oder kann zumindest dadurch optimiert werden ?

    /*
      int d, d2, *tt;
      float *uP, *vP, *wP;
      float v00,v01,v02,v10,v11,v12,v20,v21,v22,v30,v31,v32;

      d = iFound*iStep + jFound*jStep + kFound;
      uP = uData+d;
      vP = vData+d;
      wP = wData+d;

      //d = decompositionOrder(rbi->iTmp, rbi->jTmp, rbi->kTmp);

   tt = tetraTable[d_][tFound];

   d2 = tt[0];
   v00 = *(uP+d2);
   v01 = *(vP+d2);
   v02 = *(wP+d2);

   d2 = tt[1];
   v10 = *(uP+d2);
   v11 = *(vP+d2);
   v12 = *(wP+d2);

   d2 = tt[2];
   v20 = *(uP+d2);
   v21 = *(vP+d2);
   v22 = *(wP+d2);

   d2 = tt[3];
   v30 = *(uP+d2);
   v31 = *(vP+d2);
   v32 = *(wP+d2);

   gbi->u = (v00*w0_ + v10*w1_ + v20*w2_ + v30*w3_);
   gbi->v = (v01*w0_ + v11*w1_ + v21*w2_ + v31*w3_);
   gbi->w = (v02*w0_ + v12*w1_ + v22*w2_ + v32*w3_);
   */

    /*
      //cerr << "igbi" << endl;
      if( preFlag )
      {
         gbi->u = u;
         gbi->v = v;
         gbi->w = w;
      }
      else
      {
         cerr << ".";
   this->getVelocity(gbi, 0.0, 0.0, 0.0, &gbi->u, &gbi->v, &gbi->w);
   }
   */

    this->getVelocity(gbi, 0.0, 0.0, 0.0, &gbi->u, &gbi->v, &gbi->w, dc);

    // update cache
    if (!gbiCache)
        gbiCache = gbi->createCopy();
    else
        *gbiCache = *gbi;

    return (1);
}

extern long int d_getVel;

// Ausgangszeit: 3,050sec
//  3,220sec mit iTmp usw. variablen
//  Endergebnis: 2,950sec
int StrctBlock::getVelocity(GridBlockInfo *gbi, float dx, float dy, float dz, float *u, float *v, float *w, float &dc)
{
    int *tt;
    // int r;
    StrctBlockInfo *rbi = (StrctBlockInfo *)gbi;

    //if( dx==0.0 && dy==0.0 && dz==0.0 )
    //   cerr << "u";
    d_getVel++;

    //if( bruteForceSearch(rbi->x+dx, rbi->y+dy, rbi->z+dz, rbi->iTmp, rbi->jTmp, rbi->kTmp, rbi->tTmp) )
    if (findNextCell(rbi->x + dx, rbi->y + dy, rbi->z + dz, rbi->iTmp, rbi->jTmp, rbi->kTmp, rbi->tTmp, dc))
    {
        int d, d2;
        float *uP, *vP, *wP;
        float v00, v01, v02, v10, v11, v12, v20, v21, v22, v30, v31, v32;

        d = rbi->iTmp * iStep + rbi->jTmp * jStep + rbi->kTmp;
        uP = uData + d;
        vP = vData + d;
        wP = wData + d;

        //d = decompositionOrder(rbi->iTmp, rbi->jTmp, rbi->kTmp);

        tt = tetraTable[d_][rbi->tTmp];

        d2 = tt[0];
        v00 = *(uP + d2);
        v01 = *(vP + d2);
        v02 = *(wP + d2);

        d2 = tt[1];
        v10 = *(uP + d2);
        v11 = *(vP + d2);
        v12 = *(wP + d2);

        d2 = tt[2];
        v20 = *(uP + d2);
        v21 = *(vP + d2);
        v22 = *(wP + d2);

        d2 = tt[3];
        v30 = *(uP + d2);
        v31 = *(vP + d2);
        v32 = *(wP + d2);

        ///wg_;
        *u = (v00 * w0_ + v10 * w1_ + v20 * w2_ + v30 * w3_);
        ///wg_;
        *v = (v01 * w0_ + v11 * w1_ + v21 * w2_ + v31 * w3_);
        ///wg_;
        *w = (v02 * w0_ + v12 * w1_ + v22 * w2_ + v32 * w3_);

        return (1);
    }

    // NOTE: attA-condition not yet supported for STRGRD

    //   cerr << "#+# getVelocity failed due to bruteForceSearch-failure #+#" << endl;

    return (0);
}

extern long int d_fnc;

// Ausgangszeit: 6,280sec
//  Endzeit: 5,880sec
int StrctBlock::findNextCell(float x, float y, float z, int &i, int &j, int &k, int &t, float &dc)
{
    int r, c = 0;
    int d, d2, m;
    //float wg, w0, w1, w2, w3;
    float *xP, *yP, *zP;
    float p0[3], p1[3], p2[3], p3[3], px[3];
    int lastI, lastJ, lastK, lastT, l, lMax;
    int curI, curJ, curK, curT;
    int *ptrTbl;
    float vMax;

    d_fnc++;

    //   cerr << "fnC:   Starting at (" << i << "," << j << "," << k << ":" << t << ")   block " << myBlock;
    //   cerr << "  " << iDim << " " << jDim << " " << kDim;
    //   cerr << "   " << i*iStep+j*jStep+k*kStep << endl;

    px[0] = x;
    px[1] = y;
    px[2] = z;

    if (t > 5 || t < 0)
    {
        cerr << "fnC:  ERROR ERROR ERROR !!!!!  , t=" << t << "       " << i << " " << j << " " << k << endl;
        return (0);
    }

    lastI = -1;
    lastJ = -1;
    lastK = -1;
    lastT = -1;
    dc = 0.0;

    //while( 1 )
    while (c < 100)
    {
        // check the current tetrahedra
        d = i * iStep + j * jStep + k;
        xP = xCoord + d;
        yP = yCoord + d;
        zP = zCoord + d;

        d_ = ((i % 2) + (j % 2) + (k % 2)) % 2;
        // d_ = decompositionOrder(i, j, k);
        ptrTbl = tetraTable[d_][t];

        d2 = ptrTbl[0];
        p0[0] = *(xP + d2);
        p0[1] = *(yP + d2);
        p0[2] = *(zP + d2);

        d2 = ptrTbl[1];
        p1[0] = *(xP + d2);
        p1[1] = *(yP + d2);
        p1[2] = *(zP + d2);

        d2 = ptrTbl[2];
        p2[0] = *(xP + d2);
        p2[1] = *(yP + d2);
        p2[2] = *(zP + d2);

        d2 = ptrTbl[3];
        p3[0] = *(xP + d2);
        p3[1] = *(yP + d2);
        p3[2] = *(zP + d2);

        wg_ = fabsf(tetraVolume(p0, p1, p2, p3));
        w0_ = fabsf(tetraVolume(px, p1, p2, p3)) / wg_;
        w1_ = fabsf(tetraVolume(p0, px, p2, p3)) / wg_;
        w2_ = fabsf(tetraVolume(p0, p1, px, p3)) / wg_;
        w3_ = fabsf(tetraVolume(p0, p1, p2, px)) / wg_;

        //cerr << " wg = " << wg << endl;

        if (w0_ + w1_ + w2_ + w3_ <= 1.0001) // oben war w0/wg etc.
        //if( (w0_+w1_+w2_+w3_)/wg_ <= 1.001 )
        {
            // found !
            //         cerr << "     found after   " << c << "    tries" << endl;
            return (c ? 1 : -1);
        }

        // maybe the tetrahedra is degenerated...so check that
        //  (add it later if it's necessary...simply check for wg==0.0)

        curI = i;
        curJ = j;
        curK = k;
        curT = t;

        // find next tetrahedra on our way to px
        // 1. compute which side we have to pass
        r = this->tetraTrace(p0, p1, p2, p3, px);

        //this->debugTetra(p0, p1, p2, p3);

        //      cerr << "   left through side " << r << "  (d=" << d << ")";
        // 2. lookup the neighbour on that side
        m = t;
        ptrTbl = neighbourTable[d_][m][r];
        i += ptrTbl[0];
        j += ptrTbl[1];
        k += ptrTbl[2];
        t = ptrTbl[3];
        c++;
        dc += 1.0;

        //      cerr << "     continuing with:  (" << i << "," << j << "," << k << ":" << t << ")";
        //      cerr << "   " << i*iStep+j*jStep+k*kStep << endl;

        // maybe we are out-of-bounds
        if (i >= iDim - 1 || i < 0 || j >= jDim - 1 || j < 0 || k >= kDim - 1 || k < 0)
        {
            // we left this block
            i -= ptrTbl[0];
            j -= ptrTbl[1];
            k -= ptrTbl[2];
            t = m;

            return (0);
        }

        // maybe we were sent back to where we came from...
        if (i == lastI && j == lastJ && k == lastK && t == lastT)
        {
            //cerr << "troet" << endl;

            // THIS AGGRESSION WILL NOT STAND !
            i -= ptrTbl[0];
            j -= ptrTbl[1];
            k -= ptrTbl[2];
            t = m;

            lastI = i;
            lastJ = j;
            lastK = k;
            lastT = t;

            // tetraTrace told us to pass via r
            // ...so we try the three other sides and
            //   jump into the biggest neighbour cell
            // (or return an error if there is no other neighbour)
            lMax = -1;
            vMax = -1.0;
            for (l = 0; l < 4; l++)
            {
                if (l != r)
                {
                    m = t;
                    ptrTbl = neighbourTable[d_][m][l];
                    i += ptrTbl[0];
                    j += ptrTbl[1];
                    k += ptrTbl[2];
                    t = ptrTbl[3];

                    if (i >= iDim - 1 || i < 0 || j >= jDim - 1 || j < 0 || k >= kDim - 1 || k < 0)
                    {
                        // we left this block
                        i -= ptrTbl[0];
                        j -= ptrTbl[1];
                        k -= ptrTbl[2];
                        t = m;
                    }
                    else
                    {
                        d = i * iStep + j * jStep + k;
                        xP = xCoord + d;
                        yP = yCoord + d;
                        zP = zCoord + d;

                        d_ = ((i % 2) + (j % 2) + (k % 2)) % 2;
                        // d_ = decompositionOrder(i, j, k);
                        ptrTbl = tetraTable[d_][t];

                        d2 = ptrTbl[0];
                        p0[0] = *(xP + d2);
                        p0[1] = *(yP + d2);
                        p0[2] = *(zP + d2);

                        d2 = ptrTbl[1];
                        p1[0] = *(xP + d2);
                        p1[1] = *(yP + d2);
                        p1[2] = *(zP + d2);

                        d2 = ptrTbl[2];
                        p2[0] = *(xP + d2);
                        p2[1] = *(yP + d2);
                        p2[2] = *(zP + d2);

                        d2 = ptrTbl[3];
                        p3[0] = *(xP + d2);
                        p3[1] = *(yP + d2);
                        p3[2] = *(zP + d2);

                        wg_ = fabsf(tetraVolume(p0, p1, p2, p3));

                        if (wg_ > vMax)
                        {
                            vMax = wg_;
                            lMax = l;
                        }

                        ptrTbl = neighbourTable[d_][m][l];
                        i -= ptrTbl[0];
                        j -= ptrTbl[1];
                        k -= ptrTbl[2];
                        t = m;
                    }
                }
            }

            // error ?
            if (lMax == -1)
            {
                cerr << "elvis seems to have left the grid" << endl;
                return (0);
            }

            m = t;
            ptrTbl = neighbourTable[d_][m][lMax];
            i += ptrTbl[0];
            j += ptrTbl[1];
            k += ptrTbl[2];
            t = ptrTbl[3];
        }
        else
        {
            lastI = curI;
            lastJ = curJ;
            lastK = curK;
            lastT = curT;
        }

        /*      if( c>5 )
            {

               return( bruteForceSearch(x, y, z, i, j, k, t) );
            } */
    }

    //   cerr << " c OVERFLOW !!! " << c << "   cell: " << i << "," << j << "," << k << "    dim: ";
    //   cerr << iDim << "," << jDim << "," << kDim << endl;
    if (g_numElem == 0)
    {
        //      this->debugTetra(p0, p1, p2, p3);
        //      this->debugTetra(x, y, z, 0.05);
    }

    return (0);
}

int StrctBlock::advanceGBI(GridBlockInfo *gbi, float dx, float dy, float dz)
{
    //int r;

    // simply update the position and get the new velocity...this will also advance in the grid
    //   gbi->x += dx;
    //   gbi->y += dy;
    //   gbi->z += dz;

    //cerr << "StrctBlock::advanceGBI" << endl;

    //float dc;
    //r = this->getVelocity(gbi, dx, dy, dz, &gbi->u, &gbi->v, &gbi->w, dc);
    gbi->x += dx;
    gbi->y += dy;
    gbi->z += dz;
    ((StrctBlockInfo *)gbi)->i = ((StrctBlockInfo *)gbi)->iTmp;
    ((StrctBlockInfo *)gbi)->j = ((StrctBlockInfo *)gbi)->jTmp;
    ((StrctBlockInfo *)gbi)->k = ((StrctBlockInfo *)gbi)->kTmp;

    return (0);

    // diese funktion muss nur noch x+dx usw. machen und die jeweiligen
    //  temporaeren variablen uebernehmen, da diese noch vom letzten getVelocity
    //  gueltig sind

    //   cerr << "advanceGBI will return  " << ((!r)?0:1) << endl;

    //return( (!r)?0:1 );
}

int StrctBlock::decompositionOrder(int i, int j, int k)
{
    // this formula is kinda tricky to get...but if you paint some pictures and
    //  make a table with all possible combinations of odd and even, you get it :)
    return (((i % 2) + (j % 2) + (k % 2)) % 2);
}

int StrctBlock::bruteForceSearch(float x, float y, float z, int &i, int &j, int &k, int &t)
{
    int _i, _j, _k, _o = 0, m, d, d2;
    float sbx0, sby0, sbz0, sbx1, sby1, sbz1; // search-box (..0 < ..1)
    float xMin, xMax, yMin, yMax, zMin, zMax;
    float *xP, *yP, *zP, v;
    float p0[3], p1[3], p2[3], p3[3], px[3];
    float wg, w0, w1, w2, w3;
    float dc;
    // int _t,

    int c = 0;

    //   cerr << "   strct::bFS:  searching for " << x << " " << y << " " << z << "    in block: " << myBlock << "   step: " << step->stepNo() << endl;

    sbx0 = x - searchSizeX;
    sbx1 = x + searchSizeX;
    sby0 = y - searchSizeY;
    sby1 = y + searchSizeY;
    sbz0 = z - searchSizeZ;
    sbz1 = z + searchSizeZ;

    px[0] = x;
    px[1] = y;
    px[2] = z;

    for (_i = 0; _i < iDim - 1; _i++)
    {
        for (_j = 0; _j < jDim - 1; _j++)
        {
            for (_k = 0; _k < kDim - 1; _k++)
            {
                // (_k<kDim-1 && _j<jDim-1 && _i<iDim-1) &&
                //if( (_i!=iDim-1 && _j!=jDim-1 && _k!=kDim-1) && (xCoord[_o]>=sbx0 && xCoord[_o]<=sbx1 && yCoord[_o]>=sby0 && yCoord[_o]<=sby1 && zCoord[_o]>=sbz0 && zCoord[_o]<=sbz1) )
                if (xCoord[_o] >= sbx0 && xCoord[_o] <= sbx1 && yCoord[_o] >= sby0 && yCoord[_o] <= sby1 && zCoord[_o] >= sbz0 && zCoord[_o] <= sbz1)
                {
                    c++;
                    if (c == 1)
                    {
                        // this is the first cell we found -> try tracing to the point we are searching for
                        i = _i;
                        j = _j;
                        k = _k;
                        t = 0;
                        d = this->findNextCell(x, y, z, i, j, k, t, dc);

                        //cerr << "+";

                        // if we found it, return success, otherwise continue brute force search
                        //   (may be changed to fail here but we'll see)
                        if (d == 1 || d == -1)
                            return (1);
                    }

                    xP = xCoord + _o;
                    yP = yCoord + _o;
                    zP = zCoord + _o;
                    // first: check boundingbox of this cell with the point
                    xMin = *xP;
                    yMin = *yP;
                    zMin = *zP;
                    xMax = xMin;
                    yMax = yMin;
                    zMax = zMin;
                    for (m = 1; m < 8; m++)
                    {
                        v = *(xP + cornerTbl[m]);
                        if (v > xMax)
                            xMax = v;
                        else if (v < xMin)
                            xMin = v;

                        v = *(yP + cornerTbl[m]);
                        if (v > yMax)
                            yMax = v;
                        else if (v < yMin)
                            yMin = v;

                        v = *(zP + cornerTbl[m]);
                        if (v > zMax)
                            zMax = v;
                        else if (v < zMin)
                            zMin = v;
                    }
                    if (x >= xMin && x <= xMax && y >= yMin && y <= yMax && z >= zMin && z <= zMax)
                    {
                        // second: perform tetrahedra decomposition and see if the point is
                        //    inside this cell
                        d = ((_i % 2) + (_j % 2) + (_k % 2)) % 2;
                        //d = decompositionOrder(_i, _j, _k);
                        for (m = 0; m < 5; m++)
                        {
                            // check this tetrahedra
                            d2 = tetraTable[d][m][0];
                            p0[0] = *(xP + d2);
                            p0[1] = *(yP + d2);
                            p0[2] = *(zP + d2);

                            d2 = tetraTable[d][m][1];
                            p1[0] = *(xP + d2);
                            p1[1] = *(yP + d2);
                            p1[2] = *(zP + d2);

                            d2 = tetraTable[d][m][2];
                            p2[0] = *(xP + d2);
                            p2[1] = *(yP + d2);
                            p2[2] = *(zP + d2);

                            d2 = tetraTable[d][m][3];
                            p3[0] = *(xP + d2);
                            p3[1] = *(yP + d2);
                            p3[2] = *(zP + d2);

                            wg = fabsf(tetraVolume(p0, p1, p2, p3));
                            w0 = fabsf(tetraVolume(px, p1, p2, p3)) / wg;
                            w1 = fabsf(tetraVolume(p0, px, p2, p3)) / wg;
                            w2 = fabsf(tetraVolume(p0, p1, px, p3)) / wg;
                            w3 = fabsf(tetraVolume(p0, p1, p2, px)) / wg;

                            if (w0 + w1 + w2 + w3 <= 1.0001)
                            //if( (w0+w1+w2+w3)/wg <= 1.001 )
                            {
                                i = _i;
                                //   cerr << "bFS: gefunden in Zelle  " << _i << " " << _j << " " << _k << "   tetra: " << m << "   pos: " << _o << endl;
                                j = _j;
                                k = _k;
                                t = m;

                                //cerr << "x";

                                return (1);
                            }

                            // NOTE: we might cache the volumes for speedup, check later if that
                            //  improves performance or not
                        }
                    }
                }

                _o++;
            }
            _o++;
        }
        _o += kDim;
    }

    //   cerr << "bFS: not found" << endl;

    return (0);
}

float StrctBlock::proposeIncrement(GridBlockInfo *gbi)
{
    int d, d2, t;
    float *xP, *yP, *zP;
    float p0[3], p1[3], p2[3], p3[3];
    float vol, vel;
    float ds, dt;

    t = ((StrctBlockInfo *)gbi)->tetra;

    d = ((StrctBlockInfo *)gbi)->i * iStep + ((StrctBlockInfo *)gbi)->j * jStep + ((StrctBlockInfo *)gbi)->k;
    xP = xCoord + d;
    yP = yCoord + d;
    zP = zCoord + d;

    d = decompositionOrder(((StrctBlockInfo *)gbi)->i, ((StrctBlockInfo *)gbi)->j, ((StrctBlockInfo *)gbi)->k);

    d2 = tetraTable[d][t][0];
    p0[0] = *(xP + d2);
    p0[1] = *(yP + d2);
    p0[2] = *(zP + d2);

    d2 = tetraTable[d][t][1];
    p1[0] = *(xP + d2);
    p1[1] = *(yP + d2);
    p1[2] = *(zP + d2);

    d2 = tetraTable[d][t][2];
    p2[0] = *(xP + d2);
    p2[1] = *(yP + d2);
    p2[2] = *(zP + d2);

    d2 = tetraTable[d][t][3];
    p3[0] = *(xP + d2);
    p3[1] = *(yP + d2);
    p3[2] = *(zP + d2);

    // we use 3rd-root to compute border-length of a volume-identic cube
    vol = fabsf(tetraVolume(p0, p1, p2, p3));
    ds = (float)powf(vol, 1.0f / 3.0f);

    // and compute velocity
    vel = sqrt((((StrctBlockInfo *)gbi)->u * ((StrctBlockInfo *)gbi)->u) + (((StrctBlockInfo *)gbi)->v * ((StrctBlockInfo *)gbi)->v) + (((StrctBlockInfo *)gbi)->w * ((StrctBlockInfo *)gbi)->w));

    // dt = ds/v
    if (vel == 0.0)
        dt = 0.0;
    else
        dt = ds / vel;

    // done
    return (dt);
}
