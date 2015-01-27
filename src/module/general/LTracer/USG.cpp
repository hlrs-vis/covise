/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "USG.h"

USGBlock::USGBlock(GridStep *parent, Grid *owner, int b)
    : GridBlock(parent, owner, b)
{
}

USGBlock::~USGBlock()
{
    if (aUData)
        delete[] aUData;
    if (aVData)
        delete[] aVData;
    if (aWData)
        delete[] aWData;
}

int USGBlock::initialize(coDistributedObject *g, coDistributedObject *d, int sS, int bBa, float attA)
{
    usGrid = (coDoUnstructuredGrid *)g;
    usGrid->getGridSize(&numElem, &numConn, &numPoints);
    usGrid->getAddresses(&elemList, &connList, &xCoord, &yCoord, &zCoord);
    usGrid->getTypeList(&typeList);
    usGrid->getNeighborList(&numNeighbors, &neighborList, &neighborIndexList);

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

    // some lookup-tables for speedup

    // compute bounding-box (of entire block and of largest found cell inside the block)
    int i, k, n;
    float x0, y0, z0, x1, y1, z1;

    xMin = xCoord[0];
    yMin = yCoord[0];
    zMin = zCoord[0];
    xMax = xCoord[0];
    yMax = yCoord[0];
    zMax = zCoord[0];
    searchSizeX = 0.0;
    searchSizeY = 0.0;
    searchSizeZ = 0.0;
    for (i = 0; i < numElem; i++)
    {
        if (i == numElem - 1)
            n = numConn;
        else
            n = elemList[i + 1];
        x0 = x1 = xCoord[connList[elemList[i]]];
        y0 = y1 = yCoord[connList[elemList[i]]];
        z0 = z1 = zCoord[connList[elemList[i]]];
        for (k = elemList[i] + 1; k < n; k++)
        {
            if (xCoord[connList[k]] > x1)
                x1 = xCoord[connList[k]];
            else if (xCoord[connList[k]] < x0)
                x0 = xCoord[connList[k]];

            if (yCoord[connList[k]] > y1)
                y1 = yCoord[connList[k]];
            else if (yCoord[connList[k]] < y0)
                y0 = yCoord[connList[k]];

            if (zCoord[connList[k]] > z1)
                z1 = zCoord[connList[k]];
            else if (zCoord[connList[k]] < z0)
                z0 = zCoord[connList[k]];

            if (x0 < xMin)
                xMin = x0;
            if (x1 > xMax)
                xMax = x1;
            if (y0 < yMin)
                yMin = y0;
            if (y1 > yMax)
                yMax = y1;
            if (z0 < zMin)
                zMin = z0;
            if (z1 > zMax)
                zMax = z1;

            if (x1 - x0 > searchSizeX)
                searchSizeX = x1 - x0;
            if (y1 - y0 > searchSizeY)
                searchSizeY = y1 - y0;
            if (z1 - z0 > searchSizeZ)
                searchSizeZ = z1 - z0;
        }
    }

    cerr << "BoundingBox: " << xMin << " " << yMin << " " << zMin << "  -  " << xMax << " " << yMax << " " << zMax << endl;
    cerr << "SearchSize: " << searchSizeX << " " << searchSizeY << " " << searchSizeZ << endl;

    // adjust searchsize according to size of boundingbox to get better values
    //  how ?
    //searchSizeX *= 5.0;
    //searchSizeY *= 5.0;
    //searchSizeZ *= 5.0;

    // done
    if (!uData || !vData || !wData)
        return (0);
    return (1);
}

extern long int d_initGBI;

int USGBlock::initGBI(float x, float y, float z, GridBlockInfo *&gbi)
{
    int d = 0;

    d_initGBI++;

    this->debugTetra(x, y, z, 0.002f);

    // check boundingbox
    if (x < xMin || x > xMax || y < yMin || y > yMax || z < zMin || z > zMax)
        return (0);

    // cerr << "initGBI: searching for point (bbox hit)  " << x << " " << y << " " << z << endl;
    float pos[3];
    pos[0] = x;
    pos[1] = y;
    pos[2] = z;
    // search using octTree
    int fe = usGrid->getCell(pos, 0.002f);
    int i = 0;
    findNextElem(x, y, z, fe, i, d);
    /*

   // start searching using a searchbox
   int e, i, fe=-1;
   float sbx0, sbx1, sby0, sby1, sbz0, sbz1;
   sbx0 = x-searchSizeX;
   sbx1 = x+searchSizeX;
   sby0 = y-searchSizeY;
   sby1 = y+searchSizeY;
   sbz0 = z-searchSizeZ;
   sbz1 = z+searchSizeZ;

   for( e=0; e<numElem; e++ )
   {
      i = connList[elemList[e]];
      //      if( e==39996 )
      //      {
      //         cerr << "  loop: 39996" << endl;
      //         this->debugTetra(xCoord[i], yCoord[i], zCoord[i], 0.01);
      //      }

      if( xCoord[i]>sbx0 && xCoord[i]<sbx1 )
      {
         if( yCoord[i]>sby0 && yCoord[i]<sby1 )
         {
            if( zCoord[i]>sbz0 && zCoord[i]<sbz1 )
            {
               //if( e==39996 )
               //{
               //   cerr << "  loop: inside searchbox" << endl;
               //   cerr << "        isInsideCell 0: " << isInsideCell(x, y, z, e, 0) << endl;
               //   cerr << "        isInsideCell 1: " << isInsideCell(x, y, z, e, 1) << endl;
               //}
               n++;
               fe = e;
               i = 0;
               if( findNextElem(x, y, z, fe, i, d)!=0 )
               {
                  //cerr << "  found in cell " << fe << endl;
                  e = numElem;
               }
               else
                  fe = -1;
            }
         }
      }
   }
*/
    // cerr << "  Point " << x << " " << y << " " << z << " inside BBox   (" << n << " cells out of " << numElem << ")" << endl;

    if (fe != -1)
    {
        // found
        gbi = new USGBlockInfo();
        gbi->x = x;
        gbi->y = y;
        gbi->z = z;
        gbi->block = this;
        gbi->step = step;
        gbi->stepNo = step->stepNo();
        gbi->blockNo = this->blockNo();
        ((USGBlockInfo *)gbi)->cell = fe;
        ((USGBlockInfo *)gbi)->tetra = i;
        ((USGBlockInfo *)gbi)->decomposeOrder = d;

        float dc;

        this->getVelocity(gbi, 0.0, 0.0, 0.0, &gbi->u, &gbi->v, &gbi->w, dc);
        // cerr << "initGBI - succeeded" << endl;

        //      cerr << "  vel: " << gbi->u << "  " << gbi->v << "  " << gbi->w << endl;

        //this->debugTetra(x, y, z, 0.006);

        return (1);
    }
    //this->debugTetra(x, y, z, 0.002);  uncomment to see failed initGBI calls
    //cerr << "initGBI - failed" << endl;

    return (0);
}

extern long int d_getVel;

int USGBlock::getVelocity(GridBlockInfo *gbi, float dx, float dy, float dz, float *u, float *v, float *w, float &dc)
{
    //   cerr << "getVelocity, calling fNE" << endl;
    d_getVel++;

    dc = 0.0;

    if (!findNextElem(gbi->x + dx, gbi->y + dy, gbi->z + dz, ((USGBlockInfo *)gbi)->cell, ((USGBlockInfo *)gbi)->tetra, ((USGBlockInfo *)gbi)->decomposeOrder))
    {
        //      cerr << "fNE failed for cell " << ((USGBlockInfo *)gbi)->cell << endl;

        this->debugTetra(gbi->x + dx, gbi->y + dy, gbi->z + dz, 0.0005f);

        //cerr << "isInsideCell: " << this->isInsideCell(gbi->x+dx,gbi->y+dy,gbi->z+dz,((USGBlockInfo *)gbi)->cell,((USGBlockInfo *)gbi)->decomposeOrder) << endl;
        //      cerr << "==== isInsideCell: ====>" << endl;
        //      this->isInsideCell(gbi->x+dx,gbi->y+dy,gbi->z+dz,((USGBlockInfo *)gbi)->cell,((USGBlockInfo *)gbi)->decomposeOrder);
        //      cerr << "=======================<" << endl;

        //cerr << "isInsideCell(0): " << this->isInsideCell(gbi->x+dx,gbi->y+dy,gbi->z+dz,((USGBlockInfo *)gbi)->cell,0) << endl;
        //cerr << "isInsideCell(1): " << this->isInsideCell(gbi->x+dx,gbi->y+dy,gbi->z+dz,((USGBlockInfo *)gbi)->cell,1) << endl;

        // we left through side val[3] so check whether the andgle is greater than the atta parameter

        if (attackAngle == 0.0)
            return (0);

        // compute actual attackAngle

        // attack-vector
        float vel[3], l;
        float normal[3];
        float va[3], vb[3];
        l = sqrt(gbi->u * gbi->u + gbi->v * gbi->v + gbi->w * gbi->w);
        vel[0] = gbi->u / l;
        vel[1] = gbi->v / l;
        vel[2] = gbi->w / l;
        va[0] = xCoord[val[0]] - xCoord[val[1]];
        va[1] = yCoord[val[0]] - yCoord[val[1]];
        va[2] = zCoord[val[0]] - zCoord[val[1]];
        vb[0] = xCoord[val[0]] - xCoord[val[2]];
        vb[1] = yCoord[val[0]] - yCoord[val[2]];
        vb[2] = zCoord[val[0]] - zCoord[val[2]];
        normal[0] = va[1] * vb[2] - va[2] * vb[1];
        normal[1] = va[2] * vb[0] - va[0] * vb[2];
        normal[2] = va[0] * vb[1] - va[1] * vb[0];
        float nl = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
        normal[0] /= nl;
        normal[1] /= nl;
        normal[2] /= nl;

        float a = vel[0] * normal[0] + vel[1] * normal[1] + vel[2] * normal[2];

        a = fabs((float)acos(a));
        if (a > M_PI / 2.0)
            a = (float)(M_PI - a);

        if (a > attackAngle)
        {
            // calculate faked velocity
            // there are 2 possibilities here, the first would be simply to drop the
            //   velocity component that would cause us to leave the grid. the other would
            //   be to not only drop it but to make sure that the magnitude doesn't change.
            //   Has to be determined through some tests which method makes more sense.

            // it's not really nice (check it with TGenDat) but anyhow this case shouldn't
            //   happen too often so this should be a sufficient solution

            a = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
            normal[0] /= a;
            normal[1] /= a;
            normal[2] /= a;
            float sp = normal[0] * vel[0] + normal[1] * vel[1] + normal[2] * vel[2];

            vel[0] = (vel[0] - normal[0] * sp);
            vel[1] = (vel[1] - normal[1] * sp);
            vel[2] = (vel[2] - normal[2] * sp);
            a = sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
            // adapt the length
            *u = vel[0] / a * l;
            *v = vel[1] / a * l;
            *w = vel[2] / a * l;

            // done
            return (2);
        }

        return (0); // not found
    }

    int o = ((USGBlockInfo *)gbi)->tetra * 4;

    //   cerr << "cell: " << ((USGBlockInfo *)gbi)->cell << "   o: " << o/4 << endl;

    // hier wird auch 2x die geschwindigkeit berechnet...das kann man auch auf 1x reduzieren
    //   (in findNextElem und dann in isInsideTetra)

    if (isInsideTetra(tetraConnList[o], tetraConnList[o + 1], tetraConnList[o + 2], tetraConnList[o + 3], gbi->x + dx, gbi->y + dy, gbi->z + dz, *u, *v, *w))
    {
        // das kann eigentlich nicht passieren, ausser in findNextElem ist ein bug !!!
        cerr << "BUG ! BUG ! BUG ! - check USGBlock:getVelocity / findNextElem" << endl;
    }

    //   cerr << "  vel: " << *u << "  " << *v << "  " << *w << endl;

    /*
   // debugTetra basteln
   float p0[3], p1[3], p2[3], p3[3];
   p0[0] = xCoord[tetraConnList[o]];
   p0[1] = yCoord[tetraConnList[o]];
   p0[2] = zCoord[tetraConnList[o]];

   p1[0] = xCoord[tetraConnList[o+1]];
   p1[1] = yCoord[tetraConnList[o+1]];
   p1[2] = zCoord[tetraConnList[o+1]];

   p2[0] = xCoord[tetraConnList[o+2]];
   p2[1] = yCoord[tetraConnList[o+2]];
   p2[2] = zCoord[tetraConnList[o+2]];

   p3[0] = xCoord[tetraConnList[o+3]];
   p3[1] = yCoord[tetraConnList[o+3]];
   p3[2] = zCoord[tetraConnList[o+3]];
   this->debugTetra(p0, p1, p2, p3);
   */

    // done
    return (1);
}

int USGBlock::advanceGBI(GridBlockInfo *gbi, float dx, float dy, float dz)
{

    gbi->x += dx;
    gbi->y += dy;
    gbi->z += dz;

    return (1);
}

extern long int d_fnc;

int USGBlock::findNextElem(float x, float y, float z, int &e, int &t, int &dc)
{
    int c = -1, o, k, s, f, tSkip;
    //int n, i1, i2, i3;
    int numN1, numN2, numN3, n1, n2, n3;
    //float d, dt;
    float p0[3], p1[3], p2[3], p3[3], px[3];

    int debugNum = 0;

    px[0] = x;
    px[1] = y;
    px[2] = z;

    d_fnc++;
    f = 1;

    /* ?!? so funktioniert es...
   decomposeElem(3285, 0);
   p0[0] = xCoord[tetraConnList[0]];
   p0[1] = yCoord[tetraConnList[0]];
   p0[2] = zCoord[tetraConnList[0]];

   p1[0] = xCoord[tetraConnList[1]];
   p1[1] = yCoord[tetraConnList[1]];
   p1[2] = zCoord[tetraConnList[1]];

   p2[0] = xCoord[tetraConnList[2]];
   p2[1] = yCoord[tetraConnList[2]];
   p2[2] = zCoord[tetraConnList[2]];

   p3[0] = xCoord[tetraConnList[3]];
   p3[1] = yCoord[tetraConnList[3]];
   p3[2] = zCoord[tetraConnList[3]];
   debugTetra(p0, p1, p2, p3);

   p0[0] = xCoord[tetraConnList[8]];
   p0[1] = yCoord[tetraConnList[8]];
   p0[2] = zCoord[tetraConnList[8]];

   p1[0] = xCoord[tetraConnList[9]];
   p1[1] = yCoord[tetraConnList[9]];
   p1[2] = zCoord[tetraConnList[9]];

   p2[0] = xCoord[tetraConnList[10]];
   p2[1] = yCoord[tetraConnList[10]];
   p2[2] = zCoord[tetraConnList[10]];

   p3[0] = xCoord[tetraConnList[11]];
   p3[1] = yCoord[tetraConnList[11]];
   p3[2] = zCoord[tetraConnList[11]];
   debugTetra(p0, p1, p2, p3);
   */

    tSkip = -1;

    while (c && debugNum < 100)
    {
        // 1. check current (e) if x/y/z is inside
        if (f)
            o = isInsideCell(x, y, z, e, dc);
        else
            o = -1;
        f = 1;
        if (o >= 0)
        {
            t = o;
            return (c);
        }

        // so we need to find the next element
        c = 1;

        // sicherheitshalber...sollte unnoetig sein, spaeter wieder rausnehmen
        //decomposeElem(e, dc);

        // we assume to begin tracing from tetrahedra (e,t)
        p0[0] = xCoord[tetraConnList[t * 4]];
        p0[1] = yCoord[tetraConnList[t * 4]];
        p0[2] = zCoord[tetraConnList[t * 4]];

        p1[0] = xCoord[tetraConnList[(t * 4) + 1]];
        p1[1] = yCoord[tetraConnList[(t * 4) + 1]];
        p1[2] = zCoord[tetraConnList[(t * 4) + 1]];

        p2[0] = xCoord[tetraConnList[(t * 4) + 2]];
        p2[1] = yCoord[tetraConnList[(t * 4) + 2]];
        p2[2] = zCoord[tetraConnList[(t * 4) + 2]];

        p3[0] = xCoord[tetraConnList[(t * 4) + 3]];
        p3[1] = yCoord[tetraConnList[(t * 4) + 3]];
        p3[2] = zCoord[tetraConnList[(t * 4) + 3]];

        //if( e==3285 && t==2 )
        //   cerr << endl << "************" << endl;

        s = tetraTrace(p0, p1, p2, p3, px);

        /*
      if( e==3430 )
      {
         if( (t==0 || t==2) && dc==0 )
         {
            debugTetra(p0, p1, p2, p3);
            float pc[3];
            pc[0] = (p0[0]+p1[0]+p2[0]+p3[0])/4.0;
            pc[1] = (p0[1]+p1[1]+p2[1]+p3[1])/4.0;
            pc[2] = (p0[2]+p1[2]+p2[2]+p3[2])/4.0;
            debugTetra(pc[0], pc[1], pc[2], 0.001);
      debugTetra(pc, pc, pc, px);

      }
      cerr << "tetra: " << t << "    dc: " << dc << endl;
      }

      if( e==3430 )
      cerr << endl << "************  " << s << endl;
      */
        //cerr << "elem: " << e << "    tetra: " << t << "    dc: " << dc << endl;

        // load neighbors we have to check...the side we left is defined through vertices val[]
        switch (s)
        {
        case 0:
            val[0] = tetraConnList[t * 4];
            val[1] = tetraConnList[t * 4 + 1];
            val[2] = tetraConnList[t * 4 + 2];
            break;
        case 1:
            val[0] = tetraConnList[t * 4];
            val[1] = tetraConnList[t * 4 + 1];
            val[2] = tetraConnList[t * 4 + 3];
            break;
        case 2:
            val[0] = tetraConnList[t * 4];
            val[1] = tetraConnList[t * 4 + 2];
            val[2] = tetraConnList[t * 4 + 3];
            break;
        case 3:
            val[0] = tetraConnList[t * 4 + 1];
            val[1] = tetraConnList[t * 4 + 2];
            val[2] = tetraConnList[t * 4 + 3];
            break;
        }

        // check if there is another tetrahedra inside this cell which shares these vertices
        //  if yes -> continue with that tetrahedra inside current cell
        //  if no -> find a common neighbour-cell to all these vertices
        c = 0;
        for (o = 0; o < numTetra; o++)
        {
            if (o != t)
            {
                k = o * 4;
                if (tetraConnList[k] == val[0] || tetraConnList[k + 1] == val[0] || tetraConnList[k + 2] == val[0] || tetraConnList[k + 3] == val[0])
                {
                    if (tetraConnList[k] == val[1] || tetraConnList[k + 1] == val[1] || tetraConnList[k + 2] == val[1] || tetraConnList[k + 3] == val[1])
                    {
                        if (tetraConnList[k] == val[2] || tetraConnList[k + 1] == val[2] || tetraConnList[k + 2] == val[2] || tetraConnList[k + 3] == val[2])
                        {
                            //if( o!=tSkip )
                            //{
                            //   tSkip = t;
                            c = 1;
                            break;
                            //}
                        }
                    }
                }
            }
        }

        if (c)
        {
            t = o;
            f = 0;
        }
        else
        {
            // try to find a common neighbor for these 3 vertices
            // init: numN#=number of neighbors on that vertex
            if (val[0] == numConn - 1)
                numN1 = numNeighbors - neighborIndexList[val[0]];
            else
                numN1 = neighborIndexList[val[0] + 1] - neighborIndexList[val[0]];
            if (val[1] == numConn - 1)
                numN2 = numNeighbors - neighborIndexList[val[1]];
            else
                numN2 = neighborIndexList[val[1] + 1] - neighborIndexList[val[1]];
            if (val[2] == numConn - 1)
                numN3 = numNeighbors - neighborIndexList[val[2]];
            else
                numN3 = neighborIndexList[val[2] + 1] - neighborIndexList[val[2]];

            //cerr << "n " << numN1 << " " << numN2 << " " << numN3 << endl;

            // searching...
            for (n1 = neighborIndexList[val[0]]; n1 < neighborIndexList[val[0]] + numN1 && !c; n1++)
            {
                for (n2 = neighborIndexList[val[1]]; n2 < neighborIndexList[val[1]] + numN2 && !c; n2++)
                {
                    for (n3 = neighborIndexList[val[2]]; n3 < neighborIndexList[val[2]] + numN3 && !c; n3++)
                    {
                        if (neighborList[n1] == neighborList[n2] && neighborList[n1] == neighborList[n3])
                        {
                            if (neighborList[n1] != e)
                            {
                                // make sure this is a 3D element !
                                e = neighborList[n1];
                                //if( typeList[e]==TYPE_HEXAGON || typeList[e]==TYPE_PYRAMID || typeList[e]==TYPE_PRISM || typeList[e]==TYPE_TETRAHEDER )
                                {
                                    c = 1;
                                    dc = getDecomposition(e, val[0], val[1], val[2]);

                                    //cerr << "cont with elem: " << e << "    dc: " << dc << endl;
                                }
                            }
                        }
                    }
                }
            }
        }

        debugNum++;

        // continue (c=1) or abort (c=0)
    }

    /*   if( debugNum==100 )
         cerr << " aborted (USG::findNextElem)" << endl;
   */

    return (0);
}

int USGBlock::findNextElem_old(float x, float y, float z, int &e, int &t, int &dc)
{
    int c = -1, o, n, k, s;
    //int i1, i2, i3;
    int numN1, numN2, numN3, n1, n2, n3;
    int val[3];
    float d, dt;
    float p0[3], p1[3], p2[3], p3[3], px[3];

    int debugNum = 0;

    px[0] = x;
    px[1] = y;
    px[2] = z;

    d_fnc++;

    while (c && debugNum < 30)
    {
        // 1. check current (e) if x/y/z is inside
        o = isInsideCell(x, y, z, e, dc);
        if (o >= 0)
        {
            t = o;
            return (c);
        }

        // so we need to find the next element
        c = 1;

        //cerr << "elem: " << e << endl;

        // determine which vertex is the closest to x/y/z
        switch (typeList[e])
        {
        case TYPE_HEXAGON:
            k = 8;
            break;
        case TYPE_PYRAMID:
            k = 5;
            break;
        case TYPE_PRISM:
            k = 6;
            break;
        case TYPE_TETRAHEDER:
            k = 4;
            break;
        default:
            cerr << "check USGBlock::findNextElem !!" << endl;
        }

        for (o = 0; o < k; o++)
        {
            dt = (x - xCoord[connList[elemList[e] + o]]) * (x - xCoord[connList[elemList[e] + o]]) + (y - yCoord[connList[elemList[e] + o]]) * (y - yCoord[connList[elemList[e] + o]]) + (z - zCoord[connList[elemList[e] + o]]) * (z - zCoord[connList[elemList[e] + o]]);
            if (dt < d || o == 0)
            {
                n = o;
                d = dt;
            }
        }
        // -> the vertex is k
        k = connList[elemList[e] + n];

        // rausfinden welche tetraeder des elements diesen vertex verwenden und eine "ausenseite" besitzen
        // von dessen schwerpunkt in richtung x/y/z tracen und wenn wir dabei ueber eine "ausenseite" gehen
        // dann einen passenden nachbarn suchen. wenn gefunden -> weiter mit dem nachbarn, ansonsten
        // weiter mit dem naechsten passenden tetraeder dieses elements. wenn keine gefunden -> fallback/bruteforce
        //
        // das kann vereinfacht werden, einfach rausfinden welcher tetraeder, dann ueber welche
        //  seite rausgegangen wird, dann einen nachbarn suchen der an allen 3 eckpunkten nachbar ist -> fertig

        c = 0;
        for (o = 0; o < numTetra && !c; o++)
        {
            for (n = 0; n < 4 && !c; n++)
            {
                if (tetraConnList[o * 4 + n] == k)
                {
                    //cerr << "t " << o << endl;

                    n = 4;
                    // matching tetrahedra (o)
                    p0[0] = xCoord[tetraConnList[o * 4]];
                    p0[1] = yCoord[tetraConnList[o * 4]];
                    p0[2] = zCoord[tetraConnList[o * 4]];

                    p1[0] = xCoord[tetraConnList[o * 4 + 1]];
                    p1[1] = yCoord[tetraConnList[o * 4 + 1]];
                    p1[2] = zCoord[tetraConnList[o * 4 + 1]];

                    p2[0] = xCoord[tetraConnList[o * 4 + 2]];
                    p2[1] = yCoord[tetraConnList[o * 4 + 2]];
                    p2[2] = zCoord[tetraConnList[o * 4 + 2]];

                    p3[0] = xCoord[tetraConnList[o * 4 + 3]];
                    p3[1] = yCoord[tetraConnList[o * 4 + 3]];
                    p3[2] = zCoord[tetraConnList[o * 4 + 3]];

                    if (!isInsideTetra(tetraConnList[o * 4], tetraConnList[o * 4 + 1], tetraConnList[o * 4 + 2], tetraConnList[o * 4 + 3], x, y, z, d, d, d))
                        cerr << "MUH !" << endl;

                    s = tetraTrace(p0, p1, p2, p3, px);
                    // load neighbors we have to check
                    switch (s)
                    {
                    case 0:
                        val[0] = tetraConnList[o * 4];
                        val[1] = tetraConnList[o * 4 + 1];
                        val[2] = tetraConnList[o * 4 + 2];
                        break;
                    case 1:
                        val[0] = tetraConnList[o * 4];
                        val[1] = tetraConnList[o * 4 + 1];
                        val[2] = tetraConnList[o * 4 + 3];
                        break;
                    case 2:
                        val[0] = tetraConnList[o * 4];
                        val[1] = tetraConnList[o * 4 + 2];
                        val[2] = tetraConnList[o * 4 + 3];
                        break;
                    case 3:
                        val[0] = tetraConnList[o * 4 + 1];
                        val[1] = tetraConnList[o * 4 + 2];
                        val[2] = tetraConnList[o * 4 + 3];
                        break;
                    }

                    // try to find a common neighbor for these 3 vertices
                    // init: numN#=number of neighbors on that vertex
                    if (val[0] == numConn - 1)
                        numN1 = numNeighbors - neighborIndexList[val[0]];
                    else
                        numN1 = neighborIndexList[val[0] + 1] - neighborIndexList[val[0]];
                    if (val[1] == numConn - 1)
                        numN2 = numNeighbors - neighborIndexList[val[1]];
                    else
                        numN2 = neighborIndexList[val[1] + 1] - neighborIndexList[val[1]];
                    if (val[2] == numConn - 1)
                        numN3 = numNeighbors - neighborIndexList[val[2]];
                    else
                        numN3 = neighborIndexList[val[2] + 1] - neighborIndexList[val[2]];

                    //cerr << "n " << numN1 << " " << numN2 << " " << numN3 << endl;

                    // searching...
                    for (n1 = neighborIndexList[val[0]]; n1 < neighborIndexList[val[0]] + numN1 && !c; n1++)
                    {
                        for (n2 = neighborIndexList[val[1]]; n2 < neighborIndexList[val[1]] + numN2 && !c; n2++)
                        {
                            for (n3 = neighborIndexList[val[2]]; n3 < neighborIndexList[val[2]] + numN3 && !c; n3++)
                            {
                                if (neighborList[n1] == neighborList[n2] && neighborList[n1] == neighborList[n3])
                                {
                                    if (neighborList[n1] != e)
                                    {
                                        // make sure this is a 3D element !
                                        e = neighborList[n1];
                                        //if( typeList[e]==TYPE_HEXAGON || typeList[e]==TYPE_PYRAMID || typeList[e]==TYPE_PRISM || typeList[e]==TYPE_TETRAHEDER )
                                        {
                                            c = 1;
                                            dc = getDecomposition(e, val[0], val[1], val[2]);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // if found -> c=1
                }
            }
        }

        debugNum++;

        // continue (c=1) or abort (c=0)
    }

    /*   if( debugNum==30 )
         cerr << " aborted" << endl; */

    return (0);
}

int USGBlock::isInsideCell(float x, float y, float z, int e, int d)
{

    // frage was schneller ist: erst boundingbox der zelle pruefen und dann zerlegen und durch die
    //    tetraeder laufen oder direkt zu zerlegen und durch die tetraeder zu laufen.
    //    ausprobieren !!!
    decomposeElem(e, d);

    int i, o;
    float t;
    for (i = 0; i < numTetra; i++)
    {
        o = i * 4;

        /*
      if( e==8184 && i==2 )
      {

         cerr << tetraConnList[o] << endl;
         cerr << tetraConnList[o+1] << endl;
         cerr << tetraConnList[o+2] << endl;
         cerr << tetraConnList[o+3] << endl;

      float p0[3],p1[3],p2[3],p3[3];
      p0[0] = xCoord[tetraConnList[o]];
      p0[1] = yCoord[tetraConnList[o]];
      p0[2] = zCoord[tetraConnList[o]];

      p1[0] = xCoord[tetraConnList[o+1]];
      p1[1] = yCoord[tetraConnList[o+1]];
      p1[2] = zCoord[tetraConnList[o+1]];

      p2[0] = xCoord[tetraConnList[o+2]];
      p2[1] = yCoord[tetraConnList[o+2]];
      p2[2] = zCoord[tetraConnList[o+2]];

      p3[0] = xCoord[tetraConnList[o+3]];
      p3[1] = yCoord[tetraConnList[o+3]];
      p3[2] = zCoord[tetraConnList[o+3]];
      this->debugTetra(p0, p1, p2, p3);
      this->debugTetra(x, y, z, 0.001);

      }
      */

        if (!isInsideTetra(tetraConnList[o], tetraConnList[o + 1], tetraConnList[o + 2], tetraConnList[o + 3], x, y, z, t, t, t))
            return (i);
    }

    return (-1);
}

int USGBlock::getDecomposition(int e, int v1, int v2, int v3)
{
    int t, n, i;
    decomposeElem(e, 0);
    n = 0;
    for (t = 0; t < numTetra && n != 3; t++)
    {
        n = 0;
        for (i = t * 4; i < (t + 1) * 4; i++)
        {
            if (tetraConnList[i] == v1 || tetraConnList[i] == v2 || tetraConnList[i] == v3)
                n++;
        }
    }
    if (n == 3)
        return (0);
    return (1);
}

// note: decomposition schemes are taken from TETRAHEDRONIZE::Grid.cpp
void USGBlock::decomposeElem(int e, int d)
{
    int c = elemList[e];
    switch (typeList[e])
    {
    case TYPE_HEXAGON:
        numTetra = 6;
        if (!d)
        {
            tetraConnList[0] = connList[c + 4];
            tetraConnList[1] = connList[c + 6];
            tetraConnList[2] = connList[c + 5];
            tetraConnList[3] = connList[c];

            tetraConnList[4] = connList[c + 2];
            tetraConnList[5] = connList[c];
            tetraConnList[6] = connList[c + 1];
            tetraConnList[7] = connList[c + 6];

            tetraConnList[8] = connList[c + 1];
            tetraConnList[9] = connList[c];
            tetraConnList[10] = connList[c + 5];
            tetraConnList[11] = connList[c + 6];

            tetraConnList[12] = connList[c + 4];
            tetraConnList[13] = connList[c + 7];
            tetraConnList[14] = connList[c + 6];
            tetraConnList[15] = connList[c];

            tetraConnList[16] = connList[c + 3];
            tetraConnList[17] = connList[c];
            tetraConnList[18] = connList[c + 2];
            tetraConnList[19] = connList[c + 7];

            tetraConnList[20] = connList[c + 2];
            tetraConnList[21] = connList[c];
            tetraConnList[22] = connList[c + 6];
            tetraConnList[23] = connList[c + 7];
        }
        else
        {
            tetraConnList[0] = connList[c];
            tetraConnList[1] = connList[c + 1];
            tetraConnList[2] = connList[c + 3];
            tetraConnList[3] = connList[c + 4];

            tetraConnList[4] = connList[c + 7];
            tetraConnList[5] = connList[c + 5];
            tetraConnList[6] = connList[c + 4];
            tetraConnList[7] = connList[c + 3];

            tetraConnList[8] = connList[c + 5];
            tetraConnList[9] = connList[c + 1];
            tetraConnList[10] = connList[c + 4];
            tetraConnList[11] = connList[c + 3];

            tetraConnList[12] = connList[c + 1];
            tetraConnList[13] = connList[c + 2];
            tetraConnList[14] = connList[c + 3];
            tetraConnList[15] = connList[c + 5];

            tetraConnList[16] = connList[c + 7];
            tetraConnList[17] = connList[c + 6];
            tetraConnList[18] = connList[c + 5];
            tetraConnList[19] = connList[c + 3];

            tetraConnList[20] = connList[c + 6];
            tetraConnList[21] = connList[c + 2];
            tetraConnList[22] = connList[c + 5];
            tetraConnList[23] = connList[c + 3];
        }
        break;
    case TYPE_PYRAMID:
        //         cerr << "PYRAMID" << endl << endl;
        numTetra = 1;
        if (!d)
        {
            tetraConnList[0] = connList[c];
            tetraConnList[1] = connList[c + 1];
            tetraConnList[2] = connList[c + 2];
            tetraConnList[3] = connList[c + 4];

            tetraConnList[4] = connList[c];
            tetraConnList[5] = connList[c + 2];
            tetraConnList[6] = connList[c + 3];
            tetraConnList[7] = connList[c + 3];
        }
        else
        {
            tetraConnList[0] = connList[c + 1];
            tetraConnList[1] = connList[c + 2];
            tetraConnList[2] = connList[c + 3];
            tetraConnList[3] = connList[c + 4];

            tetraConnList[4] = connList[c];
            tetraConnList[5] = connList[c + 1];
            tetraConnList[6] = connList[c + 3];
            tetraConnList[7] = connList[c + 4];
        }
        break;
    case TYPE_PRISM:
        //         cerr << "PRISM" << endl << endl;
        numTetra = 3;
        if (!d)
        {
            tetraConnList[0] = connList[c + 3];
            tetraConnList[1] = connList[c + 5];
            tetraConnList[2] = connList[c + 4];
            tetraConnList[3] = connList[c];

            tetraConnList[4] = connList[c + 2];
            tetraConnList[5] = connList[c];
            tetraConnList[6] = connList[c + 1];
            tetraConnList[7] = connList[c + 5];

            tetraConnList[8] = connList[c + 1];
            tetraConnList[9] = connList[c];
            tetraConnList[10] = connList[c + 4];
            tetraConnList[11] = connList[c + 5];
        }
        else
        {
            tetraConnList[0] = connList[c];
            tetraConnList[1] = connList[c + 1];
            tetraConnList[2] = connList[c + 2];
            tetraConnList[3] = connList[c + 3];

            tetraConnList[4] = connList[c + 5];
            tetraConnList[5] = connList[c + 4];
            tetraConnList[6] = connList[c + 3];
            tetraConnList[7] = connList[c + 2];

            tetraConnList[8] = connList[c + 4];
            tetraConnList[9] = connList[c + 1];
            tetraConnList[10] = connList[c + 3];
            tetraConnList[11] = connList[c + 2];
        }
        break;
    case TYPE_TETRAHEDER: // trivial
        //         cerr << "TETRAHEDER" << endl << endl;
        numTetra = 1;
        tetraConnList[0] = connList[c];
        tetraConnList[1] = connList[c + 1];
        tetraConnList[2] = connList[c + 2];
        tetraConnList[3] = connList[c + 3];
        break;
    default:
        //cerr << "ERROR(USGBlock::decomposeElem)>  unknown type " << typeList[e] << " of element " << e << endl;
        numTetra = 0;
        numTetra = 1;
        tetraConnList[0] = connList[c];
        tetraConnList[1] = connList[c + 1];
        tetraConnList[2] = connList[c + 2];
        tetraConnList[3] = connList[c + 3];
    }
    return;
}

void USGBlock::getCentroid(int e, float &x, float &y, float &z)
{
    int i, n = 1;

    switch (typeList[e])
    {
    case TYPE_HEXAGON:
        n = 8;
        break;
    case TYPE_PYRAMID:
        n = 5;
        break;
    case TYPE_PRISM:
        n = 6;
        break;
    case TYPE_TETRAHEDER:
        n = 4;
        break;
    default:
        cerr << "check USGBlock::getCentroid !!" << endl;
    }

    x = 0.0;
    y = 0.0;
    z = 0.0;
    for (i = 0; i < n; i++)
    {
        x += xCoord[connList[elemList[e] + i]];
        y += yCoord[connList[elemList[e] + i]];
        z += zCoord[connList[elemList[e] + i]];
    }
    x /= ((float)n);
    y /= ((float)n);
    z /= ((float)n);

    return;
}

float USGBlock::tetraVol(float p0[3], float p1[3], float p2[3], float p3[3])
{
    float v;

    v = (((p2[1] - p0[1]) * (p3[2] - p0[2]) - (p3[1] - p0[1]) * (p2[2] - p0[2])) * (p1[0] - p0[0]) + ((p2[2] - p0[2]) * (p3[0] - p0[0]) - (p3[2] - p0[2]) * (p2[0] - p0[0])) * (p1[1] - p0[1]) + ((p2[0] - p0[0]) * (p3[1] - p0[1]) - (p3[0] - p0[0]) * (p2[1] - p0[1])) * (p1[2] - p0[2])) / 6.0f;

    return (v);
}

int USGBlock::isInsideTetra(int tc0, int tc1, int tc2, int tc3, float x, float y, float z, float &u, float &v, float &w)
{
    float px[3], p0[3], p1[3], p2[3], p3[3];
    int r = 1;

    float wg, w0, w1, w2, w3;

    // load coordinates
    px[0] = x;
    px[1] = y;
    px[2] = z;

    p0[0] = xCoord[tc0];
    p0[1] = yCoord[tc0];
    p0[2] = zCoord[tc0];

    p1[0] = xCoord[tc1];
    p1[1] = yCoord[tc1];
    p1[2] = zCoord[tc1];

    p2[0] = xCoord[tc2];
    p2[1] = yCoord[tc2];
    p2[2] = zCoord[tc2];

    p3[0] = xCoord[tc3];
    p3[1] = yCoord[tc3];
    p3[2] = zCoord[tc3];

    wg = fabsf(tetraVol(p0, p1, p2, p3));

    w0 = fabsf(tetraVol(px, p1, p2, p3)) / wg;
    w1 = fabsf(tetraVol(p0, px, p2, p3)) / wg;
    w2 = fabsf(tetraVol(p0, p1, px, p3)) / wg;
    w3 = fabsf(tetraVol(p0, p1, p2, px)) / wg;

    if (w0 + w1 + w2 + w3 <= 1.0001)
    {
        r = 0;
        u = uData[tc0] * w0 + uData[tc1] * w1 + uData[tc2] * w2 + uData[tc3] * w3;
        v = vData[tc0] * w0 + vData[tc1] * w1 + vData[tc2] * w2 + vData[tc3] * w3;
        w = wData[tc0] * w0 + wData[tc1] * w1 + wData[tc2] * w2 + wData[tc3] * w3;
    }
    //   if( tc0==9801 && tc1==8956 && tc2==9800 && tc3==8958 )
    //      cerr << w0+w1+w2+w3 << endl;

    //   else if( tc0>8950 && tc0<9810 && (w0+w1+w2+w3)<2.0 )
    //      cerr << w0+w1+w2+w3 << endl;

    // done
    return (r);
}
