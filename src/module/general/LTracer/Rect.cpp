/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Rect.h"

RectBlock::RectBlock(GridStep *parent, Grid *owner, int b)
    : GridBlock(parent, owner, b)
{
}

RectBlock::~RectBlock()
{
    if (aUData)
        delete[] aUData;
    if (aVData)
        delete[] aVData;
    if (aWData)
        delete[] aWData;
}

int RectBlock::initialize(coDistributedObject *g, coDistributedObject *d, int sS, int bBa, float attA)
{
    coDoRectilinearGrid *rectGrid = dynamic_cast<coDoRectilinearGrid *>(g);
    coDoUniformGrid *uniGrid = dynamic_cast<coDoUniformGrid *>(g);

    if (rectGrid)
    {
        rectGrid->getGridSize(&iDim, &jDim, &kDim);
        rectGrid->getAddresses(&xCoord, &yCoord, &zCoord);
    }
    else if (uniGrid)
    {
        float dx, dy, dz, xmin, xmax, ymin, ymax, zmin, zmax;
        uniGrid->getGridSize(&iDim, &jDim, &kDim);
        uniGrid->getDelta(&dx, &dy, &dz);
        uniGrid->getMinMax(&xmin, &xmax, &ymin, &ymax, &zmin, &zmax);
        xCoord = new float[iDim];
        yCoord = new float[jDim];
        zCoord = new float[kDim];
        for (int index = 0; index < iDim; index++)
            xCoord[index] = xmin + dx * index;
        for (int index = 0; index < jDim; index++)
            yCoord[index] = ymin + dy * index;
        for (int index = 0; index < kDim; index++)
            zCoord[index] = zmin + dz * index;
    }

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
    cerr << "DataType: " << dataType << endl;
    cerr << "Dim: " << iDim << " " << jDim << " " << kDim << endl;

    // speedup
    if (xCoord[0] < xCoord[1])
    {
        i0 = 0;
        iInc = +1;
    }
    else
    {
        i0 = iDim - 1;
        iInc = -1;
    }

    if (yCoord[0] < yCoord[1])
    {
        j0 = 0;
        jInc = +1;
    }
    else
    {
        j0 = jDim - 1;
        jInc = -1;
    }

    if (zCoord[0] < zCoord[1])
    {
        k0 = 0;
        kInc = +1;
    }
    else
    {
        k0 = kDim - 1;
        kInc = -1;
    }

    iStep = iInc * jDim * kDim;
    jStep = jInc * kDim;
    kStep = kInc;

    // "compute" bounding-box
    xMin = xCoord[i0];
    xMax = xCoord[i0 + (iDim - 1) * iInc];
    yMin = yCoord[j0];
    yMax = yCoord[j0 + (jDim - 1) * jInc];
    zMin = zCoord[k0];
    zMax = zCoord[k0 + (kDim - 1) * kInc];

    cerr << "BoundingBox: ( " << xMin << " , " << yMin << " , " << zMin << " )" << endl;
    cerr << "                  -     ( " << xMax << " , " << yMax << " , " << zMax << " )" << endl;

    // done
    if (!uData || !vData || !wData)
        return (0);
    return (1);
}

int RectBlock::initGBI(float x, float y, float z, GridBlockInfo *&gbi)
{
    int i, j, k, iFound, jFound, kFound;

    // check boundingbox
    if (x < xMin || x > xMax || y < yMin || y > yMax || z < zMin || z > zMax)
        return (0);

    // something cached ?
    iFound = -1;
    if (gbiCache)
    {
        i = ((RectBlockInfo *)gbiCache)->i;
        j = ((RectBlockInfo *)gbiCache)->j;
        k = ((RectBlockInfo *)gbiCache)->k;

        if (this->findNextCell(x, y, z, i, j, k) == 1)
        {
            iFound = i;
            jFound = j;
            kFound = k;
        }
    }

    // nothing cached or cache MISS
    if (iFound == -1)
    {
        // TODO:  use binary search here !!!

        // search.....x
        iFound = -1;
        for (i = i0; i >= 0 && i < iDim && iFound == -1; i += iInc)
        {
            if (x >= xCoord[i] && x < xCoord[i + iInc])
                iFound = i;
        }
        if (iFound == -1)
            return (0);

        // search.....y
        jFound = -1;
        for (j = j0; j >= 0 && j < jDim && jFound == -1; j += jInc)
        {
            if (y >= yCoord[j] && y < yCoord[j + jInc])
                jFound = j;
        }
        if (jFound == -1)
            return (0);

        // search.....z
        kFound = -1;
        for (k = k0; k >= 0 && k < kDim && kFound == -1; k += kInc)
        {
            if (z >= zCoord[k] && z < zCoord[k + kInc])
                kFound = k;
        }
        if (kFound == -1)
            return (0);
    }

    // found -> store information
    gbi = new RectBlockInfo();

    gbi->x = x;
    gbi->y = y;
    gbi->z = z;
    gbi->block = this;
    gbi->step = step;
    gbi->stepNo = step->stepNo();
    gbi->blockNo = this->blockNo();
    ((RectBlockInfo *)gbi)->i = iFound;
    ((RectBlockInfo *)gbi)->j = jFound;
    ((RectBlockInfo *)gbi)->k = kFound;
    ((RectBlockInfo *)gbi)->iTmp = iFound;
    ((RectBlockInfo *)gbi)->jTmp = jFound;
    ((RectBlockInfo *)gbi)->kTmp = kFound;

    // store datavalues in gbi (A..H)
    RectBlockInfo *rbi = (RectBlockInfo *)gbi;
    int o, t;
    o = rbi->iTmp * jDim * kDim + rbi->jTmp * kDim + rbi->kTmp;
    t = o + kStep;
    rbi->A0 = uData[t];
    rbi->A1 = vData[t];
    rbi->A2 = wData[t];
    t += jStep;
    rbi->B0 = uData[t];
    rbi->B1 = vData[t];
    rbi->B2 = wData[t];
    t += iStep;
    rbi->C0 = uData[t];
    rbi->C1 = vData[t];
    rbi->C2 = wData[t];
    t -= jStep;
    rbi->D0 = uData[t];
    rbi->D1 = vData[t];
    rbi->D2 = wData[t];
    t = o;
    rbi->E0 = uData[t];
    rbi->E1 = vData[t];
    rbi->E2 = wData[t];
    t += jStep;
    rbi->F0 = uData[t];
    rbi->F1 = vData[t];
    rbi->F2 = wData[t];
    t += iStep;
    rbi->G0 = uData[t];
    rbi->G1 = vData[t];
    rbi->G2 = wData[t];
    t -= jStep;
    rbi->H0 = uData[t];
    rbi->H1 = vData[t];
    rbi->H2 = wData[t];

    // get the velocity
    float dc;
    this->getVelocity(gbi, 0.0, 0.0, 0.0, &gbi->u, &gbi->v, &gbi->w, dc);

    // update cache
    if (!gbiCache)
        gbiCache = gbi->createCopy();
    else
        *gbiCache = *gbi;

    return (1);
}

extern long int d_getVel;

bool RectBlock::isect(float *pos, float *vec, float *normal, float *hitPoint)
{
    float n[3], d;
    d = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    n[0] = normal[0] / d;
    n[1] = normal[1] / d;
    n[2] = normal[2] / d;
    float div = n[0] * vec[0] + n[1] * vec[1] + n[2] * vec[2];
    if ((div < 0.000000001) && (div > -0.000000001))
        return false;
    float t = (-(n[0] * pos[0] + n[1] * pos[1] + n[2] * pos[2] - d)) / div;
    if (t < 0)
        return false;
    hitPoint[0] = vec[0] * t + pos[0];
    hitPoint[1] = vec[1] * t + pos[1];
    hitPoint[2] = vec[2] * t + pos[2];
    return true;
}
int RectBlock::getVelocity(GridBlockInfo *gbi, float dx, float dy, float dz, float *u, float *v, float *w, float &dc)
{
    int r;
    RectBlockInfo *rbi = (RectBlockInfo *)gbi;
    //   int iTmp, jTmp, kTmp;

    d_getVel++;

    dc = 0.0;

    r = findNextCell(rbi->x + dx, rbi->y + dy, rbi->z + dz, rbi->iTmp, rbi->jTmp, rbi->kTmp);
    //   iTmp = rbi->iTmp; jTmp = rbi->jTmp; kTmp = rbi->kTmp;

    /*
   // debugging (just to be sure)
   if( rbi->x+dx > xCoord[rbi->iTmp+iInc] || rbi->x+dx < xCoord[rbi->iTmp] )
      cerr << "*+# not in cell  x #+*" << endl;
   if( rbi->y+dy > yCoord[rbi->jTmp+jInc] || rbi->y+dy < yCoord[rbi->jTmp] )
      cerr << "*+# not in cell  y #+*" << endl;
   if( rbi->z+dz > zCoord[rbi->kTmp+kInc] || rbi->z+dz < zCoord[rbi->kTmp] )
      cerr << "*+# not in cell  z #+*" << endl;
   */

    //cerr << "Block::getVelocity   r = " << r << endl;

    // maybe we have to update the data
    if (aUData)
    {
        r = 1;
        uData = aUData[gbi->stepNo];
        vData = aVData[gbi->stepNo];
        wData = aWData[gbi->stepNo];
    }
    if (r == 1)
    {
        // cell changed -> update
        int o, t;
        o = rbi->iTmp * jDim * kDim + rbi->jTmp * kDim + rbi->kTmp;
        t = o + kStep;
        rbi->A0 = uData[t];
        rbi->A1 = vData[t];
        rbi->A2 = wData[t];
        t += jStep;
        rbi->B0 = uData[t];
        rbi->B1 = vData[t];
        rbi->B2 = wData[t];
        t += iStep;
        rbi->C0 = uData[t];
        rbi->C1 = vData[t];
        rbi->C2 = wData[t];
        t -= jStep;
        rbi->D0 = uData[t];
        rbi->D1 = vData[t];
        rbi->D2 = wData[t];
        t = o;
        rbi->E0 = uData[t];
        rbi->E1 = vData[t];
        rbi->E2 = wData[t];
        t += jStep;
        rbi->F0 = uData[t];
        rbi->F1 = vData[t];
        rbi->F2 = wData[t];
        t += iStep;
        rbi->G0 = uData[t];
        rbi->G1 = vData[t];
        rbi->G2 = wData[t];
        t -= jStep;
        rbi->H0 = uData[t];
        rbi->H1 = vData[t];
        rbi->H2 = wData[t];
    }
    else if (!r)
    {
        // position is outside this block...so calculate the velocity-vector for
        //   the attA-condition
        if (attackAngle == 0.0)
            return (0);

        // compute actual attackAngle

        // attack-vector
        float vec[3], l;
        /*vec[0]=rbi->u+dx;
	  vec[1]=rbi->u+dx;
	  vec[2]=rbi->u+dx;*/
        l = sqrt(rbi->u * rbi->u + rbi->v * rbi->v + rbi->w * rbi->w);
        vec[0] = rbi->u / l;
        vec[1] = rbi->v / l;
        vec[2] = rbi->w / l;

        //calculate intersection point with Xmax
        float pos[3];
        float normal[3];
        float hitPoint[3];
        pos[0] = rbi->x;
        pos[1] = rbi->y;
        pos[2] = rbi->z;
        // try xMax;
        normal[0] = xMax;
        normal[1] = 0;
        normal[2] = 0;
        if ((!isect(pos, vec, normal, hitPoint)) || (hitPoint[1] > yMax || hitPoint[1] < yMin || hitPoint[2] > zMax || hitPoint[2] < zMin))
        {

            // try xMin;
            normal[0] = xMin;
            normal[1] = 0;
            normal[2] = 0;
            if ((!isect(pos, vec, normal, hitPoint)) || (hitPoint[1] > yMax || hitPoint[1] < yMin || hitPoint[2] > zMax || hitPoint[2] < zMin))
            {
                // try yMax;
                normal[0] = 0;
                normal[1] = yMax;
                normal[2] = 0;
                if ((!isect(pos, vec, normal, hitPoint)) || (hitPoint[0] > xMax || hitPoint[0] < xMin || hitPoint[2] > zMax || hitPoint[2] < zMin))
                {

                    // try yMin;
                    normal[0] = 0;
                    normal[1] = yMin;
                    normal[2] = 0;
                    if ((!isect(pos, vec, normal, hitPoint)) || (hitPoint[0] > xMax || hitPoint[0] < xMin || hitPoint[2] > zMax || hitPoint[2] < zMin))
                    {
                        // try zMax;
                        normal[0] = 0;
                        normal[1] = 0;
                        normal[2] = zMax;
                        if ((!isect(pos, vec, normal, hitPoint)) || (hitPoint[1] > yMax || hitPoint[1] < yMin || hitPoint[0] > xMax || hitPoint[0] < xMin))
                        {

                            // try zMin;
                            normal[0] = 0;
                            normal[1] = 0;
                            normal[2] = zMin;
                            if ((!isect(pos, vec, normal, hitPoint)) || (hitPoint[1] > yMax || hitPoint[1] < yMin || hitPoint[0] > xMax || hitPoint[0] < xMin))
                            {
                                cerr << "oops, we did not leave at all, strange" << endl;
                                return 0;
                            }
                        }
                    }
                }
            }
        }

        // and check if condition is met or not

        // it's possible that the vel is allready orthogonal to n
        float a = vec[0] * normal[0] + vec[1] * normal[1] + vec[2] * normal[2];
        //if( a!=0.0 )
        a = fabs((float)acos(a));

        //a *= 180.0/M_PI;

        //cerr << "attA = " << a << "      " << attackAngle << "   " << n[0] << " " << n[1] << " " << n[2] << endl;

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
            float sp = normal[0] * vec[0] + normal[1] * vec[1] + normal[2] * vec[2];

            vec[0] = (vec[0] - normal[0] * sp);
            vec[1] = (vec[1] - normal[1] * sp);
            vec[2] = (vec[2] - normal[2] * sp);
            a = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
            // adapt the length
            *u = vec[0] / a * l;
            *v = vec[1] / a * l;
            *w = vec[2] / a * l;

            //cerr << "   faked vel:  " << *u << "  " << *v << "  " << *w << endl;

            // done
            return (2);
        }

        // done
        return (0);
    }
    //   else
    //      cerr << "."; // cell unchanged...

    // we use standard trilinear interpolation inside a cube
    float x0, y0, z0, xInv, yInv, zInv; // "normalized" coordinates 0<=x0<=1 ,  xInv = 1-x0
    float m0, m1, m2, m3, m4, m5, m6, m7; // for the multiplications (speedup)

    // init vars
    x0 = ((rbi->x + dx) - xCoord[rbi->iTmp]) / (xCoord[rbi->iTmp + iInc] - xCoord[rbi->iTmp]);
    y0 = ((rbi->y + dy) - yCoord[rbi->jTmp]) / (yCoord[rbi->jTmp + jInc] - yCoord[rbi->jTmp]);
    z0 = ((rbi->z + dz) - zCoord[rbi->kTmp]) / (zCoord[rbi->kTmp + kInc] - zCoord[rbi->kTmp]);
    xInv = 1.0f - x0;
    yInv = 1.0f - y0;
    zInv = 1.0f - z0;

    // interpolate
    m0 = xInv * yInv * zInv;
    m1 = x0 * yInv * zInv;
    m2 = xInv * y0 * zInv;
    m3 = xInv * yInv * z0;
    m4 = x0 * yInv * z0;
    m5 = xInv * y0 * z0;
    m6 = x0 * y0 * zInv;
    m7 = x0 * y0 * z0;
    *u = rbi->E0 * m0 + rbi->H0 * m1 + rbi->F0 * m2 + rbi->A0 * m3 + rbi->D0 * m4 + rbi->B0 * m5 + rbi->G0 * m6 + rbi->C0 * m7;
    *v = rbi->E1 * m0 + rbi->H1 * m1 + rbi->F1 * m2 + rbi->A1 * m3 + rbi->D1 * m4 + rbi->B1 * m5 + rbi->G1 * m6 + rbi->C1 * m7;
    *w = rbi->E2 * m0 + rbi->H2 * m1 + rbi->F2 * m2 + rbi->A2 * m3 + rbi->D2 * m4 + rbi->B2 * m5 + rbi->G2 * m6 + rbi->C2 * m7;
    //   *u = rbi->trilinear(0, m0, m1, m2, m3, m4, m5, m6, m7);
    //   *v = rbi->trilinear(1, m0, m1, m2, m3, m4, m5, m6, m7);
    //   *w = rbi->trilinear(2, m0, m1, m2, m3, m4, m5, m6, m7);

    // done
    return (1);
}

extern long int d_fnc;

int RectBlock::findNextCell(float x, float y, float z, int &i, int &j, int &k)
{
    int r = -1, c = 0;

    d_fnc++;

    while (r == -1)
    {
        // check if the point is still inside the cell
        if (x < xCoord[i])
        {
            i -= iInc;
            c = 1;
            if (i >= iDim || i < 0)
            {
                r = 0;
                i += iInc;
            }
        }
        else if (x > xMax)
        {
            r = 0;
        }
        else if (i < iDim - 1 && x > xCoord[i + iInc])
        {
            i += iInc;
            c = 1;
            if (i >= iDim || i < 0)
            {
                i -= iInc;
                r = 0;
            }
        }
        else if (y < yCoord[j])
        {
            j -= jInc;
            c = 1;
            if (j >= jDim || j < 0)
            {
                j += jInc;
                r = 0;
            }
        }
        else if (y > yMax)
        {
            r = 0;
        }
        else if (j < jDim - 1 && y > yCoord[j + jInc])
        {
            j += jInc;
            c = 1;
            if (j >= jDim || j < 0)
            {
                j -= jInc;
                r = 0;
            }
        }
        else if (z < zCoord[k])
        {
            k -= kInc;
            c = 1;
            if (k >= kDim || k < 0)
            {
                k += kInc;
                r = 0;
            }
        }
        else if (z > zMax)
        {
            r = 0;
        }
        else if (k < kDim - 1 && z > zCoord[k + kInc])
        {
            k += kInc;
            c = 1;
            if (k >= kDim || k < 0)
            {
                k -= kInc;
                r = 0;
            }
        }
        else
            r = 1;
    }
    if (r == 0)
    {
        // the particle has left the grid
        //      cerr << "  particle left @  " << x << "  " << y << "  " << z << endl;
    }

    // done
    return ((r == 1 && !c) ? -1 : r);
}

int RectBlock::advanceGBI(GridBlockInfo *gbi, float dx, float dy, float dz)
{
    int r;

    // simply update the position and get the new velocity...this will also advance in the grid
    gbi->x += dx;
    gbi->y += dy;
    gbi->z += dz;
    //r = this->getVelocity(gbi, 0.0, 0.0, 0.0, &gbi->u, &gbi->v, &gbi->w);
    ((RectBlockInfo *)gbi)->i = ((RectBlockInfo *)gbi)->iTmp;
    ((RectBlockInfo *)gbi)->j = ((RectBlockInfo *)gbi)->jTmp;
    ((RectBlockInfo *)gbi)->k = ((RectBlockInfo *)gbi)->kTmp;

    return (0);

    return ((!r) ? 0 : 1);
}
