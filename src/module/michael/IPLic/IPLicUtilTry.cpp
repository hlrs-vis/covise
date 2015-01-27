/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "IPLicUtilTry.h"

////////////////////////////////////////////////////////////////////////

void dimension(int *width, int *height, int nt)
{
    /////////////////////////////////////////////////////////////////////

    int counter = 0;

    int nw = 1;
    int nh = 1;

    while (true)
    {
        if (((nh * nw)) < nt) //changed 2*(nh*nw)
        {
            //if( (counter%2) == 0 )
            //{
            nw *= 2; //nw or nh ?
            //}
            //else
            //{
            nh *= 2; //nh or nw ?
            //}
        }
        else
        {
            break;
        }

        if (counter >= 12) //4^12 => (16*PIXEL_SIZE)MB texture memory
        {
            break;
        }
        else
            ;
        ++counter;
    }

    (*width) = nw;
    (*height) = nh;

    /////////////////////////////////////////////////////////////////////

    //cout << "\ndimension(...): width, height, num_triangles\n" << flush;
    //cout << (*width) << flush << "  " << flush;
    //cout << (*height) << flush << "  " << flush;
    //cout << nt << flush << "\n\n" << flush;
}

////////////////////////////////////////////////////////////////////////

//unnecessary in the current version
//matrix(3,5) - col 4 -> row_scaling, col 5 -> col_pivoting !!
f2ten gauss3D(f2ten matrix)
{
    /////////////////////////////////////////////////////////////////////

    //check matrix

    if (matrix.size() != 3)
    {
        cerr << "\n... problem in transforming matrix\n" << flush;
        exit(99);
    }
    else if ((matrix[0]).size() != 5 || (matrix[1]).size() != 5 || (matrix[2]).size() != 5)
    {
        cerr << "\n... problem in transforming matrix\n" << flush;
        exit(99);
    }
    else
        ;

    /////////////////////////////////////////////////////////////////////

    //scale matrix

    int i;
    int j;

    fvec max = fvec(3, 0.0);

    for (i = 0; i < 3; i++)
    {
        j = maxIndex(matrix[i]);

        max[i] = fabs(matrix[i][j]);
        matrix[i][3] = max[i];
    }

    for (i = 0; i < 3; i++)
    {
        if (max[i] > 1e-10)
        {
            int j;
            for (j = 0; j < 3; j++)
            {
                matrix[i][j] /= max[i];
            }
        }
        else
            ;
    }

    /////////////////////////////////////////////////////////////////////

    //initialize permutation = matrix[*][4]

    for (i = 0; i < 3; i++)
    {
        matrix[i][4] = i;
    }

    /////////////////////////////////////////////////////////////////////

    //do LU decomposition with col_pivoting & row_scaling

    //first step
    {
        fvec tmp = fvec(3, 0.0);
        tmp[0] = matrix[0][0];
        tmp[1] = matrix[1][0];
        tmp[2] = matrix[2][0];
        int kk = maxIndex(tmp);

        cout << "\nfirst: maxIndex = " << flush << kk << flush;

        //pivoting
        swap(matrix[0], matrix[kk]);

        //transformation
        float lambda = 0.0;

        matrix[1][0] /= matrix[0][0];
        lambda = (-1.0) * matrix[1][0];

        matrix[1][1] += lambda * matrix[0][1];
        matrix[1][2] += lambda * matrix[0][2];

        matrix[2][0] /= matrix[0][0];
        lambda = (-1.0) * matrix[2][0];

        matrix[2][1] += lambda * matrix[0][1];
        matrix[2][2] += lambda * matrix[0][2];
    }

    //second step
    {
        fvec tmp = fvec(2, 0.0);
        tmp[0] = matrix[1][1];
        tmp[1] = matrix[2][1];
        int kk = maxIndex(tmp);
        ++kk;

        //pivoting
        swap(matrix[1], matrix[kk]);

        //transformation
        float lambda = 0.0;

        matrix[2][1] /= matrix[1][1];
        lambda = (-1.0) * matrix[2][1];

        matrix[2][2] += lambda * matrix[1][2];
    }

    return matrix;
}

////////////////////////////////////////////////////////////////////////

//computes barycentric coordinates of a triangle in the moment only
//anybody is free to extend it ;-)
fvec bary(const fvec &point, const f2ten &coord)
{
    int nvert = (coord[0]).size();
    int size = point.size();
    fvec lambda = fvec(nvert, 0);

    float delta = 0;
    fvec tmp = fvec();

    if (nvert != 3)
    {
        cout << "\njust triangles implemented," << flush;
        cout << "\nreturning lambda = (0,0,0)\n" << flush;

        return lambda;
    }
    else
    {
    }

    if (size == 3)
    {
        tmp.resize(3, 0);
        lambda.resize(3, 0.0);

        fvec vec1 = fvec(3);
        fvec vec2 = fvec(3);
        fvec vec3 = fvec(3);

        {
            vec1[0] = coord[0][0];
            vec1[1] = coord[1][0];
            vec1[2] = coord[2][0];

            vec2[0] = coord[0][1];
            vec2[1] = coord[1][1];
            vec2[2] = coord[2][1];

            vec3[0] = coord[0][2];
            vec3[1] = coord[1][2];
            vec3[2] = coord[2][2];
        }

        delta = doubleArea(vec1, vec2, vec3);

        lambda[0] = doubleArea(point, vec2, vec3) / delta;
        lambda[1] = doubleArea(vec1, point, vec3) / delta;
        lambda[2] = doubleArea(vec1, vec2, point) / delta;
    }

    else if (size == 2)
    {
        lambda.resize(3, 0.0);

        fvec vec1 = fvec(2);
        fvec vec2 = fvec(2);
        fvec vec3 = fvec(2);

        {
            vec1[0] = coord[0][0];
            vec1[1] = coord[1][0];

            vec2[0] = coord[0][1];
            vec2[1] = coord[1][1];

            vec3[0] = coord[0][2];
            vec3[1] = coord[1][2];
        }

        delta = doubleArea(vec1, vec2, vec3);

        lambda[0] = doubleArea(point, vec2, vec3) / delta;
        lambda[1] = doubleArea(vec1, point, vec3) / delta;
        lambda[2] = doubleArea(vec1, vec2, point) / delta;
    }

    else
    {
    }

    return lambda;
}

fvec doLambda(const fvec &point, const f2ten &coord)
{
    fvec lambda = fvec(3, 0.0);
    fvec A = fvec(2, 0.0);

    float Px = 0.0;
    float Py = 0.0;
    float L = 0.0;

    if ((coord[0]).size() != 3)
    {
        cout << "\njust triangles implemented," << flush;
        cout << "\nreturning lambda = (0,0,0)\n" << flush;

        return lambda;
    }
    else
    {
    }

    if (point.size() == 2)
    {
        A[0] = coord[0][0];
        A[1] = coord[1][0];

        Px = point[0] - A[0];
        Py = point[1] - A[1];
        L = coord[0][1] - coord[0][0];

        lambda[2] = Py / L;
        lambda[1] = Px / L;
        lambda[0] = 1.0 - ((Px + Py) / L);
    }
    else
    {
        cout << "\ndesigned for texture coordinates," << flush;
        cout << "\nreturning lambda = (0,0,0)\n" << flush;

        return lambda;
    }

    return lambda;
}

//returns vector of size 2
fvec vec(const fvec &lambda, const f2ten &v)
{
    int l_size = lambda.size();
    int v_size = v.size();
    int vx_size = 0;
    int vy_size = 0;

    fvec tmp = fvec(2, 0.0);

    if ((l_size == 3) && (v_size == 2))
    {
        vx_size = (v[0]).size();
        vy_size = (v[1]).size();

        if ((vx_size == 3) && (vy_size == 3))
        {
            tmp[0] = (lambda[0] * v[0][0]) + (lambda[1] * v[0][1]) + (lambda[2] * v[0][2]);
            tmp[1] = (lambda[0] * v[1][0]) + (lambda[1] * v[1][1]) + (lambda[2] * v[1][2]);
            return tmp;
        }
        else
        {
            cerr << "\n\nsize problem as interpolating vector\n" << flush;
            return tmp;
        }
    }
    else
    {
        cerr << "\n\nsize problem as interpolating vector\n" << flush;
        return tmp;
    }
}

////////////////////////////////////////////////////////////////////////

//returns vector of size 2
//using midpoint rule in the moment
//returns 0-vector if v-vector too small
fvec pstep(Triangles *triangle, fvec pos, float h)
{
    const float VMIN = 1e-06;

    /////////////////////////////////////////////////////////////////////

    fvec next = fvec(2, 0.0);
    fvec temp = fvec(2, 0.0);

    f2ten points = f2ten(2);
    (points[0]).resize(3, 0.0);
    (points[1]).resize(3, 0.0);

    fvec v1 = fvec(2, 0.0);
    fvec v2 = fvec(2, 0.0);
    fvec v3 = fvec(2, 0.0);
    fvec v = fvec(2, 0.0);

    fvec k1 = fvec(2, 0.0);
    fvec k2 = fvec(2, 0.0);

    //float delta = 0.0;
    float vabs = 0.0;
    fvec lambda = fvec(3, 0.0);

    {
        points = triangle->getC2dTex();

        v1 = triangle->getVTex(0);
        v2 = triangle->getVTex(1);
        v3 = triangle->getVTex(2);
    }

    //delta = doubleArea(p1, p2, p3);
    {

        lambda = doLambda(pos, points);
        //lambda[0] = 1.0;
        v[0] = lambda[0] * v1[0] + lambda[1] * v2[0] + lambda[2] * v3[0];
        v[1] = lambda[0] * v1[1] + lambda[1] * v2[1] + lambda[2] * v3[1];
        vabs = abs(v);

        if (vabs > VMIN)
        {
            v[0] = v[0] / vabs;
            v[1] = v[1] / vabs;
            k1[0] = h * v[0];
            k1[1] = h * v[1];
        }
        else
        {
            return next;
        }

        //temp[0] = 0.5*k1[0];
        //temp[1] = 0.5*k1[1];

        temp[0] += pos[0];
        temp[1] += pos[1];

        //next = pos + k1;

        lambda = doLambda(temp, points);

        v[0] = lambda[0] * v1[0] + lambda[1] * v2[0] + lambda[2] * v3[0];
        v[1] = lambda[0] * v1[1] + lambda[1] * v2[1] + lambda[2] * v3[1];
        vabs = abs(v);

        if (vabs > VMIN)
        {
            v[0] = v[0] / vabs;
            v[1] = v[1] / vabs;
            k2[0] = h * v[0];
            k2[1] = h * v[1];
        }
        else
        {
            return next;
        }

        next[0] = pos[0] + k2[0]; //?? !!
        next[1] = pos[1] + k2[1]; //?? !!
    }

    return next;
}

////////////////////////////////////////////////////////////////////////

void doFieldLine(trivec &triangles, const i2ten &neighbours,
                 f3ten &accum, i2ten &hitcount, int res, float delta,
                 int ii, int jj, int hh, int ww, long *seed)
{
    int triLimit = triangles.size() - 1;

    int stri = tInd(ii, jj, hh, ww, res); //start triangle index
    int ftri = stri; //forward triangle index
    int btri = stri; //backward triangle index
    //long seed = SEED;  //defined in header

    /////////////////////////////////////////////////////////////////////

    if ((stri < 0) || (stri > triLimit))
    {
        return;
    }
    else
        ;

    ///////////////////////////////////////////////////////////////////////int first  = (triangles[ftri]).getCoordIndex(1);

    const int first = 0;
    const int second = 1;
    const int third = 2;

    const float hf = delta / 2.0; //forward direction
    const float hb = (-1.0) * hf; //backward direction

    fvec forward = fvec(2, 0.0); //position
    ivec fpix = ivec(2, 0); //pixel

    fvec backward = fvec(2, 0.0); //position
    ivec bpix = ivec(2, 0); //pixel

    fvec tmp_forward = fvec(2, 0.0);
    ivec tmp_fpix = ivec(2, 0);

    fvec tmp_backward = fvec(2, 0.0);
    ivec tmp_bpix = ivec(2, 0);

    /////////////////////////////////////////////////////////////////////

    //init for forward stepping (fLen+1 -> 2*fLen)
    //position -> middle of pixel
    forward[0] = jj * delta;
    forward[0] += hf;
    forward[1] = ii * delta;
    forward[1] += hf;

    //init for backward stepping (fLen-1 -> 0)
    backward[0] = jj * delta;
    backward[0] += hb;
    backward[1] = ii * delta;
    backward[1] += hb;

    /////////////////////////////////////////////////////////////////////

    //const float threshold = 1e-03;

    //break, if abs(vector) <= threshold
    //step "forward" & "backward" simultaneously
    {
        //////////////////////////////////////////////////////////////////

        int bcol;
        int brow;
        int fcol;
        int frow;
        bpix[0] = ii;
        bpix[1] = jj;
        fpix[0] = ii;
        fpix[1] = jj;

        //i2ten nhits(hh*res);

        //for(fcol = 0; fcol < (hh*res ); fcol++)
        //{
        //(nhits[fcol]).resize(ww*res, 0);
        //}

        //////////////////////////////////////////////////////////////////

        fcol = ii;
        frow = jj;
        bcol = ii;
        brow = jj;
        //++(nhits[fcol][frow]); //haeh ?  aah !

        //////////////////////////////////////////////////////////////////

        int loop;
        int accumSize = accum.size();
        fvec ranVal = fvec(accumSize, 1.0);

        {
            if (accumSize == 1)
            {
                ranVal[0] = random2(&(*seed));
            }

            else if (accumSize == 4)
            {
                ranVal[0] = random2(&(*seed));
                ranVal[1] = random2(&(*seed));
                ranVal[2] = random2(&(*seed));
            }

            else
                ;
        }

        //////////////////////////////////////////////////////////////////

        //cout << "\n\nftri = " << flush << ftri << flush;

        //preparation => P R O B L E M ! ! !
        //int first  = (triangles[ftri]).getCoordIndex(1);
        //int second = (triangles[ftri]).getCoordIndex(2);
        //int third  = (triangles[ftri]).getCoordIndex(3);

        //cout << "\n\nfirst = " << flush << first << flush;
        //cout << "  second = " << flush << second << flush;
        //cout << "  third = " << flush << third << flush;
        //cout << "\nftri = " << flush << ftri << flush;

        fvec posIn = fvec(2, 0.0);
        fvec posOut = fvec(2, 0.0);

        fvec tFirst = fvec(2, 0.0);
        tFirst = (triangles[ftri]).getC2dTex(first);
        fvec tSecond = fvec(2, 0.0);
        tSecond = (triangles[ftri]).getC2dTex(second);

        const float tLength = tSecond[0] - tFirst[0];

        //////////////////////////////////////////////////////////////////

        /***************************************\ 
       *                                     *
       *  F O R W A R D   D I R E C T I O N  *
       *                                     *
      \***************************************/

        //break if the streamline passes the boundary
        int counter = 0;
        while (true)
        {
            //next step in forward direction
            tmp_forward = pstep(&triangles[ftri], forward, hf);

            ///////////////////////////////////////////////////////////////

            //tmp_forward still in triangle ?
            posOut = tmp_forward - tFirst;
            posIn = forward - tFirst;

            //if( (ftri%2) == 0 )
            //{
            if ((posOut[0] < 0) || (posOut[1] < 0) || ((posOut[0] + posOut[1]) > tLength))
            {
                //get new triangle & new pos
                ftri = getTri(triangles, neighbours, ftri, first,
                              second, third, posIn, posOut, tLength,
                              delta);

                if ((ftri < 0) || (ftri > triLimit))
                {
                    return;
                }
                else
                    ;

                tFirst = (triangles[ftri]).getC2dTex(first);
                tSecond = (triangles[ftri]).getC2dTex(second);
            }
            else
                ;
            //}

            /*
                  else
                  {
                     if( ( posOut[0] > 0 ) || ( posOut[1] > 0 ) ||\ 
                         ( (posOut[0]+posOut[1]) < tLength ) )
                     {
                        //get new triangle & new pos
                        ftri = getTri(triangles, neighbours, ftri, &first,\ 
                                      &second, &third, posIn, posOut, tLength,\ 
                                      delta);

         if( (ftri < 0) || (ftri > triLimit) )
         {
         return;
         }
         else;

         tFirst = (triangles[ftri]).getC2dTex(first);
         tThird = (triangles[ftri]).getC2dTex(third);
         }
         else;
         }
         */
            ///////////////////////////////////////////////////////////////

            fpix = pos2pix(tmp_forward, delta);
            fcol = fpix[0];
            frow = fpix[1];

            if ((fcol < 0) || (frow < 0))
            {
                break; //out of area
            }
            else if ((fcol >= ww) || (frow >= hh))
            {
                break; //out of area
            }
            else
                ;

            ///////////////////////////////////////////////////////////////

            //cout << "\n\nfcol = fpix[0] = " << flush << fcol << flush;
            //cout << "  frow = fpix[1] = " << flush << frow << flush;

            for (loop = 0; loop < accumSize - 1; loop++)
            {
                accum[loop][frow][fcol] += ranVal[loop];
            }
            ++hitcount[frow][fcol];

            ///////////////////////////////////////////////////////////////

            if ((hitcount[frow][fcol] > NHITS))
            {
                break;
            }
            else if ((ftri < 0) || (ftri > triLimit))
            {
                break; //schon oben erledigt !
            }
            else
                ;

            forward = tmp_forward;

            //++counter;
            //if(counter > (ww*hh))
            //{
            //break;
            //}

            //printing steps in forward direction
            {
                int fc = 0;

                //cout << "\nforward step : " << flush << fc << flush;
                //cout << "\nforward point: " << flush << forward[0] << flush;
                //cout << "  " << flush << forward[1] << flush;

                //cout << "\n\nfrow fcol : " << flush << frow << flush;
                //cout << "  " << flush << fcol << flush;

                ++fc;
            }
        }

        //////////////////////////////////////////////////////////////////

        //preparation => P R O B L E M ! ! ! ? ? ?
        //first  = (triangles[btri]).getCoordIndex(1);
        //second = (triangles[btri]).getCoordIndex(2);
        //third  = (triangles[btri]).getCoordIndex(3);

        tFirst = (triangles[btri]).getC2dTex(first);
        tSecond = (triangles[btri]).getC2dTex(second);

        //////////////////////////////////////////////////////////////////

        /*****************************************\ 
       *                                       *
       *  B A C K W A R D   D I R E C T I O N  *
       *                                       *
      \*****************************************/

        //break if the streamline passes the boundary
        while (true)
        {
            //next step in backward direction
            tmp_backward = pstep(&triangles[btri], backward, hb);

            ///////////////////////////////////////////////////////////////

            //tmp_forward still in triangle ?
            posOut = tmp_backward - tFirst;
            posIn = backward - tFirst;

            //if( (btri%2) == 0 )
            //{
            if ((posOut[0] < 0) || (posOut[1] < 0) || ((posOut[0] + posOut[1]) > tLength))
            {
                //get new triangle & new pos
                btri = getTri(triangles, neighbours, btri, first,
                              second, third, posIn, posOut, tLength,
                              delta);

                if ((btri < 0) || (btri > triLimit))
                {
                    return;
                }
                else
                    ;

                tFirst = (triangles[btri]).getC2dTex(first);
                tSecond = (triangles[btri]).getC2dTex(second);
            }
            else
                ;
            //}

            /* 
                  else
                  {
                     if( ( posOut[0] > 0 ) || ( posOut[1] > 0 ) ||\ 
                         ( (posOut[0]+posOut[1]) < tLength ) )
                     {
                        //get new triangle & new pos
                        btri = getTri(triangles, neighbours, btri, &first,\ 
                                      &second, &third, posIn, posOut, tLength,\ 
                                      delta);

         if( (btri < 0) || (btri > triLimit) )
         {
         return;
         }
         else;

         tFirst = (triangles[btri]).getC2dTex(first);
         tThird = (triangles[btri]).getC2dTex(third);
         }
         else;
         }
         */
            ///////////////////////////////////////////////////////////////

            bpix = pos2pix(tmp_backward, delta);
            bcol = bpix[0];
            brow = bpix[1];

            if ((bcol < 0) || (brow < 0))
            {
                break; //out of area
            }
            else if ((bcol >= ww) || (brow >= hh))
            {
                break; //out of area
            }
            else
                ;

            ///////////////////////////////////////////////////////////////

            for (loop = 0; loop < accumSize - 1; loop++)
            {
                accum[loop][brow][bcol] += ranVal[loop];
            }
            ++hitcount[brow][bcol];

            ///////////////////////////////////////////////////////////////

            if ((hitcount[brow][bcol] > NHITS))
            {
                break;
            }
            else if ((btri < 0) || (btri > triLimit))
            {
                break; //schon oben erledigt !
            }
            else
                ;

            backward = tmp_backward;

            ++counter;
            if (counter > (ww * hh))
            {
                break;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////

fvec pix2pos(int ii, int jj, float delta)
{
    fvec pos = fvec(2, 0.0);

    pos[0] = jj * delta;
    pos[0] += (delta / 2.0);

    pos[1] = ii * delta;
    pos[1] += (delta / 2.0);

    return pos;
}

ivec pos2pix(fvec pos, float delta)
{
    ivec pix = ivec(2, 0);
    //float tmp = 0.0;

    if (pos.size() != 2)
    {
        return pix;
    }
    else
        ;

    pos[0] = floor(pos[0] / delta);
    pos[1] = floor(pos[1] / delta);

    pix[0] = f2int(pos[0]);
    pix[1] = f2int(pos[1]);

    return pix;
}

void accum2pixels(const f3ten &accum, const i2ten &hitcount,
                  char **image, int img_size)
{
    int i;
    int j;
    int k;
    int p; //pixel index
    float tmp = 0.0;

    int asize = accum.size();
    int hsize = (accum[0]).size();
    int wsize = (accum[0][0]).size();

    for (i = 0; i < asize - 1; i++)
    {
        p = asize * i;
        for (j = 0; j < hsize; j++)
        {
            p += (hsize * j);
            for (k = 0; k < wsize; k++)
            {
                p += k;
                if ((hitcount[j][k] > 0) && (p < img_size))
                {
                    tmp = accum[i][j][k] / hitcount[j][k];
                    (*image)[p] = static_cast<unsigned char>((255.0 * tmp) + 0.5);
                }
                else
                    ;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////

//index <= the triangle that contains the pixel (ii,jj)
//ww: width(in pixels) = width(in quads)*res(olution)
//hh: the same with height
int tInd(int ii, int jj, int hh, int ww, int res)
{
    int index = 0;

    int iq = ii / res;
    int jq = jj / res;
    /* int hq = hh/res; */ int wq = ww / res;

    int ir = ii % res;
    int jr = jj % res;

    index = (wq * iq) + jq;
    index *= 2;

    if ((ir + jr) == (res - 1)) //(0,0)...(0,res-1),(1,0),...,(res-1,res-1)
    {
        return -1;
    }
    else if ((ir + jr) > (res - 1))
    {
        index += 1;
    }
    else
        ;

    return index;
}

//more than 1 neighbour possible !! we take the first we find ...
int getTri(trivec &triangles, const i2ten &neighbours,
           int tri, int first, int second, int third,
           fvec &posIn, fvec posOut, float tLength, float delta)
{
    //cout << "\n\nleaving triangle " << flush << tri << flush;
    //cout << "\nposIn  = " << flush << posIn[0];
    //cout << "  " << flush << posIn[1] << '\n' << flush;
    //cout << "\nposOut = " << flush << posOut[0];
    //cout << "  " << flush << posOut[1] << '\n' << flush;

    int index = -1;
    int no1 = 0;
    int no2 = 0;

    ivec tNodes = ivec(3);
    tNodes = (triangles[tri]).getNodes();

    fvec direction = fvec(2, 0.0);
    direction = posOut - posIn;
    float lambda = 0.0;
    {
        if (abs(direction) > (0.1 * delta))
        {
            lambda = 1.0 / abs(direction);
        }
        else
        {
            return -1;
        }
    }

    //intersection of the line posIn -> posOut
    //with the triangle boundary
    fvec intersection = fvec(2, 0.0);
    ivec iNodes = ivec(2, -1);

    //if( (tri%2) == 0 )
    //{
    if ((posOut[0] + posOut[1]) > tLength)
    {
        iNodes[0] = tNodes[second];
        no1 = second;
        iNodes[1] = tNodes[third];
        no2 = third;

        intersection[0] = (tLength * direction[0]);
        intersection[0] += (posIn[0] * direction[1]);
        intersection[0] -= (posIn[1] * direction[0]);
        intersection[0] /= (direction[0] + direction[1]);

        intersection[1] = tLength - intersection[0];
    }
    else if ((posOut[1] < 0) && (posOut[1] <= posOut[0]))
    {
        lambda *= posIn[1];

        iNodes[0] = tNodes[first];
        no1 = first;
        iNodes[1] = tNodes[second];
        no2 = second;

        intersection = posIn + (lambda * direction);
    }
    else if ((posOut[0] < 0) && (posOut[0] < posOut[1]))
    {
        lambda *= posIn[0];

        iNodes[0] = tNodes[first];
        no1 = first;
        iNodes[1] = tNodes[third];
        no2 = third;

        intersection = posIn + (lambda * direction);
    }
    else
        ;
    //}
    /* 
      else
      {
         if( (posOut[0] + posOut[1]) <= tLength )
         {
            iNodes[0] = tNodes[(*first)];
            no1 = (*first);
            iNodes[1] = tNodes[(*second)];
            no2 = (*second);

            intersection[0] = (tLength*direction[0]);
   intersection[0] += (posIn[0]*direction[1]);
   intersection[0] -= (posIn[1]*direction[0]);
   intersection[0] /= (direction[0]+direction[1]);

   intersection[1] = tLength - intersection[0];
   }
   else if( (posOut[1] > 0)  && (posOut[1] >= posOut[0]) )
   {
   lambda *= posIn[1];

   iNodes[0] = tNodes[(*third)];
   no1 = (*third);
   iNodes[1] = tNodes[(*first)];
   no2 = (*first);

   intersection = posIn + (lambda*direction);
   }
   else if( (posOut[0] > 0)  && (posOut[0] >= posOut[1]) )
   {
   lambda *= posIn[0];

   iNodes[0] = tNodes[(*third)];
   no1 = (*third);
   iNodes[1] = tNodes[(*second)];
   no2 = (*second);

   intersection = posIn + (lambda*direction);
   }
   else;
   }
   */
    //find a neighbour triangle
    {
        int i, j;

        int t1 = iNodes[0];
        int t2 = iNodes[1];

        int size1 = (neighbours[t1]).size();
        int size2 = (neighbours[t2]).size();

        for (i = 0; i < size1; i++)
        {
            for (j = 0; j < size2; j++)
            {
                if (neighbours[t1][i] == neighbours[t2][j])
                {
                    index = neighbours[t1][i];

                    if (index != tri)
                    {
                        break;
                    }
                }
                else
                    ;
            }
            if ((index >= 0) && (index != tri))
            {
                break;
            }
            else
                ;
        }
    }

    //we need now  : entry point of the field line -> store in posIn
    {
        float divisor = 0.0;

        fvec p1 = fvec(2, 0.0);
        fvec p2 = fvec(2, 0.0);

        //vector node1 -> node2 of the common edge
        p1 = (triangles[tri]).getC2dTex(no1);
        p2 = (triangles[tri]).getC2dTex(no2);

        if (abs(p2 - p1) > (delta * 0.5))
        {
            divisor = abs(intersection - p1) / abs(p2 - p1);
        }
        else
            ;

        if (index >= 0)
        {
            tNodes = (triangles[index]).getNodes();

            int loop;
            for (loop = 0; loop < 3; loop++)
            {
                if (tNodes[loop] == iNodes[0])
                {
                    no1 = loop;
                }
                else if (tNodes[loop] == iNodes[1])
                {
                    no2 = loop;
                }
                else
                    ;
            }

            //vector node1 -> node2 of the common edge
            p1 = (triangles[index]).getC2dTex(no1);
            p2 = (triangles[index]).getC2dTex(no2);

            posIn = p1 + divisor * (p2 - p1);
        }
        else
            ;
    }

    //(*first) = (triangles[index]).getCoordIndex(1);
    //(*second) = (triangles[index]).getCoordIndex(2);
    //(*third) = (triangles[index]).getCoordIndex(3);

    //cout << "\n\nentering triangle " << flush << index << flush;
    //cout << "\nIntersection  = " << flush << intersection[0];
    //cout << "  " << flush << intersection[1] << '\n' << flush;
    //cout << "\nposIn = " << flush << posIn[0];
    //cout << "  " << flush << posIn[1] << '\n' << flush;

    return index;
}
