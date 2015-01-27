/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "ReadASCIIDyna.h"
#include <math.h>

float
Laenge(float x, float y)
{
    return (sqrt(x * x + y * y));
}

float
LaengeHoch23D(float x0, float y0, float z0, float x1, float y1, float z1)
{
    return ((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) + (z0 - z1) * (z0 - z1));
}

void
ReadASCIIDyna::MergeNodes(const vector<float> &exc,
                          const vector<float> &eyc,
                          const vector<float> &ezc,
                          vector<int> &ecl,
                          const vector<int> &epl,
                          float tolerance, float cutX, float cutY)
{
    (void)epl;
    vector<int> coincidence;
    // initialise to -1;
    int node;
    for (node = 0; node < exc.size(); ++node)
    {
        coincidence.push_back(-1);
    }
    float tolerance2 = tolerance * tolerance;
    // the following two nested loops ought to be simplified
    vector<int> candidateNode;
    for (node = 0; node < exc.size(); ++node)
    {
        float resX = 0.0, resY = 0.0;
        if (cutX != 0.0)
        {
#if defined(__APPLE__) || defined(__MINGW32__)
            resX = remainder(exc[node], cutX);
#else
            // drem() is legacy API !!  remainder() is C99 and should be prefered!
            resX = drem(exc[node], cutX);
#endif
        }
        if (cutY != 0.0)
        {
#if defined(__APPLE__) || defined(__MINGW32__)
            resY = remainder(eyc[node], cutY);
#else
            resY = drem(eyc[node], cutY);
#endif
        }
        if (fabs(resX) > tolerance && fabs(resY) > tolerance)
        {
            continue;
        }
        candidateNode.push_back(node);
    }
    int c_node;
    for (c_node = 1; c_node < candidateNode.size(); ++c_node)
    {
        int node = candidateNode[c_node];
        int c_node_prev;
        /*
            float resX=0.0,resY=0.0;
            if(cutX!=0.0){
               resX = drem(exc[node],cutX);
            }
            if(cutY!=0.0){
               resY = drem(eyc[node],cutY);
            }
            if(fabs(resX) > tolerance && fabs(resY) > tolerance){
               continue;
            }
      */
        for (c_node_prev = 0; c_node_prev < c_node; ++c_node_prev)
        {
            int node_prev = candidateNode[c_node_prev];
            float len = LaengeHoch23D(exc[node_prev], eyc[node_prev], ezc[node_prev],
                                      exc[node], eyc[node], ezc[node]);
            if (len < tolerance2)
            {
                if (coincidence[node_prev] < 0)
                {
                    coincidence[node] = node_prev;
                }
                else
                {
                    coincidence[node] = coincidence[node_prev];
                }
            }
        }
    }
    int vert;
    for (vert = 0; vert < ecl.size(); ++vert)
    {
        int node = ecl[vert];
        if (coincidence[node] >= 0)
        {
            ecl[vert] = coincidence[node];
        }
    }
}

// do nothing with the normals
int
ReadASCIIDyna::rectangularKnob(float height, float groundRad, float abrasRad,
                               float winkel, float laeng1, float laeng2,
                               int minDiv, int upDiv, int downDiv,
                               vector<int> &el, vector<int> &vl,
                               vector<float> &xl, vector<float> &yl, vector<float> &zl,
                               vector<float> &nx, vector<float> &ny, vector<float> &nz)
{
    winkel *= M_PI / 180.0;
    el.clear();
    vl.clear();
    xl.clear();
    yl.clear();
    zl.clear();
    nx.clear();
    ny.clear();
    nz.clear();
    int xDiv, yDiv;
    if (laeng1 > laeng2)
    {
        yDiv = minDiv;
        xDiv = (int)((float)yDiv * laeng1 / laeng2);
    }
    else
    {
        xDiv = minDiv;
        yDiv = (int)((float)xDiv * laeng2 / laeng1);
    }
    // zuerst top nodes
    int i, j;
    float beta = M_PI * 0.25 - 0.5 * winkel;
    float Rx = laeng1 * 0.5;
    float Ry = laeng2 * 0.5;
    float Rpx = Rx - abrasRad * tan(beta);
    float Rpy = Ry - abrasRad * tan(beta);
    float dx = Rpx / xDiv;
    float dy = Rpy / yDiv;
    for (j = 0; j <= yDiv; ++j)
    {
        for (i = 0; i <= xDiv; ++i)
        {
            xl.push_back(i * dx);
            yl.push_back(j * dy);
            zl.push_back(height);
        }
    }
    // top elements
    for (j = 0; j < yDiv; ++j)
    {
        for (i = 0; i < xDiv; ++i)
        {
            el.push_back(vl.size());
            vl.push_back(j * (xDiv + 1) + i);
            vl.push_back(j * (xDiv + 1) + i + 1);
            vl.push_back((j + 1) * (xDiv + 1) + i + 1);
            vl.push_back((j + 1) * (xDiv + 1) + i);
        }
    }
    // X side
    vector<float> xn;
    vector<float> yn;
    xn.push_back(0.0);
    yn.push_back(0.0);
    xn.push_back(Rx - abrasRad * tan(beta));
    yn.push_back(height);
    int div;
    for (div = 1; div <= upDiv; ++div)
    {
        xn.push_back(xn[1] + abrasRad * sin((2 * beta * div) / upDiv));
        yn.push_back(yn[1] - (1.0 - cos((2 * beta * div) / upDiv)) * abrasRad);
    }
    /*
      x2.push_back(x1 + abrasRad * sin(beta));
      float y2 = y1 - (1.0 - cos(beta))*abrasRad;
      float x3 = x1 + abrasRad * sin(2*beta);
      float y3 = y1 - (1.0 - cos(2*beta))*abrasRad;
   */
    // now from the end...
    int start = xn.size();
    for (div = 0; div <= downDiv; ++div)
    {
        xn.push_back(Rx + height * tan(winkel) + groundRad * tan(beta));
        yn.push_back(0.0);
    }
    for (div = downDiv; div >= 1; --div)
    {
        xn[start + div - 1] = xn[xn.size() - 1] - groundRad * sin((2 * beta * (downDiv - div + 1)) / downDiv);
        yn[start + div - 1] = groundRad * (1 - cos((2 * beta * (downDiv - div + 1)) / downDiv));
    }
    /*
      float x6 = Rpx + height*tan(winkel) + groundRad*tan(beta);
      float y6 = 0.0;
      float x5 = x6 - groundRad*sin(beta);
      float y5 = groundRad*(1-cos(beta));
      float x4 = x6 - groundRad*sin(2*beta);
      float y4 = groundRad*(1-cos(2*beta));
   */
    // what happens if y4 > y3?
    if (yn[start] > yn[start - 1])
    {
        xn[start - 1] = 0.5 * (xn[start - 1] + xn[start]);
        yn[start - 1] = 0.5 * (yn[start - 1] + yn[start]);
        xn[start] = xn[start - 1];
        yn[start] = yn[start - 1];
    }
    float scale = 1.0 / (yDiv * dy);
    for (j = 0; j <= yDiv; ++j)
    {
        int point;
        for (point = 1; point < xn.size(); ++point)
        {
            xl.push_back(xn[point]);
            yl.push_back(j * dy + (xn[point] - Rpx) * j * dy * scale);
            zl.push_back(yn[point]);
        }
        /*
            xl.push_back(x1);
            yl.push_back(j*dy+(x1-Rpx)*j*dy*scale);
            zl.push_back(height);
            xl.push_back(x2);
            yl.push_back(j*dy+(x2-Rpx)*j*dy*scale);
            zl.push_back(y2);
            xl.push_back(x3);
            yl.push_back(j*dy+(x3-Rpx)*j*dy*scale);
            zl.push_back(y3);
            xl.push_back(x4);
      yl.push_back(j*dy+(x4-Rpx)*j*dy*scale);
      zl.push_back(y4);
      xl.push_back(x5);
      yl.push_back(j*dy+(x5-Rpx)*j*dy*scale);
      zl.push_back(y5);
      xl.push_back(x6);
      yl.push_back(j*dy+(x6-Rpx)*j*dy*scale);
      zl.push_back(y6);
      */
    }
    // now the elements
    int base = (xDiv + 1) * (yDiv + 1);
    int band = xn.size() - 1; // 6 when taking 2 + 2 divisions
    for (j = 0; j < yDiv; ++j)
    {
        el.push_back(vl.size());

        // vl.push_back(j*6+base);
        vl.push_back(j * (xDiv + 1) + xDiv);
        vl.push_back(j * band + 1 + base);
        vl.push_back((j + 1) * band + 1 + base);
        // vl.push_back((j+1)*6+base);
        vl.push_back((j + 1) * (xDiv + 1) + xDiv);

        for (i = 1; i < band - 1; ++i)
        {
            el.push_back(vl.size());
            // inverted orientation FIXME
            vl.push_back(j * band + i + base);
            vl.push_back(j * band + i + 1 + base);
            vl.push_back((j + 1) * band + i + 1 + base);
            vl.push_back((j + 1) * band + i + base);
        }
    }
    xn.clear();
    yn.clear();
    xn.push_back(0.0);
    yn.push_back(0.0);
    xn.push_back(Ry - abrasRad * tan(beta));
    yn.push_back(height);
    for (div = 1; div <= upDiv; ++div)
    {
        xn.push_back(xn[1] + abrasRad * sin((2 * beta * div) / upDiv));
        yn.push_back(yn[1] - (1.0 - cos((2 * beta * div) / upDiv)) * abrasRad);
    }
    // Y side
    /*
      x1 = Ry - abrasRad * tan(beta);
      y1 = height;
      x2 = x1 + abrasRad * sin(beta);
      y2 = height - (1.0 - cos(beta))*abrasRad;
      x3 = x1 + abrasRad * sin(2*beta);
      y3 = height - (1.0 - cos(2*beta))*abrasRad;
   */
    // now from the end...
    for (div = 0; div <= downDiv; ++div)
    {
        xn.push_back(Ry + height * tan(winkel) + groundRad * tan(beta));
        yn.push_back(0.0);
    }
    for (div = downDiv; div >= 1; --div)
    {
        xn[start + div - 1] = xn[xn.size() - 1] - groundRad * sin((2 * beta * (downDiv - div + 1)) / downDiv);
        yn[start + div - 1] = groundRad * (1 - cos((2 * beta * (downDiv - div + 1)) / downDiv));
    }
    /*
      x6 = Rpy + height*tan(winkel) + groundRad*tan(beta);
      y6 = 0.0;
      x5 = x6 - groundRad*sin(beta);
      y5 = groundRad*(1-cos(beta));
      x4 = x6 - groundRad*sin(2*beta);
      y4 = groundRad*(1-cos(2*beta));
   */
    // what happens if y4 > y3?
    if (yn[start] > yn[start - 1])
    {
        xn[start - 1] = 0.5 * (xn[start - 1] + xn[start]);
        yn[start - 1] = 0.5 * (yn[start - 1] + yn[start]);
        xn[start] = xn[start - 1];
        yn[start] = yn[start - 1];
    }
    /*
      if(y4 > y3){
         x3 = 0.5*(x3+x4);
         y3 = 0.5*(y3+y4);
         x4 = x3;
         y4 = y3;
      }
   */
    scale = 1.0 / (xDiv * dx);
    for (i = 0; i <= xDiv; ++i)
    {
        int point;
        for (point = 1; point < xn.size(); ++point)
        {
            xl.push_back(i * dx + (xn[point] - Rpy) * i * dx * scale);
            yl.push_back(xn[point]);
            zl.push_back(yn[point]);
        }
        /*
         for(i=0;i<=xDiv;++i){
            xl.push_back(i*dx+(x1-Rpy)*i*dx*scale);
            yl.push_back(x1);
            zl.push_back(height);
            xl.push_back(i*dx+(x2-Rpy)*i*dx*scale);
            yl.push_back(x2);
            zl.push_back(y2);
            xl.push_back(i*dx+(x3-Rpy)*i*dx*scale);
            yl.push_back(x3);
            zl.push_back(y3);
      xl.push_back(i*dx+(x4-Rpy)*i*dx*scale);
      yl.push_back(x4);
      zl.push_back(y4);
      xl.push_back(i*dx+(x5-Rpy)*i*dx*scale);
      yl.push_back(x5);
      zl.push_back(y5);
      xl.push_back(i*dx+(x6-Rpy)*i*dx*scale);
      yl.push_back(x6);
      zl.push_back(y6);
      */
    }
    // now the elements
    base = (xDiv + 1) * (yDiv + 1) + band * (yDiv + 1);
    for (i = 0; i < xDiv; ++i)
    {
        int band = xn.size() - 1; // 6 when taking 2 + 2 divisions
        el.push_back(vl.size());
        vl.push_back(i * band + 1 + base);
        // vl.push_back(i*6+base);
        vl.push_back((xDiv + 1) * yDiv + i);
        // vl.push_back((i+1)*6+base);
        vl.push_back((xDiv + 1) * yDiv + i + 1);
        vl.push_back((i + 1) * band + 1 + base);
        for (j = 1; j < band - 1; ++j)
        {
            el.push_back(vl.size());
            // inverted orientation FIXME
            vl.push_back(i * band + j + 1 + base);
            vl.push_back(i * band + j + base);
            vl.push_back((i + 1) * band + j + base);
            vl.push_back((i + 1) * band + j + 1 + base);
        }
    }
    return 0;
}

int
ReadASCIIDyna::ellipticKnob(float height, float groundRad, float abrasRad,
                            float winkel, float laeng1, float laeng2,
                            int angDiv, int upDiv, int downDiv,
                            vector<int> &el, vector<int> &vl,
                            vector<float> &xl, vector<float> &yl, vector<float> &zl,
                            vector<float> &nx, vector<float> &ny, vector<float> &nz)
{
    winkel *= M_PI / 180.0;
    el.clear();
    vl.clear();
    xl.clear();
    yl.clear();
    zl.clear();
    nx.clear();
    ny.clear();
    nz.clear();
    // calculate number of divisions in
    // azimuthal directions
    float divAngle = M_PI * 0.5 / angDiv;

    // zuerst nodes
    int ang;
    float angle = 0.0;
    for (ang = 0; ang <= angDiv; ++ang, angle += divAngle)
    {
        vector<float> xn, yn;
        xn.push_back(0.0);
        yn.push_back(0.0);
        float R0 = 1.0 / Laenge(cos(angle) / (0.5 * laeng1), sin(angle) / (0.5 * laeng2));
        float ratio = (laeng2 / laeng1);
        ratio *= ratio;
        float deltaX = ratio * R0 * cos(angle);
        float R1 = R0 * Laenge(cos(angle) * ratio, sin(angle));
        float beta = M_PI * 0.25 - 0.5 * winkel;
        xn.push_back(R1 - abrasRad * tan(beta));
        yn.push_back(height);
        /*
            float x1 = R1 - abrasRad * tan(beta);
            float y1 = height;
      */
        int div;
        for (div = 1; div <= upDiv; ++div)
        {
            xn.push_back(xn[1] + abrasRad * sin((2 * beta * div) / upDiv));
            yn.push_back(yn[1] - (1.0 - cos((2 * beta * div) / upDiv)) * abrasRad);
        }
        /*
            float x2 = x1 + abrasRad * sin(beta);
            float y2 = y1 - (1.0 - cos(beta))*abrasRad;
            float x3 = x1 + abrasRad * sin(2*beta);
            float y3 = y1 - (1.0 - cos(2*beta))*abrasRad;
      */
        int start = xn.size();
        for (div = 0; div <= downDiv; ++div)
        {
            xn.push_back(R1 + height * tan(winkel) + groundRad * tan(beta));
            yn.push_back(0.0);
        }
        for (div = downDiv; div >= 1; --div)
        {
            xn[start + div - 1] = xn[xn.size() - 1] - groundRad * sin((2 * beta * (downDiv - div + 1)) / downDiv);
            yn[start + div - 1] = groundRad * (1 - cos((2 * beta * (downDiv - div + 1)) / downDiv));
        }
        // now from the end...
        /*
            float x6 = R1 + height*tan(winkel) + groundRad*tan(beta);
            float y6 = 0.0;
            float x5 = x6 - groundRad*sin(beta);
            float y5 = groundRad*(1-cos(beta));
            float x4 = x6 - groundRad*sin(2*beta);
            float y4 = groundRad*(1-cos(2*beta));
      */
        // what happens if y4 > y3?
        if (yn[start] > yn[start - 1])
        {
            xn[start - 1] = 0.5 * (xn[start - 1] + xn[start]);
            yn[start - 1] = 0.5 * (yn[start - 1] + yn[start]);
            xn[start] = xn[start - 1];
            yn[start] = yn[start - 1];
        }
        float cosangle = cos(angle);
        float sinangle = sin(angle);
        float xRand = R0 * cosangle;
        float cosbeta = cosangle / Laenge(cosangle, sinangle / ratio);
        float sinbeta = sinangle / Laenge(sinangle, cosangle * ratio);
        int point;
        for (point = 1; point < xn.size(); ++point)
        {
            xl.push_back(xRand - deltaX + xn[point] * cosbeta);
            yl.push_back(xn[point] * sinbeta);
            zl.push_back(yn[point]);
        }
        /*
            xl.push_back(x1*cosangle);
            xl.push_back(x2*cosangle);
            xl.push_back(x3*cosangle);
            xl.push_back(x4*cosangle);
            xl.push_back(x5*cosangle);
            xl.push_back(x6*cosangle);
            yl.push_back(x1*sinangle);
            yl.push_back(x2*sinangle);
            yl.push_back(x3*sinangle);
            yl.push_back(x4*sinangle);
      yl.push_back(x5*sinangle);
      yl.push_back(x6*sinangle);
      zl.push_back(y1);
      zl.push_back(y2);
      zl.push_back(y3);
      zl.push_back(y4);
      zl.push_back(y5);
      zl.push_back(y6);
      */
        // and normals
        /*
            nx.push_back(0.0);
            nx.push_back(sin(beta)*cosangle);
            nx.push_back(sin(2*beta)*cosangle);
            nx.push_back(sin(2*beta)*cosangle);
            nx.push_back(sin(beta)*cosangle);
            nx.push_back(0.0);
            ny.push_back(0.0);
            ny.push_back(sin(beta)*sinangle);
            ny.push_back(sin(2*beta)*sinangle);
            ny.push_back(sin(2*beta)*sinangle);
      ny.push_back(sin(beta)*sinangle);
      ny.push_back(0.0);
      nz.push_back(1.0);
      nz.push_back(cos(beta));
      nz.push_back(cos(2*beta));
      nz.push_back(cos(2*beta));
      nz.push_back(cos(beta));
      nz.push_back(1.0);
      */
    }
    // now we add the cork
    xl.push_back(0.0);
    yl.push_back(0.0);
    zl.push_back(height);
    /*
      nx.push_back(0.0);
      ny.push_back(0.0);
      nz.push_back(1.0);
   */
    // the elements (except the cork)
    int band = upDiv + downDiv + 2;
    for (ang = 0; ang < angDiv; ++ang)
    {
        int base = ang * band;
        int basen = ang * band + band;
        int pol;
        for (pol = 0; pol < band - 1; ++pol)
        {
            el.push_back(vl.size());
            vl.push_back(base + pol);
            vl.push_back(base + pol + 1);
            vl.push_back(basen + pol + 1);
            vl.push_back(basen + pol);
        }
    }
    // add the cork
    for (ang = 0; ang < angDiv; ++ang)
    {
        int base = ang * band;
        int basen = ang * band + band;
        el.push_back(vl.size());
        vl.push_back(base);
        vl.push_back(basen);
        vl.push_back(xl.size() - 1);
        vl.push_back(xl.size() - 1);
    }
    return 0;
}
