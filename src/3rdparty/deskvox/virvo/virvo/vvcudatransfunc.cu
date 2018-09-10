// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 2010 University of Cologne
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the 
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA


__global__ void makePreintLUTCorrectKernel(int width, uchar4 *__restrict__ preIntTable,
        float thickness, float min, float max, const float4 *__restrict__ rgba)
{
  const int minLookupSteps = 2;
  const int addLookupSteps = 1;

  const int sb = blockIdx.x;
  const int sf = threadIdx.x;

  int n=minLookupSteps+addLookupSteps*abs(sb-sf);
  float stepWidth = 1.f/n;
  float r=0.f, g=0.f, b=0.f, tau=0.f;
  for (int i=0; i<n; ++i)
  {
      const float s = sf+(sb-sf)*i*stepWidth;
      const int is = (int)s;
      const float fract_s = s-floorf(s);
      const float tauc = thickness*stepWidth*(rgba[is].w*fract_s+rgba[is+1].w*(1.f-fract_s));
      const float e_tau = expf(-tau);
#ifdef STANDARD
      /* standard optical model: r,g,b densities are multiplied with opacity density */
      const float rc = e_tau*tauc*(rgba[is].x*fract_s+rgba[is+1].x*(1.f-fract_s));
      const float gc = e_tau*tauc*(rgba[is].y*fract_s+rgba[is+1].y*(1.f-fract_s));
      const float bc = e_tau*tauc*(rgba[is].z*fract_s+rgba[is+1].z*(1.f-fract_s));

#else
      /* Willhelms, Van Gelder optical model: r,g,b densities are not multiplied */
      const float rc = e_tau*stepWidth*(rgba[is].x*fract_s+rgba[is+1].x*(1.f-fract_s));
      const float gc = e_tau*stepWidth*(rgba[is].y*fract_s+rgba[is+1].y*(1.f-fract_s));
      const float bc = e_tau*stepWidth*(rgba[is].z*fract_s+rgba[is+1].z*(1.f-fract_s));
#endif

      r = r+rc;
      g = g+gc;
      b = b+bc;
      tau = tau + tauc;
  }

  saturate(r);
  saturate(g);
  saturate(b);

  preIntTable[sf*width+sb] = make_uchar4(r*255.99f,g*255.99f, b*255.99f,(1.f-__expf(-tau))*255.99f); // using expf causes ptxas to die on windows quda 6.5
}


extern "C" void CallCudaTransFuncKernel(int width, uchar4* preIntTable, float thickness, float min, float max, const float4* rgba)
{
    makePreintLUTCorrectKernel<<<width, width>>>(width, preIntTable, thickness, min, max, rgba);
}


// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
