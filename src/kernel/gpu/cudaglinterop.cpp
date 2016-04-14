/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstring>
#include <cstdio>

#include "cudaglinterop.h"
#ifdef WIN32
#include <winsock2.h>
#include "windows.h"
#endif

#ifdef HAVE_X11
#include <GL/glx.h>
#endif

#ifdef HAVE_CUDA
#include <cuda_gl_interop.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include <config/CoviseConfig.h>

bool initCudaGlInterop()
{
    static bool done = false;
    if (done)
        return true;

#ifdef HAVE_CUDA
#ifdef HAVE_X11
    GLXContext ctx = glXGetCurrentContext();
    if (ctx != NULL)
    {
        Display *dsp = XOpenDisplay(NULL);
        if (dsp)
            XSynchronize(dsp, True);
        if (!dsp || !glXIsDirect(dsp, ctx))
        {
            fprintf(stderr, "initCudaGlInterop: no CUDA/GL interoperability possible\n");
            return false;
        }
    }
#endif

    int dev = -1;
    dev = covise::coCoviseConfig::getInt("device", "System.CUDA", -1);
    if (dev == -1)
    {
        int num_devices, device;
        cudaGetDeviceCount(&num_devices);
        if (num_devices >= 1)
        {
            int max_multiprocessors = 0, max_device = 0;
            for (device = 0; device < num_devices; device++)
            {
                cudaDeviceProp properties;
                cudaGetDeviceProperties(&properties, device);
                if (max_multiprocessors < properties.multiProcessorCount)
                {
                    max_multiprocessors = properties.multiProcessorCount;
                    max_device = device;
                }
            }
            cudaSetDevice(max_device);
            dev = max_device;
        }
    }
    else
    {
        cudaSetDevice(dev);
    }
    /*cudaDeviceProp prop;
   memset(&prop, 0, sizeof(prop));
   prop.major = 1;
   prop.minor = 0;

   if(cudaChooseDevice(&dev, &prop) != cudaSuccess)
   {
      fprintf(stderr, "initCudaGlInterop: error choosing device\n");
   }*/
    if (dev == -1)
    {
        fprintf(stderr, "initCudaGlInterop: did not find a CUDA device\n");
    }
    else if (cudaGLSetGLDevice(dev) != cudaSuccess)
    {
        fprintf(stderr, "initCudaGlInterop: error setting GL device %d\n", dev);
    }
#endif

    done = true;
    return true;
}
