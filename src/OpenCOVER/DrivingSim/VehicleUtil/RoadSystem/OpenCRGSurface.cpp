/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "OpenCRGSurface.h"
#include "../opencrg/crgSurface.h"

//#include <fftw3.h>

OpenCRGSurface::OpenCRGSurface(const std::string &setFilename, double setSStart, double setSEnd)
    : filename(setFilename)
    , dataSetId(crgLoaderReadFile(filename.c_str()))
    , sStart(setSStart)
    , sEnd(setSEnd)
    , parallaxMap(NULL)
    , pavementTextureImage(NULL)
{
    if (dataSetId != 0)
    {
        contactPointId = crgContactPointCreate(dataSetId);
    }
}

OpenCRGSurface::OpenCRGSurface(const std::string &setFilename, double setSStart, double setSEnd, SurfaceOrientation setOrient, double setSOff, double setTOff, double setZOff, double setZScale, double setHOff)
    : filename(setFilename)
    , dataSetId(crgLoaderReadFile(filename.c_str()))
    , sStart(setSStart)
    , sEnd(setSEnd)
    , parallaxMap(NULL)
    , pavementTextureImage(NULL)
{
    if (dataSetId != 0)
    {
        //crgDataSetModifierSetInt(dataSetId, dCrgModRefPointOrient, setOrient);
        crgDataSetModifierSetDouble(dataSetId, dCrgModRefPointUOffset, setSOff);
        crgDataSetModifierSetDouble(dataSetId, dCrgModRefPointVOffset, setTOff);
        crgDataSetModifierSetDouble(dataSetId, dCrgModRefLineOffsetZ, setZOff);
        crgDataSetModifierSetDouble(dataSetId, dCrgModScaleZ, setZScale);
        crgDataSetModifierSetDouble(dataSetId, dCrgModRefLineOffsetPhi, setHOff);

        crgDataSetModifierSetInt(dataSetId, dCrgModGridNaNMode, dCrgGridNaNKeepLast);

        crgDataSetModifiersApply(dataSetId);

        contactPointId = crgContactPointCreate(dataSetId);
    }
}

osg::Image *OpenCRGSurface::getParallaxMap()
{
    if (!parallaxMap)
    {
        opencrg::Surface surface(filename);
        parallaxMap = createParallaxMapTextureImage();
        return parallaxMap;
    }
    else
    {
        return parallaxMap;
    }
}

osg::Image *OpenCRGSurface::getPavementTextureImage()
{
    if (!pavementTextureImage)
    {
        pavementTextureImage = createDiffuseMapTextureImage();
        return pavementTextureImage;
    }
    else
    {
        return pavementTextureImage;
    }
}

double OpenCRGSurface::height(double s, double t)
{
    if (dataSetId == 0 || s < sStart || s > sEnd)
        return 0.0;

    double z = 0.0;
    crgEvaluv2z(contactPointId, s, t, &z);

    return z;
}

double OpenCRGSurface::getLength()
{
    double uMin;
    double uMax;
    crgDataSetGetURange(dataSetId, &uMin, &uMax);
    return uMax - uMin;
}

double OpenCRGSurface::getWidth()
{
    double vMin;
    double vMax;
    crgDataSetGetVRange(dataSetId, &vMin, &vMax);
    return vMax - vMin;
}

osg::Image *OpenCRGSurface::createDiffuseMapTextureImage()
{
    double uMin, uMax;
    double vMin, vMax;
    double uInc, vInc;
    crgDataSetGetURange(dataSetId, &uMin, &uMax);
    crgDataSetGetVRange(dataSetId, &vMin, &vMax);
    crgDataSetGetIncrements(dataSetId, &uInc, &vInc);

    unsigned int numUNodes = (unsigned int)ceil((uMax - uMin) / uInc - 1e-1) + 1;
    unsigned int numVNodes = (unsigned int)ceil((vMax - vMin) / vInc - 1e-1) + 1;

    /*double* heightArray = (double*)fftw_malloc(sizeof(double)*numUNodes*numVNodes);
   fftw_complex* frequencyArray = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*numUNodes*numVNodes);
   fftw_plan plan_forward = fftw_plan_dft_r2c_2d(numUNodes, numVNodes, heightArray, frequencyArray, FFTW_MEASURE);
   fftw_plan plan_reverse = fftw_plan_dft_c2r_2d(numUNodes, numVNodes, frequencyArray, heightArray, FFTW_MEASURE);

   for(int u=0; u<numUNodes; ++u) {
      for(unsigned int v=0; v<numVNodes; ++v) {
         double s = uMin + u*uInc;
         double t = vMin + v*vInc;
         crgEvaluv2z(contactPointId, s, t, heightArray + u*numVNodes + v);
      }
   }
   fftw_execute(plan_forward);
   double freq_limit = 1.0/0.02;
   for(int u=0; u<numUNodes; ++u) {
      for(unsigned int v=0; v<numVNodes; ++v) {
         double freq_u = u/uInc;
         double freq_v = v/vInc;
			//if(freq_u < freq_limit || freq_v < freq_limit) {
			if(u < 0.5*numUNodes) {
				(*(frequencyArray + u*numVNodes + v))[0] = 0.0;
				(*(frequencyArray + u*numVNodes + v))[1] = 0.0;
			}
		}
	}
   fftw_execute(plan_reverse);*/

    osg::Image *diffuseMap = new osg::Image();

    diffuseMap->allocateImage(numUNodes, numVNodes, 1, GL_RGB, GL_UNSIGNED_BYTE);

    /*for(int u=0; u<numUNodes; ++u) {
      for(unsigned int v=0; v<numVNodes; ++v) {
         double z = *(heightArray + u*numVNodes + v)/((double)(numUNodes*numVNodes));
         int zColor = 86 + z*1e4; if(zColor>255) zColor = 255; if(zColor<0) zColor = 0;
         *(diffuseMap->data(u, v)+0) = zColor;
         *(diffuseMap->data(u, v)+1) = 86;// + z*1e3;
         *(diffuseMap->data(u, v)+2) = 86;// + z*1e3;
      }
   }*/

    Lowpass<double> lpass(500.0, vInc);
    Highpass<double> hpass(100.0, vInc);
    for (unsigned int u = 0; u < numUNodes; ++u)
    {
        for (unsigned int v = 0; v < numVNodes; ++v)
        {
            double s = uMin + u * uInc;
            double t = vMin + v * vInc;
            double z;
            crgEvaluv2z(contactPointId, s, t, &z);
            z = lpass(z);
            z = hpass(z);

            int zColorPlus = 128 + z * 5e4;
            if (zColorPlus > 255)
                zColorPlus = 255;
            if (zColorPlus < 0)
                zColorPlus = 0;
            //int zColorMinus = 128 - z*5e4; if(zColorMinus>255) zColorMinus = 255; if(zColorMinus<0) zColorMinus = 0;
            *(diffuseMap->data(u, v) + 0) = zColorPlus;
            *(diffuseMap->data(u, v) + 1) = zColorPlus;
            *(diffuseMap->data(u, v) + 2) = zColorPlus;
        }
    }

    /*fftw_destroy_plan(plan_reverse);
   fftw_destroy_plan(plan_forward);
   fftw_free(frequencyArray);
   fftw_free(heightArray);*/

    return diffuseMap;
}

osg::Image *OpenCRGSurface::createParallaxMapTextureImage()
{
    double uMin, uMax;
    double vMin, vMax;
    double uInc, vInc;
    crgDataSetGetURange(dataSetId, &uMin, &uMax);
    crgDataSetGetVRange(dataSetId, &vMin, &vMax);
    crgDataSetGetIncrements(dataSetId, &uInc, &vInc);
    //std::cout << "uMin: " << uMin << ", uMax: " << uMax << std::endl;
    //std::cout << "vMin: " << vMin << ", vMax: " << vMax << std::endl;
    //std::cout << "uInc: " << uInc << ", vInc: " << vInc << std::endl;

    int uMode, vMode;
    crgContactPointOptionGetInt(contactPointId, dCrgCpOptionBorderModeU, &uMode);
    crgContactPointOptionGetInt(contactPointId, dCrgCpOptionBorderModeV, &vMode);
    //std::cout << "uMode: " << uMode << ", vMode : " << vMode << std::endl;

    unsigned int numUNodes = (unsigned int)ceil((uMax - uMin) / uInc - 1e-1) + 1;
    unsigned int numVNodes = (unsigned int)ceil((vMax - vMin) / vInc - 1e-1) + 1;

    osg::Image *parallaxMap = new osg::Image();

    parallaxMap->allocateImage(numUNodes, numVNodes, 1, GL_RGBA, GL_UNSIGNED_BYTE);

    //CrgDataStruct* dataSet = crgDataSetAccess(dataSetId);
    //double minElev = dataSet->util.zMin;
    //double maxElev = dataSet->util.zMax;

    for (unsigned int u = 0; u < numUNodes; ++u)
    {
        for (unsigned int v = 0; v < numVNodes; ++v)
        {
            double s = uMin + u * uInc;
            double t = vMin + v * vInc;
            double x13 = 2 * uInc;
            double y24 = 2 * vInc;
            double z1;
            crgEvaluv2z(contactPointId, s + uInc, t, &z1);
            double z2;
            crgEvaluv2z(contactPointId, s, t + vInc, &z2);
            double z3;
            crgEvaluv2z(contactPointId, s - uInc, t, &z3);
            double z4;
            crgEvaluv2z(contactPointId, s, t - vInc, &z4);
            double nx = -y24 * (z1 - z3);
            double ny = -x13 * (z2 - z4);
            double nz = x13 * y24;
            double inv_m_n = 1.0 / sqrt(nx * nx + ny * ny + nz * nz);
            nx *= inv_m_n;
            ny *= inv_m_n;
            nz *= inv_m_n;

            *(parallaxMap->data(u, v) + 0) = (unsigned char)((nx + 1.0) * 0.5 * 255.0);
            *(parallaxMap->data(u, v) + 1) = (unsigned char)((ny + 1.0) * 0.5 * 255.0);
            *(parallaxMap->data(u, v) + 2) = (unsigned char)((nz + 1.0) * 0.5 * 255.0);

            //double minMaxScale = 255.0/(maxElev-minElev);

            double z;
            crgEvaluv2z(contactPointId, s, t, &z);
            //*(parallaxMap->data(u, v)+3) = (unsigned char)((z-minElev)*minMaxScale);
            *(parallaxMap->data(u, v) + 3) = (unsigned char)((z + 1.0) * 0.5 * 255.0);
        }
    }

    return parallaxMap;
}

OpenCRGSurface::SurfaceWrapMode OpenCRGSurface::getSurfaceWrapModeU()
{
    OpenCRGSurface::SurfaceWrapMode uMode;
    crgContactPointOptionGetInt(contactPointId, dCrgCpOptionBorderModeU, (int *)(&uMode));
    return uMode;
}

OpenCRGSurface::SurfaceWrapMode OpenCRGSurface::getSurfaceWrapModeV()
{
    OpenCRGSurface::SurfaceWrapMode vMode;
    crgContactPointOptionGetInt(contactPointId, dCrgCpOptionBorderModeV, (int *)(&vMode));
    return vMode;
}
