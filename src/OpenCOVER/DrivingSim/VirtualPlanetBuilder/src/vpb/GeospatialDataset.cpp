/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* -*-c++-*- VirtualPlanetBuilder - Copyright (C) 1998-2009 Robert Osfield
 *
 * This library is open source and may be redistributed and/or modified under
 * the terms of the OpenSceneGraph Public License (OSGPL) version 0.0 or
 * (at your option) any later version.  The full license is in LICENSE file
 * included with this distribution, and on the openscenegraph.org website.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * OpenSceneGraph Public License for more details.
*/

#include <vpb/System>
#include <osg/Notify>
#include <OpenThreads/ScopedLock>

using namespace vpb;

GeospatialDataset::GeospatialDataset(const std::string &filename, AccessMode accessMode)
{
    updateTimeStamp();
    _dataset = (GDALDataset *)GDALOpen(filename.c_str(), accessMode == READ_ONLY ? GA_ReadOnly : GA_Update);

    //osg::notify(osg::NOTICE)<<"GDALOpen("<<filename<<") = "<<_dataset<<std::endl;
}

GeospatialDataset::GeospatialDataset(GDALDataset *dataset)
{
    //osg::notify(osg::NOTICE)<<"GDALOpen(dataset)="<<_dataset<<std::endl;

    updateTimeStamp();
    _dataset = dataset;
}

GeospatialDataset::~GeospatialDataset()
{
    //osg::notify(osg::NOTICE)<<"GDALClose("<<_dataset<<")"<<std::endl;
    if (_dataset)
        GDALClose(_dataset);
}

CPLErr GeospatialDataset::GetGeoTransform(double *ptr)
{
    updateTimeStamp();
    return _dataset->GetGeoTransform(ptr);
}

int GeospatialDataset::GetRasterXSize(void)
{
    updateTimeStamp();
    return _dataset->GetRasterXSize();
}

int GeospatialDataset::GetRasterYSize(void)
{
    updateTimeStamp();
    return _dataset->GetRasterYSize();
}

int GeospatialDataset::GetRasterCount(void)
{
    updateTimeStamp();
    return _dataset->GetRasterCount();
}

GDALRasterBand *GeospatialDataset::GetRasterBand(int band)
{
    updateTimeStamp();
    return _dataset->GetRasterBand(band);
}

int GeospatialDataset::GetOverviewCount(int band)
{
    updateTimeStamp();
    GDALRasterBand *raster = _dataset->GetRasterBand(band);
    return band ? raster->GetOverviewCount() : 0;
}

bool GeospatialDataset::containsOverviews()
{
    int count = GetRasterCount();
    for (int i = 1; i <= count; ++i)
    {
        if (GetOverviewCount(i) > 0)
            return true;
    }
    return false;
}

int GeospatialDataset::GetGCPCount()
{
    updateTimeStamp();
    return _dataset->GetGCPCount();
}

const char *GeospatialDataset::GetGCPProjection()
{
    updateTimeStamp();
    return _dataset->GetGCPProjection();
}

const GDAL_GCP *GeospatialDataset::GetGCPs()
{
    updateTimeStamp();
    return _dataset->GetGCPs();
}

CPLErr GeospatialDataset::RasterIO(GDALRWFlag a, int b, int c, int d, int e,
                                   void *f, int g, int h, GDALDataType i,
                                   int j, int *k, int l, int m, int n)
{
    updateTimeStamp();
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    return _dataset->RasterIO(a, b, c, d, e, f, g, h, i, j, k, l, m, n);
}

CPLErr GeospatialDataset::BuildOverviews(const char *a, int b, int *c,
                                         int d, int *e, GDALProgressFunc f, void *g)
{
    updateTimeStamp();
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    return _dataset->BuildOverviews(a, b, c, d, e, f, g);
}

const char *GeospatialDataset::GetProjectionRef(void)
{
    updateTimeStamp();
    return _dataset->GetProjectionRef();
}
