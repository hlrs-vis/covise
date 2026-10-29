/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * Author: Siva Karthikeya Mandarapu
 * License: LGPL 2+ */
#include "VrmlNodeRasterProbe.h"

#include <cmath>
#include <filesystem>
#include <iostream>

#include <gdal_priv.h>
#include <vrml97/vrml/System.h>

#include "RasterProbe.h"

unsigned VrmlNodeRasterProbe::s_enableCounter = 0;

VrmlNodeRasterProbe::VrmlNodeRasterProbe(VrmlScene *scene)
    : VrmlNodeChild(scene, typeName())
{
    RasterProbe::addProbe(this);
}

VrmlNodeRasterProbe::VrmlNodeRasterProbe(const VrmlNodeRasterProbe &n)
    : VrmlNodeChild(n)
{
    RasterProbe::addProbe(this);
}

VrmlNodeRasterProbe::~VrmlNodeRasterProbe()
{
    RasterProbe::removeProbe(this);
    if (m_dataset)
        GDALClose(GDALDataset::ToHandle(m_dataset));
}

void VrmlNodeRasterProbe::initFields(VrmlNodeRasterProbe *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
        exposedField("url", &VrmlNodeRasterProbe::d_url),
        exposedField("unit", &VrmlNodeRasterProbe::d_unit),
        exposedField("enabled", &VrmlNodeRasterProbe::d_enabled, [node](auto f)
            {
                if (node->d_enabled.get())
                    node->m_enableSerial = ++s_enableCounter;
                else
                    node->setValue(false, 0.0, true); }));

    if (t)
    {
        t->addEventOut("value_changed", VrmlField::SFFLOAT);
        t->addEventOut("hasValue_changed", VrmlField::SFBOOL);
    }
}

const char *VrmlNodeRasterProbe::typeName()
{
    return "RasterProbe";
}

void VrmlNodeRasterProbe::addToScene(VrmlScene *scene, const char *relativeUrl)
{
    m_relativeUrl = relativeUrl == NULL ? "" : std::string(relativeUrl);
    VrmlNodeChild::addToScene(scene, relativeUrl);
}

// opens the raster on first use and whenever the url changed
bool VrmlNodeRasterProbe::open()
{
    if (m_openedUrl == d_url.get())
        return m_dataset != nullptr;

    if (m_dataset)
        GDALClose(GDALDataset::ToHandle(m_dataset));
    m_dataset = nullptr;
    m_openedUrl = d_url.get();

    std::filesystem::path path(m_openedUrl);
    if (path.is_relative() && !std::filesystem::exists(path))
    {
        path = std::filesystem::path(m_relativeUrl).parent_path() / path;
    }

    m_dataset = GDALDataset::Open(path.string().c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY);
    double geoTransform[6];
    if (!m_dataset || m_dataset->GetGeoTransform(geoTransform) != CE_None)
    {
        std::cerr << "[RasterProbe] Cannot open raster: " << path << std::endl;
        if (m_dataset)
            GDALClose(GDALDataset::ToHandle(m_dataset));
        m_dataset = nullptr;
        return false;
    }
    GDALInvGeoTransform(geoTransform, m_invGeoTransform);
    return true;
}

bool VrmlNodeRasterProbe::sample(double easting, double northing, double &value)
{
    if (!open())
        return false;

    double px, py;
    GDALApplyGeoTransform(m_invGeoTransform, easting, northing, &px, &py);
    int x = (int)std::floor(px);
    int y = (int)std::floor(py);
    if (x < 0 || y < 0 || x >= m_dataset->GetRasterXSize() || y >= m_dataset->GetRasterYSize())
        return false;

    GDALRasterBand *band = m_dataset->GetRasterBand(1);
    int hasNoData = 0;
    double noData = band->GetNoDataValue(&hasNoData);
    if (band->RasterIO(GF_Read, x, y, 1, 1, &value, 1, 1, GDT_Float64, 0, 0) != CE_None)
        return false;
    return !std::isnan(value) && !(hasNoData && value == noData);
}

void VrmlNodeRasterProbe::setValue(bool valid, double value, bool force)
{
    if (!force && valid == d_hasValue.get() && (!valid || (float)value == d_value.get()))
        return;

    double time = System::the->time();
    if (force)
        eventOut(time, "unit", d_unit); // the display may show another probe's unit
    if (valid)
    {
        d_value.set(value);
        eventOut(time, "value_changed", d_value);
    }
    d_hasValue.set(valid);
    eventOut(time, "hasValue_changed", d_hasValue);
}
