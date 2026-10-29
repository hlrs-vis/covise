/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * Author: Siva Karthikeya Mandarapu
 * License: LGPL 2+ */

#ifndef _VRMLNODERasterProbe_
#define _VRMLNODERasterProbe_

#include <string>

#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFBool.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFString.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>

class GDALDataset;

using namespace vrml;

// Raster value under the pointer, sent as value_changed/hasValue_changed while enabled.
// The raster must be in the scene's GeoData projection (EPSG:25832 by default).
class VrmlNodeRasterProbe : public VrmlNodeChild
{
public:
    VrmlNodeRasterProbe(VrmlScene *scene = 0);
    VrmlNodeRasterProbe(const VrmlNodeRasterProbe &n);
    virtual ~VrmlNodeRasterProbe();

    static void initFields(VrmlNodeRasterProbe *node, vrml::VrmlNodeType *t);
    static const char *typeName();

    virtual void addToScene(VrmlScene *scene, const char *relativeUrl);

    bool enabled() const { return d_enabled.get(); }
    unsigned enableSerial() const { return m_enableSerial; }

    // false outside the raster or on NoData
    bool sample(double easting, double northing, double &value);
    // sends the value if it changed, or always (with the unit) with force
    void setValue(bool valid, double value, bool force);

private:
    bool open();

    VrmlSFString d_url { "" };
    VrmlSFString d_unit { "" };
    VrmlSFBool d_enabled { false };
    VrmlSFFloat d_value { 0.0f }; // value_changed
    VrmlSFBool d_hasValue { false }; // hasValue_changed

    std::string m_relativeUrl { "" };

    unsigned m_enableSerial = 0; // higher = enabled later
    static unsigned s_enableCounter;

    GDALDataset *m_dataset = nullptr;
    std::string m_openedUrl;
    double m_invGeoTransform[6];
};
#endif
