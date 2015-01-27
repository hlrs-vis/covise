/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __opencrg_Surface_h
#define __opencrg_Surface_h

#include <iostream>
#include <vector>
#include <map>
#include <limits>
#include <cmath>
#include <osg/Image>
#include <util/coExport.h>

namespace opencrg
{

struct SurfaceNormal
{
    SurfaceNormal()
    {
        u = 0.0;
        v = 0.0;
        w = 1.0;
    }

    SurfaceNormal(float setU, float setV, float setW)
        : u(setU)
        , v(setV)
        , w(setW)
    {
    }

    void normalize()
    {
        float scale = 1 / sqrt(pow(u, 2) + pow(v, 2) + pow(w, 2));
        u = u * scale;
        v = v * scale;
        w = w * scale;
    }

    float u;
    float v;
    float w;
};

class VEHICLEUTILEXPORT Surface
{
public:
    Surface(const std::string &);
    ~Surface();

    void parseFile(const std::string &);

    const std::string &getCommentTextBlock() const;
    double getParameterValue(const std::string &) const;
    unsigned int getNumberOfLongitudinalDataElements() const;
    unsigned int getNumberOfHeightDataElements() const;
    unsigned int getNumberOfSurfaceDataLines() const;

    double gridPointElevation(unsigned int, unsigned int) const;
    double operator()(double, double) const;

    double getLongitudinalData(unsigned int, unsigned int) const;

    double getMinimumElevation() const;
    double getMaximumElevation() const;

    double getLength() const;
    double getWidth() const;

    void computeNormals();
    SurfaceNormal *getSurfaceNormal(unsigned int, unsigned int) const;

    osg::Image *createDiffuseMapTextureImage();

    osg::Image *createParallaxMapTextureImage();

    void computeConeStepMappingRatio();
    osg::Image *createConeStepMapTextureImage();

    osg::Image *createPavementTextureImage();

protected:
    void parseRoadParameterLine(const std::string &);
    void parseDataDefinitionLine(const std::string &);
    void parseSurfaceDataLine(const std::string &);

    int warpIndex(int, const unsigned int &width);

    std::string commentTextBlock;

    std::map<std::string, double> roadParameterValueMap;

    std::vector<std::string> surfaceLongDataTypeVector;
    std::vector<std::string> surfaceHeightTypeVector;

    std::vector<float *> surfaceLongDataLineVector;
    std::vector<float *> surfaceHeightDataLineVector;
    std::vector<SurfaceNormal *> surfaceNormalDataLineVector;
    std::vector<float *> surfaceConeRatioDataLineVector;
    unsigned int numLongDataElements;
    unsigned int numHeightDataElements;

    double length;
    double width;

    double reference_line_start_u;
    double reference_line_end_u;
    double reference_line_increment;
    double long_section_v_right;
    double long_section_v_left;
    double long_section_v_increment;
    double inverseGridCellArea;

    double minElev;
    double maxElev;

    enum ParseType
    {
        ParseTypeNone,
        ParseTypeCommentTextBlock,
        ParseTypeRoadParameters,
        ParseTypeDataDefinition,
        ParseTypeSurfaceData,
        ParseTypeBinarySurfaceData
    };
};

inline const std::string &Surface::getCommentTextBlock() const
{
    return commentTextBlock;
}

inline double Surface::getParameterValue(const std::string &name) const
{
    std::map<std::string, double>::const_iterator it = roadParameterValueMap.find(name);
    if (it == roadParameterValueMap.end())
    {
        return std::numeric_limits<double>::signaling_NaN();
    }
    else
    {
        return it->second;
    }
}

inline unsigned int Surface::getNumberOfLongitudinalDataElements() const
{
    return numLongDataElements;
}

inline unsigned int Surface::getNumberOfHeightDataElements() const
{
    return numHeightDataElements;
}

inline unsigned int Surface::getNumberOfSurfaceDataLines() const
{
    return surfaceLongDataLineVector.size();
}

inline double Surface::gridPointElevation(unsigned int u, unsigned int v) const
{
    return surfaceHeightDataLineVector[u][v];
}

inline double Surface::operator()(double u, double v) const
{
    double uRel = (u) / reference_line_increment;
    int u1 = (int)(floor(uRel));
    if (u1 < 0)
        u1 = 0;
    else if (u1 >= (int)surfaceHeightDataLineVector.size())
        u1 = surfaceHeightDataLineVector.size() - 1;
    int u2 = (int)(ceil(uRel));
    if (u2 < 0)
        u2 = 0;
    else if (u2 >= (int)surfaceHeightDataLineVector.size())
        u2 = surfaceHeightDataLineVector.size() - 1;

    double vRel = (v - long_section_v_right) / long_section_v_increment;
    int v1 = (int)(floor(vRel));
    if (v1 < 0)
        v1 = 0;
    else if (v1 >= (int)numHeightDataElements)
        v1 = numHeightDataElements - 1;
    int v2 = (int)(ceil(vRel));
    if (v2 < 0)
        v2 = 0;
    if (v2 >= (int)numHeightDataElements)
        v2 = numHeightDataElements - 1;

    /*double height = inverseGridCellArea*
   (     (surfaceHeightDataLineVector[u1][v1] - (maxElev+minElev)/2) * (reference_line_increment*u2-u)*(long_section_v_increment*v2-v)
      +  (surfaceHeightDataLineVector[u2][v1] - (maxElev+minElev)/2) * (u-reference_line_increment*u1)*(long_section_v_increment*v2-v)
      +  (surfaceHeightDataLineVector[u1][v2] - (maxElev+minElev)/2) * (reference_line_increment*u2-u)*(v-long_section_v_increment*v1)
      +  (surfaceHeightDataLineVector[u2][v2] - (maxElev+minElev)/2) * (u-reference_line_increment*u1)*(v-long_section_v_increment*v1)
   );*/
    double height = surfaceHeightDataLineVector[u1][v1] - (maxElev + minElev) / 2;
    if (height != height)
        height = 0.0;

    //std::cerr << "Surface::operator(): u: " << u << ", v: " << v << ", uRel: " << uRel << ", vRel: " << vRel << ", u1: " << u1 << ", u2: " << u2 << ", v1: " << v1 << ", v2: " << v2 << ", height: " << height << std::endl;

    return height;
}

inline double Surface::getLongitudinalData(unsigned int u, unsigned int v) const
{
    return (surfaceLongDataLineVector[u])[v];
}

inline double Surface::getMinimumElevation() const
{
    return minElev;
}

inline double Surface::getMaximumElevation() const
{
    return maxElev;
}

inline double Surface::getLength() const
{
    return length;
}

inline double Surface::getWidth() const
{
    return width;
}

inline int Surface::warpIndex(int x, const unsigned int &width)
{
    // take the info and compute the array index

    // x-tile
    while (x < 0)
        x += width;
    while (x >= (int)width)
        x -= width;

    return x;
}
}

VEHICLEUTILEXPORT std::ostream &operator<<(std::ostream &, const opencrg::Surface &);

#endif
