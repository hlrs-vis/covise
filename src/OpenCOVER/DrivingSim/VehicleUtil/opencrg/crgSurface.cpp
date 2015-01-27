/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "crgSurface.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <cctype>
#include <cmath>

#include "../gealg/GeometricAlgebra.h"

std::ostream &operator<<(std::ostream &os, const opencrg::Surface &surface)
{
    return os << surface.getCommentTextBlock();
}

namespace opencrg
{

Surface::Surface(const std::string &filename)
    : numLongDataElements(0)
    , numHeightDataElements(0)
    , length(0.0)
    , width(0.0)
    , reference_line_start_u(0.0)
    , reference_line_end_u(0.0)
    , reference_line_increment(0.0)
    , long_section_v_right(0.0)
    , long_section_v_left(0.0)
    , long_section_v_increment(0.0)
    , inverseGridCellArea(9e10)
    , minElev(std::numeric_limits<double>::infinity())
    , maxElev(-std::numeric_limits<double>::infinity())
{
    parseFile(filename);
}

Surface::~Surface()
{
    for (unsigned int lineIt = 0; lineIt < surfaceLongDataLineVector.size(); ++lineIt)
    {
        delete[] surfaceLongDataLineVector[lineIt];
    }
    for (unsigned int lineIt = 0; lineIt < surfaceHeightDataLineVector.size(); ++lineIt)
    {
        delete[] surfaceHeightDataLineVector[lineIt];
    }
}

void Surface::parseFile(const std::string &filename)
{
    std::ifstream crgFile;
    crgFile.open(filename.c_str(), std::ios::in | std::ios::binary);

    unsigned int mode = ParseTypeNone;

    std::string line;
    while (crgFile.good())
    {
        if (mode != ParseTypeBinarySurfaceData)
        {
            std::getline(crgFile, line);
            if (line.substr(0, 1) == "$")
            {
                std::string word = line.substr(1, line.find_first_of(" ", 1) - 1);
                std::transform(word.begin(), word.end(), word.begin(), (int (*)(int))std::tolower);

                if (word == "ct")
                {
                    mode = ParseTypeCommentTextBlock;
                }
                else if (word == "road_crg")
                {
                    mode = ParseTypeRoadParameters;
                }
                else if (word == "kd_definition")
                {
                    mode = ParseTypeDataDefinition;
                }
                else if (word.substr(0, 1) == "$")
                {
                    if (word.find_first_of('0') != word.npos)
                    {
                        mode = ParseTypeSurfaceData;
                    }
                    else
                    {
                        mode = ParseTypeBinarySurfaceData;
                    }
                    /*numSurfaceDataLines =
                  (unsigned int)(ceil((getParameterValue("reference_line_end_u")-getParameterValue("reference_line_start_u")) / 
                           getParameterValue("reference_line_increment"))) + 1;
               if(numSurfaceDataLines==numSurfaceDataLines) {
                  surfaceLongDataMatrix = new float*[numSurfaceDataLines];
                  surfaceHeightMatrix = new float*[numSurfaceDataLines];
               }*/
                }
                else if (word.substr(0, 1) == "!")
                {
                    mode = ParseTypeNone;
                }
                else
                {
                    mode = ParseTypeNone;
                }
            }
            else if (mode != ParseTypeSurfaceData && line.substr(0, 1) == "*")
            {
                //do nothing
            }
            else
            {
                switch (mode)
                {
                case ParseTypeCommentTextBlock:
                    commentTextBlock.append(line);
                    commentTextBlock.append("\n");
                    break;
                case ParseTypeRoadParameters:
                    parseRoadParameterLine(line);
                    break;
                case ParseTypeDataDefinition:
                    parseDataDefinitionLine(line);
                    break;
                case ParseTypeSurfaceData:
                    parseSurfaceDataLine(line);
                    break;
                default:
                    break;
                }
            }
        }
        else
        {
            float *longDataElementArray = new float[numLongDataElements];
            surfaceLongDataLineVector.push_back(longDataElementArray);
            float *heightDataElementArray = new float[numHeightDataElements];
            surfaceHeightDataLineVector.push_back(heightDataElementArray);

            for (unsigned int i = 0; i < numLongDataElements; ++i)
            {
                float value;
                crgFile.read((char *)(&value), sizeof(float));
                *((uint8_t *)(&(longDataElementArray[i])) + 0) = *((uint8_t *)(&value) + 3);
                *((uint8_t *)(&(longDataElementArray[i])) + 1) = *((uint8_t *)(&value) + 2);
                *((uint8_t *)(&(longDataElementArray[i])) + 2) = *((uint8_t *)(&value) + 1);
                *((uint8_t *)(&(longDataElementArray[i])) + 3) = *((uint8_t *)(&value) + 0);
                //std::cout << "(" << surfaceLongDataLineVector.size()-1 << "," << i << "): " << longDataElementArray[i] << std::endl;
            }
            for (unsigned int i = 0; i < numHeightDataElements; ++i)
            {
                float value;
                crgFile.read((char *)(&value), sizeof(float));
                *((uint8_t *)(&(heightDataElementArray[i])) + 0) = *((uint8_t *)(&value) + 3);
                *((uint8_t *)(&(heightDataElementArray[i])) + 1) = *((uint8_t *)(&value) + 2);
                *((uint8_t *)(&(heightDataElementArray[i])) + 2) = *((uint8_t *)(&value) + 1);
                *((uint8_t *)(&(heightDataElementArray[i])) + 3) = *((uint8_t *)(&value) + 0);
                if (heightDataElementArray[i] < minElev)
                {
                    minElev = heightDataElementArray[i];
                }
                if (heightDataElementArray[i] > maxElev)
                {
                    maxElev = heightDataElementArray[i];
                }
            }
            //std::cout << "(" << surfaceDataLineIterator << "," << 100 << "): " << heightDataElementArray[100] << std::endl;
        }
    }

    reference_line_start_u = getParameterValue("reference_line_start_u");
    reference_line_end_u = getParameterValue("reference_line_end_u");
    reference_line_increment = getParameterValue("reference_line_increment");
    long_section_v_right = getParameterValue("long_section_v_right");
    long_section_v_left = getParameterValue("long_section_v_left");
    long_section_v_increment = getParameterValue("long_section_v_increment");
    inverseGridCellArea = 1 / (reference_line_increment * long_section_v_increment);

    length = (double)(surfaceHeightDataLineVector.size() - 1) * reference_line_increment;
    width = (double)(numHeightDataElements - 1) * long_section_v_increment;
    //std::cout << "crgSurface: lenght: " << length << ", width: " << width << ", cell length: " << reference_line_increment << ", cell width: " << long_section_v_increment << std::endl;

    crgFile.close();
}

SurfaceNormal *Surface::getSurfaceNormal(unsigned int u, unsigned int v) const
{
    return &surfaceNormalDataLineVector[u][v];
}

void Surface::parseRoadParameterLine(const std::string &line)
{
    size_t equalPos = line.find_first_of("=");

    std::string name = line.substr(0, equalPos);
    name.erase(0, name.find_first_not_of(" "));
    size_t firstSpaceInEnd = name.find_first_of(" ");
    if (firstSpaceInEnd != name.npos)
    {
        name.erase(firstSpaceInEnd);
    }
    std::transform(name.begin(), name.end(), name.begin(), (int (*)(int))std::tolower);

    std::string valueString = line.substr(equalPos + 1);
    valueString.erase(0, valueString.find_first_not_of(" "));
    firstSpaceInEnd = valueString.find_first_of(" !");
    if (firstSpaceInEnd != valueString.npos)
    {
        valueString.erase(firstSpaceInEnd);
    }
    double value;
    std::stringstream(valueString) >> value;

    roadParameterValueMap[name] = value;
}

void Surface::parseDataDefinitionLine(const std::string &line)
{
    if (line.substr(0, 2) == "D:")
    {
        std::stringstream defStream(line.substr(2));
        std::string firstWord;
        defStream >> firstWord;
        std::string secondWord;
        defStream >> secondWord;
        if (firstWord == "reference" && secondWord == "line")
        {
            ++numLongDataElements;
        }
        else if (firstWord == "long" && secondWord == "section")
        {
            ++numHeightDataElements;
        }
    }
}

void Surface::parseSurfaceDataLine(const std::string &line)
{
    std::stringstream lineStream(line);

    float *longDataElementArray = new float[numLongDataElements];
    surfaceLongDataLineVector.push_back(longDataElementArray);
    float *heightDataElementArray = new float[numHeightDataElements];
    surfaceHeightDataLineVector.push_back(heightDataElementArray);

    for (unsigned int i = 0; i < numLongDataElements; ++i)
    {
        double value;
        lineStream >> value;
        if (lineStream.fail())
        {
            value = std::numeric_limits<double>::quiet_NaN();
            lineStream.clear();
            lineStream.ignore(256, ' ');
        }
        longDataElementArray[i] = value;
    }
    for (unsigned int i = 0; i < numHeightDataElements; ++i)
    {
        double value;
        lineStream >> value;
        if (lineStream.fail())
        {
            value = std::numeric_limits<double>::quiet_NaN();
            lineStream.clear();
            lineStream.ignore(256, ' ');
        }
        heightDataElementArray[i] = value;
        if (value < minElev)
        {
            minElev = value;
        }
        if (value > maxElev)
        {
            maxElev = value;
        }
    }
}

void Surface::computeNormals()
{
    surfaceNormalDataLineVector.resize(surfaceHeightDataLineVector.size());
    for (unsigned int i = 0; i < surfaceHeightDataLineVector.size(); ++i)
    {
        surfaceNormalDataLineVector[i] = new SurfaceNormal[numHeightDataElements];
    }

    for (unsigned int u = 1; u < surfaceHeightDataLineVector.size() - 1; ++u)
    {
        for (unsigned int v = 1; v < numHeightDataElements - 1; ++v)
        {
            //if(surfaceHeightDataLineVector[u][v]!=surfaceHeightDataLineVector[u][v]) {
            //std::cout << u << ", " << v << ": height is nan!" << std::endl;
            //}
            typedef gealg::mv<3, 0x40201>::type NormalVector;
            NormalVector r_p0;
            r_p0[0] = reference_line_increment;
            r_p0[1] = 0;
            r_p0[2] = surfaceHeightDataLineVector[u + 1][v] - surfaceHeightDataLineVector[u][v];
            NormalVector r_0p;
            r_0p[0] = 0;
            r_0p[1] = long_section_v_increment;
            r_0p[2] = surfaceHeightDataLineVector[u][v + 1] - surfaceHeightDataLineVector[u][v];
            NormalVector r_m0;
            r_m0[0] = -reference_line_increment;
            r_m0[1] = 0;
            r_m0[2] = surfaceHeightDataLineVector[u - 1][v] - surfaceHeightDataLineVector[u][v];
            NormalVector r_0m;
            r_0m[0] = 0;
            r_0m[1] = -long_section_v_increment;
            r_0m[2] = surfaceHeightDataLineVector[u][v - 1] - surfaceHeightDataLineVector[u][v];

            gealg::mv<4, 0x06050300>::type A = r_p0 * r_0p - r_0p * r_p0
                                               + r_0p * r_m0 - r_m0 * r_0p
                                               + r_m0 * r_0m - r_0m * r_m0
                                               + r_0m * r_p0 - r_p0 * r_0m;

            SurfaceNormal *normal = &surfaceNormalDataLineVector[u][v];
            if (A[1] == A[1] && A[2] == A[2] && A[3] == A[3])
            {
                normal->u = A[3];
                normal->v = -A[2];
                normal->w = A[1];
                normal->normalize();
            }
            else
            {
                normal->u = 0.0;
                normal->v = 0.0;
                normal->w = 1.0;
            }
        }
    }
}

osg::Image *Surface::createDiffuseMapTextureImage()
{
    if (surfaceNormalDataLineVector.size() < surfaceHeightDataLineVector.size())
    {
        computeNormals();
    }

    osg::Image *diffuseMap = new osg::Image();

    diffuseMap->allocateImage(surfaceHeightDataLineVector.size(), numHeightDataElements, 1, GL_RGB, GL_UNSIGNED_BYTE);

    for (unsigned int u = 0; u < surfaceHeightDataLineVector.size(); ++u)
    {
        for (unsigned int v = 0; v < numHeightDataElements; ++v)
        {
            *(diffuseMap->data(u, v) + 0) = 86;
            *(diffuseMap->data(u, v) + 1) = 86;
            *(diffuseMap->data(u, v) + 2) = 86;
        }
    }

    return diffuseMap;
}

osg::Image *Surface::createParallaxMapTextureImage()
{
    if (surfaceNormalDataLineVector.size() < surfaceHeightDataLineVector.size())
    {
        computeNormals();
    }

    osg::Image *parallaxMap = new osg::Image();

    parallaxMap->allocateImage(surfaceHeightDataLineVector.size(), numHeightDataElements, 1, GL_RGBA, GL_UNSIGNED_BYTE);

    double minMaxScale = 255.0 / (maxElev - minElev);

    for (unsigned int u = 0; u < surfaceHeightDataLineVector.size(); ++u)
    {
        for (unsigned int v = 0; v < numHeightDataElements; ++v)
        {
            *(parallaxMap->data(u, v) + 0) = (unsigned char)((surfaceNormalDataLineVector[u][v].u + 1.0) * 0.5 * 255.0);
            *(parallaxMap->data(u, v) + 1) = (unsigned char)((surfaceNormalDataLineVector[u][v].v + 1.0) * 0.5 * 255.0);
            *(parallaxMap->data(u, v) + 2) = (unsigned char)((surfaceNormalDataLineVector[u][v].w + 1.0) * 0.5 * 255.0);

            if (surfaceHeightDataLineVector[u][v] == surfaceHeightDataLineVector[u][v])
            {
                *(parallaxMap->data(u, v) + 3) = (unsigned char)((surfaceHeightDataLineVector[u][v] - minElev) * minMaxScale);
            }
            else
            {
                *(parallaxMap->data(u, v) + 3) = 0;
            }
        }
    }

    return parallaxMap;
}

void Surface::computeConeStepMappingRatio()
{
    /*
      basically, 99% of all pixels will fall in under 2.0
      (most of the time, on the heightmaps I've tested)
      the question:
      Is reduced resolution worth missing
      the speedup of the slow ones?
      */
    const float max_ratio = 1.0;
    /*
      do I want to sqrt my cone value (better
      spread, a bit more work in shader)?
      */
    const bool sqrt_cone_ratio = true;
    /*
      do I want to invert the heightmap
      (make it a depthmap)?
      It removes 1 op in the shader
      */
    //const bool invert_heightmap = true;
    /*
      I need to have safety because of the linear
      interpolation of the Cone Ratios, inside the
      shader.  So safety values:
      1.0 = fully safe (slower)
      0.0 = possibly unsafe (faster)

      changed my mind...always safe
      */
    /*
      Do I want the textures to be computed
      as tileable?  This makes the Cone Step
      Mapping safer if the texture actually
      is tiled.  Slower processing, though.
      I auto-detect these after image load.

      I'm changing my mind...ALWAYS TILE!
      */
    /*
      Do I want to use the linear or squared
      average height for mip-mapping?  The
      squared average weights the higher
      pixels more.
      */
    //const bool use_squared_height = true;
    /*
      Do I want the cones to be round or
      square?
      0.0 => square
      1.0 => round
      */
    const float rounded_cones = 0.0f;

    surfaceConeRatioDataLineVector.resize(surfaceHeightDataLineVector.size());
    for (unsigned int i = 0; i < surfaceConeRatioDataLineVector.size(); ++i)
    {
        surfaceConeRatioDataLineVector[i] = new float[numHeightDataElements];
    }

    unsigned int height = surfaceConeRatioDataLineVector.size();
    unsigned int width = numHeightDataElements;
    float iheight = 1.0 / height;
    float iwidth = 1.0 / width;

    for (int y = 0; y < (int)height; ++y)
    {
        printf("%c%c%c%c%c%c", 8, 8, 8, 8, 8, 8);
        printf("%c%c%c%c%c%c%c%c%c", 8, 8, 8, 8, 8, 8, 8, 8, 8);

        //	print the % done
        printf("%5.2f%%", (y + 1) * 100.0 / height);
        //	print the time elapsed
        //printf("  %6.0fs", 0.001*(clock() - tin));

        for (int x = 0; x < (int)width; ++x)
        {
            float min_ratio2, actual_ratio;
            int x1, x2, y1, y2;
            //unsigned char ht;
            float ht;
            float r2, h2;

            //  set up some initial values
            // (note I'm using ratio squared throughout,
            // and taking sqrt at the end...faster)
            //ht = Data[y*ScanWidth + chans*x + OFF_HEIGHT];
            //ht = intData [y*width + x];
            ht = 1.0 - (surfaceHeightDataLineVector[y][x] - minElev) / (maxElev - minElev);
            if (ht != ht)
            {
                ht = 0.0;
            }

            // ok, start with the largest area i'm willing to search
            min_ratio2 = max_ratio * max_ratio;

            // scan in outwardly expanding blocks
            // (so I can stop if I reach my minimum ratio)
            for (int rad = 1;
                 //rad*rad <= (255-ht)*(255-ht)*min_ratio2*width*height/255/255;
                 rad * rad <= (1.0 - ht) * (1.0 - ht) * min_ratio2 * width * height;
                 ++rad)
            {
                // do the box for this "radius"
                // (so for each of these lines...)

                // West and east
                x1 = x - rad;
                x2 = x + rad;
                {
                    float delx = -rad * iwidth;
                    // y limits
                    // (+- 1 because I'll cover the corners in the X-run)
                    y1 = y - rad + 1;
                    y2 = y + rad - 1;

                    // and check the line
                    for (int dy = y1; dy <= y2; ++dy)
                    {
                        //	west
                        //int idx = xy2index (x1, dy, width, height, 1);
                        {
                            h2 = (1.0 - ((surfaceHeightDataLineVector[warpIndex(y, height)][warpIndex(x1, width)] - minElev) / (maxElev - minElev)) - ht);
                            //h2 = (intData[idx] - ht) / 255.0;
                            if (h2 > 0.0)
                            {
                                float dely = (dy - y) * iheight * rounded_cones;
                                r2 = delx * delx + dely * dely;
                                h2 *= h2;
                                if (h2 * min_ratio2 > r2)
                                {
                                    //  this is the new (lowest) value
                                    min_ratio2 = r2 / h2;
                                }
                            }
                        }

                        //	east
                        //idx = xy2index (x2, dy, width, height, 1);
                        {
                            h2 = (1.0 - ((surfaceHeightDataLineVector[warpIndex(dy, height)][warpIndex(x2, width)] - minElev) / (maxElev - minElev)) - ht);
                            //h2 = (intData[idx] - ht) / 255.0;
                            if (h2 > 0.0)
                            {
                                float dely = (dy - y) * iheight * rounded_cones;
                                r2 = delx * delx + dely * dely;
                                h2 *= h2;
                                if (h2 * min_ratio2 > r2)
                                {
                                    //  this is the new (lowest) value
                                    min_ratio2 = r2 / h2;
                                }
                            }
                        }
                    }
                }

                // North
                y1 = y - rad;
                {
                    float dely = -rad * iheight;
                    // x limits
                    x1 = x - rad;
                    x2 = x + rad;

                    // and check the line
                    for (int dx = x1; dx <= x2; ++dx)
                    {
                        //int idx = xy2index (dx, y1, width, height, 1);
                        {
                            h2 = (1.0 - ((surfaceHeightDataLineVector[warpIndex(y1, height)][warpIndex(dx, width)] - minElev) / (maxElev - minElev)) - ht);
                            //h2 = (intData[idx] - ht) / 255.0;
                            if (h2 > 0.0)
                            {
                                float delx = (dx - x) * iwidth * rounded_cones;
                                r2 = delx * delx + dely * dely;
                                h2 *= h2;
                                if (h2 * min_ratio2 > r2)
                                {
                                    //  this is the new (lowest) value
                                    min_ratio2 = r2 / h2;
                                }
                            }
                        }
                    }
                }

                // South
                y2 = y + rad;
                {
                    float dely = rad * iheight;
                    // x limits
                    x1 = x - rad;
                    x2 = x + rad;

                    // and check the line
                    for (int dx = x1; dx <= x2; ++dx)
                    {
                        //int idx = xy2index (dx, y2, width, height, 1);
                        {
                            h2 = (1.0 - ((surfaceHeightDataLineVector[warpIndex(y2, height)][warpIndex(dx, width)] - minElev) / (maxElev - minElev)) - ht);
                            //h2 = (intData[idx] - ht) / 255.0;
                            if (h2 > 0.0)
                            {
                                float delx = (dx - x) * iwidth * rounded_cones;
                                r2 = delx * delx + dely * dely;
                                h2 *= h2;
                                if (h2 * min_ratio2 > r2)
                                {
                                    //  this is the new (lowest) value
                                    min_ratio2 = r2 / h2;
                                }
                            }
                        }
                    }
                }

                // done with the expanding loop
            }

            //  actually I have the ratio squared.  Sqrt
            actual_ratio = sqrt(min_ratio2);
            // scale to 1.0
            actual_ratio /= max_ratio;
            // most of the data is on the low end...sqrting again spreads it better
            // (plus multiply is a cheap operation in shaders!)
            if (sqrt_cone_ratio)
            {
                actual_ratio = sqrt(actual_ratio);
            }
            //  Red stays height
            //  Green becomes Step-Cone-Ratio
            //Data[y*ScanWidth + chans*x + OFF_CONERATIO] = static_cast<unsigned char>(255.0 * actual_ratio);
            if (actual_ratio < (1.0 / 255.0))
            {
                actual_ratio = 1.0 / 255.0;
            }
            surfaceConeRatioDataLineVector[y][x] = actual_ratio;
            // but make sure it is > 0.0, since I divide by it in the shader
            //if (Data[y*ScanWidth + chans*x + OFF_CONERATIO] < 1)
            //   Data[y*ScanWidth + chans*x + OFF_CONERATIO] = 1;
            // Blue stays df/dx
            // Alpha stays df/dy

        } // done with the column

    } // done with the row
}

osg::Image *Surface::createConeStepMapTextureImage()
{
    if (surfaceNormalDataLineVector.size() < surfaceHeightDataLineVector.size())
    {
        computeNormals();
    }
    if (surfaceConeRatioDataLineVector.size() < surfaceHeightDataLineVector.size())
    {
        computeConeStepMappingRatio();
    }

    osg::Image *parallaxMap = new osg::Image();

    parallaxMap->allocateImage(surfaceHeightDataLineVector.size(), numHeightDataElements, 1, GL_RGBA, GL_UNSIGNED_BYTE);

    double minMaxScale = 255.0 / (maxElev - minElev);

    for (unsigned int u = 0; u < surfaceHeightDataLineVector.size(); ++u)
    {
        for (unsigned int v = 0; v < numHeightDataElements; ++v)
        {
            if (surfaceHeightDataLineVector[u][v] == surfaceHeightDataLineVector[u][v])
            {
                *(parallaxMap->data(u, v) + 0) = 255 - (unsigned char)((surfaceHeightDataLineVector[u][v] - minElev) * minMaxScale);
            }
            else
            {
                *(parallaxMap->data(u, v) + 0) = 0;
            }

            *(parallaxMap->data(u, v) + 1) = (unsigned char)((surfaceConeRatioDataLineVector[u][v]) * 255.0);

            *(parallaxMap->data(u, v) + 2) = (unsigned char)((surfaceNormalDataLineVector[u][v].u + 1.0) * 0.5 * 255.0);
            *(parallaxMap->data(u, v) + 3) = (unsigned char)((surfaceNormalDataLineVector[u][v].v + 1.0) * 0.5 * 255.0);
        }
    }

    return parallaxMap;
}

osg::Image *Surface::createPavementTextureImage()
{
    if (surfaceNormalDataLineVector.size() < surfaceHeightDataLineVector.size())
    {
        computeNormals();
    }

    osg::Image *pavement = new osg::Image();

    pavement->allocateImage(surfaceHeightDataLineVector.size(), numHeightDataElements, 1, GL_RGBA, GL_UNSIGNED_BYTE);

    double minMaxScale = 255.0 / (maxElev - minElev);

    for (unsigned int u = 0; u < surfaceHeightDataLineVector.size(); ++u)
    {
        for (unsigned int v = 0; v < numHeightDataElements; ++v)
        {
            if (surfaceHeightDataLineVector[u][v] == surfaceHeightDataLineVector[u][v])
            {
                *(pavement->data(u, v) + 0) = (unsigned char)((surfaceHeightDataLineVector[u][v] - minElev) * minMaxScale * 0.5) + 63;
                *(pavement->data(u, v) + 1) = (unsigned char)((surfaceHeightDataLineVector[u][v] - minElev) * minMaxScale * 0.5) + 63;
                *(pavement->data(u, v) + 2) = (unsigned char)((surfaceHeightDataLineVector[u][v] - minElev) * minMaxScale * 0.5) + 63;
                *(pavement->data(u, v) + 3) = 255;
            }
            else
            {
                *(pavement->data(u, v) + 0) = 0;
                *(pavement->data(u, v) + 1) = 0;
                *(pavement->data(u, v) + 2) = 0;
                *(pavement->data(u, v) + 3) = 255;
            }
        }
    }

    return pavement;
}

} //end namespace opencrg
