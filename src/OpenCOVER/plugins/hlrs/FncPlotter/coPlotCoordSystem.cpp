/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include <cmath>
#include <sstream>
#include <cstring>
#include <osg/Geometry>

#include "coPlotCoordSystem.h"

using std::cout;
using std::endl;
using std::cerr;

CoordSystem::CoordSystem(void)
    : m_strFontname("")
    , m_charSize(12)
    , m_fontColor(1.0f, 1.0f, 1.0f, 1.0f)
    , m_xColor(1.0f, 0.0f, 0.0f, 1.0f)
    , m_yColor(1.0f, 0.0f, 0.0f, 1.0f)
    , m_position(0.0f, -1000.0f, 0.0f)
    , m_xSize(200.0f)
    , m_zSize(200.0f)
    , m_xMin(-1.0f)
    , m_xMax(1.0f)
    , m_yMin(-1.0f)
    , m_yMax(1.0f)
    , m_transformNode(0)
    , m_precision(2)
    , m_rootNode(0)
    , m_subGeode(0)
    , m_scalingX(1.0f)
    , m_scalingZ(1.0f)
{
}

CoordSystem::~CoordSystem(void)
{
    for (size_t l = 0; l < m_graphs.size(); l++)
        delete m_graphs[l];
    m_graphs.clear();
}

void CoordSystem::setFont(const std::string &fontname)
{
    m_strFontname = fontname;
}

void CoordSystem::setCharacterSize(unsigned int size)
{
    m_charSize = size;
}

void CoordSystem::setFontColor(const osg::Vec4 &color)
{
    m_fontColor = color;
}

void CoordSystem::setAxesSize(float x, float y)
{
    // Calculate local coordinates and not mm
    m_xSize = x / 100;
    m_zSize = y / 100;
}

void CoordSystem::setPosition(const osg::Vec3 &c)
{
    if (m_transformNode.valid())
        m_transformNode->setPosition(c);
    // save position for later (maybe node creation)
    m_position = c;
}

void CoordSystem::setAxesColors(const osg::Vec4 &x, const osg::Vec4 &y)
{
    m_xColor = x;
    m_yColor = y;
}

void CoordSystem::setMinMaxValues(float xmin, float xmax, float ymin, float ymax)
{
    m_xMin = xmin;
    m_xMax = xmax;
    m_yMin = ymin;
    m_yMax = ymax;
}

osg::ref_ptr<osgText::Text> CoordSystem::createText(const std::string &s, const osg::Vec3 &pos)
{
    osg::ref_ptr<osgText::Text> t = new osgText::Text();
    t->setCharacterSize(m_charSize);
    t->setFont(m_strFontname);
    t->setText(s, osgText::String::ENCODING_UTF8);
    t->setAxisAlignment(osgText::Text::REVERSED_XZ_PLANE);
    t->setPosition(pos);
    t->setColor(m_fontColor);

    return t.get();
}

osg::ref_ptr<osgText::Text> CoordSystem::createText(float value, const osg::Vec3 &pos)
{
    osg::ref_ptr<osgText::Text> t = new osgText::Text();
    t->setCharacterSize(m_charSize);
    t->setFont(m_strFontname);
    std::string formatString = "%.";
    formatString += convertNumberToString(m_precision);
    formatString += "f";
    //FIXME OUCH!!!!
    char c[10];
    sprintf(c, formatString.c_str(), value);
    t->setText(c, osgText::String::ENCODING_UTF8);
    t->setAxisAlignment(osgText::Text::REVERSED_XZ_PLANE);
    t->setPosition(pos);
    t->setColor(m_fontColor);

    return t.get();
}

std::string CoordSystem::convertNumberToString(float n) const
{
    std::ostringstream s;
    s << n;
    return s.str();
}

float CoordSystem::calcHeightForX(float /*tickStepZ*/) const
{
    if (m_yMax <= 0 && m_yMin < 0)
    {
        //cout << "HERE" << endl;
        return m_zSize;
    }
    else if (m_yMax > 0 && m_yMin >= 0)
    {
        //cout << "HERE2" << endl;
        return -m_zSize;
    }
    else
    {
        float max = fabs(m_yMax);
        float min = fabs(m_yMin);
        float s = min + max;

        if (max == min)
        {
            return 0.0f;
        }
        else if (max > min)
        {
            float ratio = (min / s);
            //cout << "YRatio1: "<< (ratio) << endl;
            return -m_zSize * ratio;
        }
        else
        {
            float ratio = (min / s);
            //cout << "YRatio2: "<< ratio << endl;
            return m_zSize * (1.0f - ratio);
        }
    }

    // Should never come here
    return 0.0f;
}

osg::ref_ptr<osg::Group> CoordSystem::createWithLinearScaling(void)
{
    // check, if max>min
    if (m_xMin > m_xMax)
    {
        float temp = m_xMin;
        m_xMin = m_xMax;
        m_xMax = temp;
    }

    if (m_yMin > m_yMax)
    {
        float temp = m_yMin;
        m_yMin = m_yMax;
        m_yMax = temp;
    }
    /*
   Depth Values
   0.01 = 1mm
   0.1 = 1cm
   10 = 1m
   100 = 10m
   1000 = 100m
   */

    // This makes enough space so we can see the numbers without any problems
    //FIXME Ouch again....
    char n1[10], n2[10];
    sprintf(n1, "%.1f", m_xMax);
    sprintf(n2, "%.1f", m_xMin);
    unsigned int n = 1;
    if (strlen(n1) > strlen(n2))
        n = strlen(n1);
    else
        n = strlen(n2);

    // we only need to be careful with string size in x-direction
    // y is not so vulnerable for this, so you can set a default
    const float tickStepX = (m_charSize * n);
    const float tickStepZ = (m_charSize * 3);

    //FIXME wozu tickStepZ
    const float zPosForXAxis = calcHeightForX(tickStepZ);

    // Count Ticks, so we can calculate step size between two ticks
    // Need this for placing the axis
    float tempPos = -m_xSize;
    unsigned int nTicksX = 0;
    while (tempPos <= m_xSize)
    {
        nTicksX++;
        tempPos += tickStepX;
    }
    // HACK for some reasons the step size never matches the needed size. With 1 subtracted everything is fine.
    // I think it doesnt count the first tick
    nTicksX--;

    float arrayDimensions = 0.0f;
    if (m_xMax > 0 && m_xMin > 0)
        arrayDimensions = m_xMax - m_xMin;
    else if (m_xMin < 0 && m_xMax < 0)
        arrayDimensions = fabs(m_xMin) - fabs(m_xMax);
    else
        arrayDimensions = m_xMax + fabs(m_xMin);

    const float stepX = (float)((arrayDimensions) / (nTicksX));
    //cout << "ticksX: " << nTicksX << endl;
    //cout << "stepX: " << stepX << endl;

    tempPos = -m_zSize;
    unsigned int nTicksY = 0;
    while (tempPos <= m_zSize)
    {
        nTicksY++;
        tempPos += tickStepZ;
    }
    nTicksY--;
    if (m_yMax > 0 && m_yMin > 0)
        arrayDimensions = m_yMax - m_yMin;
    else if (m_yMin < 0 && m_yMax < 0)
        arrayDimensions = fabs(m_yMin) - fabs(m_yMax);
    else
        arrayDimensions = m_yMax + fabs(m_yMin);
    // HACK Same as above + we dont need the tick for '0' further down, so we remove this, too
    const float stepY = (float)(arrayDimensions / (nTicksY));

    // procedure: Create and position x-Axis. based on this, position the y-axis

    // Create x-axis
    osg::ref_ptr<osg::Vec3Array> axesVertices = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec4Array> axesColors = new osg::Vec4Array();
    osg::ref_ptr<osg::Geometry> axesGeometry = new osg::Geometry();
    m_subGeode = new osg::Geode();

    // Calculate y-value of x-axis. This depends on function

    float xPosForYAxis = 0; // calculated below

    // Position x-axis
    // right vertex
    axesColors->push_back(m_xColor);
    axesVertices->push_back(osg::Vec3(-m_xSize, 0.0f, zPosForXAxis));
    // left vertex
    axesColors->push_back(m_xColor);
    axesVertices->push_back(osg::Vec3(m_xSize, 0.0f, zPosForXAxis));
    // Create Ticks on X-axis, begin right and end left
    {
        float x = -m_xSize;
        float xValue = m_xMax;
        const float tickHeight = m_xSize * 0.025f;

        // if we need the y-axis left or right, position them here before
        bool yHasPosition = false;
        if (m_xMin >= 0 && m_xMax > 0)
        {
            xPosForYAxis = x + tickStepX * nTicksX;
            yHasPosition = true;
            // top vertex
            axesColors->push_back(m_yColor);
            axesVertices->push_back(osg::Vec3(xPosForYAxis, 0.0f, m_zSize));
            // bottom vertex of tick
            axesColors->push_back(m_yColor);
            axesVertices->push_back(osg::Vec3(xPosForYAxis, 0.0f, -m_zSize));
        }
        else if (m_xMax <= 0 && m_xMin < 0)
        {
            xPosForYAxis = x;
            yHasPosition = true;
            // top vertex
            axesColors->push_back(m_yColor);
            axesVertices->push_back(osg::Vec3(xPosForYAxis, 0.0f, m_zSize));
            // bottom vertex of tick
            axesColors->push_back(m_yColor);
            axesVertices->push_back(osg::Vec3(xPosForYAxis, 0.0f, -m_zSize));
        }

        while (x <= m_xSize)
        {
            // top vertex of tick
            axesColors->push_back(m_xColor);
            axesVertices->push_back(osg::Vec3(x, 0.0f, zPosForXAxis + tickHeight));
            // bottom vertex of tick
            axesColors->push_back(m_xColor);
            axesVertices->push_back(osg::Vec3(x, 0.0f, zPosForXAxis - tickHeight));

            // add number
            if (0.0f != xValue)
            {
                std::string n = convertNumberToString(xValue);
                float length = 0;
                for (size_t l = 0; l < n.length(); l++)
                {
                    if (n[l] == '.')
                    {
                        // one position after comma
                        length++;
                        break;
                    }
                    else
                        length++;
                }
                length = length * m_charSize * 0.5f;
                m_subGeode->addDrawable(createText(xValue, osg::Vec3(x + length, 0.0f, zPosForXAxis - 2.0f * tickHeight)).get());
            }
            //place zero separate
            else
            {
                int oldPrec = m_precision;
                m_precision = 0;
                m_subGeode->addDrawable(createText(0, osg::Vec3(x + m_charSize, 0.0f, zPosForXAxis - 2.0f * tickHeight)).get());
                m_precision = oldPrec;
            }

            // check if we have the intersection of both axis here
            if (!yHasPosition)
            {
                if (xValue > 0 && xValue - stepX <= 0)
                {
                    // found it
                    float min = fabs(xValue - stepX);
                    float max = fabs(xValue);
                    float s = min + max;
                    float ratio = max / s;
                    //if(min>max)
                    // ratio = 1.0f - ratio;
                    //cout << "ratio1: " << ratio << endl;
                    xPosForYAxis = x + tickStepX * ratio;

                    yHasPosition = true;

                    // Now position y-axis
                    // top vertex
                    axesColors->push_back(m_yColor);
                    axesVertices->push_back(osg::Vec3(xPosForYAxis, 0.0f, m_zSize));
                    // bottom vertex of tick
                    axesColors->push_back(m_yColor);
                    axesVertices->push_back(osg::Vec3(xPosForYAxis, 0.0f, -m_zSize));
                }
            }
            xValue -= stepX;
            x += tickStepX;
        }
    }

    {
        // position ticks in +y direction
        float z = zPosForXAxis + tickStepZ;

        float yValue = 0;
        if (m_yMin > 0 && m_yMax > 0)
            yValue = m_yMin + stepY;
        else if (m_yMin < 0 && m_yMax < 0)
            yValue = m_yMax - stepY;
        else
            yValue = stepY;

        const float tickWidth = m_zSize * 0.025f;
        while (z <= m_zSize)
        {
            // left vertex of tick
            axesColors->push_back(m_yColor);
            axesVertices->push_back(osg::Vec3(xPosForYAxis + tickWidth, 0.0f, z));
            // right vertex of tick
            axesColors->push_back(m_yColor);
            axesVertices->push_back(osg::Vec3(xPosForYAxis - tickWidth, 0.0f, z));

            // place numbers
            float length = 0.0f;
            std::string n = convertNumberToString(yValue);
            bool hasPoint = false;
            for (size_t l = 0; l < n.length(); l++)
            {
                if (n[l] == '.')
                    hasPoint = true;
                length++;
            }

            if (!hasPoint)
                length += m_precision;

            length--;
            length *= m_charSize * 0.5f;

            m_subGeode->addDrawable(createText(yValue, osg::Vec3(xPosForYAxis + length, 0.0f, z - m_charSize / 2)).get());

            yValue += stepY;
            z += tickStepZ;
        }
    }

    {
        // position ticks in -y direction
        float z = zPosForXAxis - tickStepZ;
        float yValue = 0.0f;

        if (m_yMin > 0 && m_yMax > 0)
            yValue = m_yMin + stepY;
        else if (m_yMin < 0 && m_yMax < 0)
            yValue = m_yMax - stepY;
        else
            yValue = -stepY;

        const float tickWidth = m_zSize * 0.025f;
        while (z >= -m_zSize)
        {
            // left vertex of tick
            axesColors->push_back(m_yColor);
            axesVertices->push_back(osg::Vec3(xPosForYAxis + tickWidth, 0.0f, z));
            // right vertex of tick
            axesColors->push_back(m_yColor);
            axesVertices->push_back(osg::Vec3(xPosForYAxis - tickWidth, 0.0f, z));

            // place numbers
            float length = 0.0f;
            std::string n = convertNumberToString(yValue);
            bool hasPoint = false;

            for (size_t l = 0; l < n.length(); l++)
            {
                if (n[l] == '.')
                    hasPoint = true;
                length++;
            }

            if (!hasPoint)
                length += m_precision;

            length--;
            length *= m_charSize * 0.5f;

            m_subGeode->addDrawable(createText(yValue, osg::Vec3(xPosForYAxis + length, 0.0f, z - m_charSize / 2)).get());

            yValue -= stepY;
            z -= tickStepZ;
        }
    }

    // Set Vertex Arrays for Rendering
    axesGeometry->setColorArray(axesColors.get());
    axesGeometry->setVertexArray(axesVertices.get());
    axesGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    osg::ref_ptr<osg::DrawElementsUInt> axesIndices = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0);

    for (size_t i = 0; i < axesVertices->size(); i++)
        axesIndices->push_back(i);

    axesGeometry->addPrimitiveSet(axesIndices.get());

    /*
        The Tree should look like this now:
                                rootNode
                                   |
                      m_transformNode
                        |          |
           -----m_subGeode       Graph's transform nodes
          /           |                          |
        Axes Text   axesGeometry              Graph's Geode
   */

    m_subGeode->addDrawable(axesGeometry.get());

    m_transformNode = new osg::PositionAttitudeTransform();
    m_transformNode->setPosition(m_position);
    m_transformNode->addChild(m_subGeode.get());

    m_rootNode = new osg::Group();
    m_rootNode->addChild(m_transformNode.get());

    // calculations/preparations for graphs
    m_cross.set(xPosForYAxis, 0.0f, zPosForXAxis);

    m_scalingX = tickStepX / stepX;
    m_scalingZ = tickStepZ / stepY;

    return m_rootNode.get();
}

osg::ref_ptr<osg::Group> CoordSystem::createWithBinaryLogScaling(void)
{
    // check, if max>min
    if (m_xMin > m_xMax)
    {
        float temp = m_xMin;
        m_xMin = m_xMax;
        m_xMax = temp;
    }
    if (m_yMin > m_yMax)
    {
        float temp = m_yMin;
        m_yMin = m_yMax;
        m_yMax = temp;
    }
    // as logarithms can only be calculated for numbers >= 0, make sure min/max is > 0
    if (m_yMin < 0 || m_yMax < 0 || m_xMin < 0 || m_xMax < 0)
    {
        cerr << __FUNCTION__ << "please specify boundaries >= 0 for x and y!" << endl;
        return NULL;
    }
    /*
   Depth Values
   0.01 = 1mm
   0.1 = 1cm
   10 = 1m
   100 = 10m
   1000 = 100m
   */

    // here we dont need to be so careful, because we always have 2^x
    // y is not so vulnerable for this, so you can set a default
    const float minTickStepX = (m_charSize * 3);
    const float minTickStepZ = (m_charSize * 3);

    // step* has a different meaning here: its the difference in exponents between two ticks
    // I interpret m_xMax as the usual maximum number(same as above).
    // log() gives as the natural logarithm, but I need the binary. Formula is: log2(x) = log10(x)/log10(2)
    // instead of log10() I can choose any logarithm I need...
    const float log2 = log(2.0f);
    const int maxLogX = (int)(log(m_xMax) / log2);
    const int minLogX = (int)(log(m_xMin) / log2);
    int stepX = maxLogX - minLogX;
    //cout << "maxLog: " << maxLogX << endl;
    //cout << "minLog: " << minLogX << endl;
    //cout << "diff: " << stepX << endl;

    float tickStepX = (2.0f * m_xSize) / stepX;
    while (tickStepX < minTickStepX)
    {
        stepX /= 2;
        tickStepX = (2 * m_xSize) / stepX;
    }
    //cout << "diff: " << stepX << endl;

    const int nTicksX = (int)floor((float)(2 * m_xSize / tickStepX));
    //cout << "ticksX: " << nTicksX << endl;

    // Same for y
    const int maxLogY = (int)(log(m_yMax) / log2);
    const int minLogY = (int)(log(m_yMin) / log2);
    int stepY = maxLogY - minLogY;

    float tickStepZ = (2 * m_zSize) / stepY;
    while (tickStepZ < minTickStepZ)
    {
        stepY /= 2;
        tickStepZ = (2 * m_zSize) / stepY;
    }

    //const int nTicksY = (int) floor((float) (2 * m_zSize / tickStepZ));
    //cout << "ticksY: " << nTicksY << endl;

    // I can always position the y-axis at the left and, because m_xMin >= 0 && m_xMax > 0
    // Create x-axis
    osg::ref_ptr<osg::Vec3Array> axesVertices = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec4Array> axesColors = new osg::Vec4Array();
    osg::ref_ptr<osg::Geometry> axesGeometry = new osg::Geometry();
    m_subGeode = new osg::Geode();

    // Calculate y-value of x-axis. This depends on function
    const float zPosForXAxis = calcHeightForX(tickStepZ);
    float xPosForYAxis = 0.0f; // calculated below

    // Position x-axis
    // right vertex
    axesColors->push_back(m_xColor);
    axesVertices->push_back(osg::Vec3(-m_xSize, 0.0f, zPosForXAxis));
    // left vertex
    axesColors->push_back(m_xColor);
    axesVertices->push_back(osg::Vec3(m_xSize, 0.0f, zPosForXAxis));
    // Create Ticks on X-axis, begin right and end left
    {
        float x = -m_xSize;
        float xValue = maxLogX;
        const float tickHeight = m_xSize * 0.025f;

        // log() is only defined for values > 0, so y-axis is always left
        // if we need the y-axis left or right, position them here before
        xPosForYAxis = x + tickStepX * nTicksX;
        // top vertex
        axesColors->push_back(m_yColor);
        axesVertices->push_back(osg::Vec3(xPosForYAxis, 0.0f, m_zSize));
        // bottom vertex of tick
        axesColors->push_back(m_yColor);
        axesVertices->push_back(osg::Vec3(xPosForYAxis, 0.0f, -m_zSize));

        while (x <= m_xSize)
        {
            // top vertex of tick
            axesColors->push_back(m_xColor);
            axesVertices->push_back(osg::Vec3(x, 0.0f, zPosForXAxis + tickHeight));
            // bottom vertex of tick
            axesColors->push_back(m_xColor);
            axesVertices->push_back(osg::Vec3(x, 0.0f, zPosForXAxis - tickHeight));

            // add number
            if (0.0f != xValue)
            {
                std::string n = convertNumberToString(xValue);
                float length = 0.0f;
                for (size_t l = 0; l < n.length(); l++)
                {
                    if (n[l] == '.')
                    {
                        // one position after comma
                        length++;
                        break;
                    }
                    else
                        length++;
                }
                length = length * m_charSize * 0.5f;
                std::string s = "2^";
                s += convertNumberToString(xValue);
                m_subGeode->addDrawable(createText(s, osg::Vec3(x + length, 0.0f, zPosForXAxis - 2 * tickHeight)).get());
            }
            // place zero separate
            else
            {
                int oldPrec = m_precision;
                m_precision = 0;
                m_subGeode->addDrawable(createText(0.0f, osg::Vec3(x + m_charSize, 0.0f, zPosForXAxis - 2 * tickHeight)).get());
                m_precision = oldPrec;
            }
            xValue -= stepX;
            x += tickStepX;
        }
    }

    {
        // position ticks in +y direction
        float z = zPosForXAxis + tickStepZ;

        float yValue = 0.0f;
        if (m_yMin > 0 && m_yMax > 0)
            yValue = minLogY + 1.0f;
        else if (m_yMin < 0 && m_yMax < 0)
            yValue = minLogY - 1.0f;
        else
            yValue = minLogY;

        const float tickWidth = m_zSize * 0.025f;
        while (z <= m_zSize)
        {
            // left vertex of tick
            axesColors->push_back(m_yColor);
            axesVertices->push_back(osg::Vec3(xPosForYAxis + tickWidth, 0.0f, z));
            // right vertex of tick
            axesColors->push_back(m_yColor);
            axesVertices->push_back(osg::Vec3(xPosForYAxis - tickWidth, 0.0f, z));

            // place numbers
            float length = 0.0f;
            std::string n = convertNumberToString(yValue);
            bool hasPoint = false;
            for (size_t l = 0; l < n.length(); l++)
            {
                if (n[l] == '.')
                    hasPoint = true;
                length++;
            }
            if (!hasPoint)
                length += m_precision;

            length--;
            length *= m_charSize * 0.5f;

            std::string s = "2^";
            s += convertNumberToString(yValue);
            m_subGeode->addDrawable(createText(s, osg::Vec3(xPosForYAxis + length, 0.0f, z - m_charSize / 2.0f)).get());

            yValue += stepY;
            z += tickStepZ;
        }
    }

    {
        // position ticks in -y direction
        float z = zPosForXAxis - tickStepZ;

        float yValue = 0.0f;
        if (m_yMin > 0 && m_yMax > 0)
            yValue = minLogY + 1.0f;
        else if (m_yMin < 0 && m_yMax < 0)
            yValue = minLogY - 1.0f;
        else
            yValue = minLogY;

        const float tickWidth = m_zSize * 0.025f;
        while (z >= -m_zSize)
        {
            // left vertex of tick
            axesColors->push_back(m_yColor);
            axesVertices->push_back(osg::Vec3(xPosForYAxis + tickWidth, 0.0f, z));
            // right vertex of tick
            axesColors->push_back(m_yColor);
            axesVertices->push_back(osg::Vec3(xPosForYAxis - tickWidth, 0.0f, z));

            // place numbers
            float length = 0.0f;
            std::string n = convertNumberToString(yValue);
            bool hasPoint = false;
            for (size_t l = 0; l < n.length(); l++)
            {
                if (n[l] == '.')
                    hasPoint = true;
                length++;
            }
            if (!hasPoint)
                length += m_precision;

            length--;
            length *= m_charSize * 0.5f;

            std::string s = "2^";
            s += convertNumberToString(yValue);
            m_subGeode->addDrawable(createText(s, osg::Vec3(xPosForYAxis + length, 0.0f, z - m_charSize / 2)).get());

            yValue -= stepY;
            z -= tickStepZ;
        }
    }

    // Set Vertex Arrays for Rendering
    axesGeometry->setColorArray(axesColors.get());
    axesGeometry->setVertexArray(axesVertices.get());
    axesGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    osg::ref_ptr<osg::DrawElementsUInt> axesIndices = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0);
    for (size_t i = 0; i < axesVertices->size(); i++)
        axesIndices->push_back(i);
    axesGeometry->addPrimitiveSet(axesIndices.get());

    /*
        The Tree should look like this now:
                                  rootNode
                                    |
                                m_transformNode
                                |            |
                  /-----m_subGeode         Graph's transform nodes
                 /           |                  |
           Axes Text    axesGeometry        Graph's Geode
   */

    m_subGeode->addDrawable(axesGeometry.get());

    m_transformNode = new osg::PositionAttitudeTransform();
    m_transformNode->setPosition(m_position);
    m_transformNode->addChild(m_subGeode.get());

    m_rootNode = new osg::Group();
    m_rootNode->addChild(m_transformNode.get());

    // calculations/preparations for graphs
    // m_cross.set(xPosForYAxis, zPosForXAxis, 0);

    // m_ScalingX = tickStepX/stepX;
    // m_ScalingZ = tickStepZ/stepY;

    return m_rootNode.get();
}

void CoordSystem::addGraph(CoordSystem::Graph *g)
{
    if (!m_subGeode.valid() || !m_transformNode.valid())
        return;

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(g->m_geometry.get());
    osg::ref_ptr<osg::PositionAttitudeTransform> transformNode = new osg::PositionAttitudeTransform();
    osg::Vec3 pos;
    pos.set(m_cross.x(), 0, m_cross.z());
    transformNode->setPosition(pos);
    transformNode->setScale(osg::Vec3(m_scalingX, 1.0f, m_scalingZ));
    transformNode->addChild(geode.get());

    m_graphs.push_back(g);
    m_transformNode->addChild(transformNode.get());
}

void CoordSystem::enableGraph(unsigned int i) const
{
    if (i < m_graphs.size())
        m_graphs[i]->enable();
}

void CoordSystem::disableGraph(unsigned int i) const
{
    if (i < m_graphs.size())
        m_graphs[i]->disable();
}
