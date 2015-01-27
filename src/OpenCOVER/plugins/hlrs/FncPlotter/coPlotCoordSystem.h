/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _COORDSYSTEM_H_
#define _COORDSYSTEM_H_
#include <string>
#include <vector>
#include <osg/Geode>
#include <osg/PositionAttitudeTransform>
#include <osgText/Font>
#include <osgText/Text>

class CoordSystem
{
public:
    class Graph
    {
    public:
        friend class CoordSystem;

        Graph(const float *values, size_t numOfPairs, const osg::Vec4 &color, GLenum renderMode = GL_LINE_STRIP, const std::string &name = "");

        void setValues(const float *values, size_t nPairs);
        void setRenderMode(GLenum newMode);
        void setColor(const osg::Vec4 &color);
        // Name has the same color as graph
        void setName(const std::string &name);
        void enable(void);
        void disable(void);

    protected:
        osg::ref_ptr<osg::Geometry> m_geometry;
        size_t m_nPairs;
        osg::ref_ptr<osg::Vec4Array> m_color;
        osg::ref_ptr<osg::DrawElementsUInt> m_indices;
        std::string m_name;
        GLenum m_renderMode;
        const float *m_values;
    };

public:
    CoordSystem(void);
    ~CoordSystem(void);

    // you have to set some of these things before calling create()

    //! \fn void setFont(String)
    //! \brief Set the font we will use everywhere. I assume a monospaced Font here, such as Courier.
    //! So this gives a character width of ~9 units
    //! default: uses (ugly) OSG Font
    void setFont(const std::string &Fontname);

    //! \fn void setCharacterSize(size_t)
    //! \brief set the character size for each char. This is important. Default value: 16
    //! default: 12
    void setCharacterSize(unsigned int size);

    //! \fn void setFontColor(osg:Vec4)
    //! \brief Set fonts color (RGBA format). Default: White with 1.0 alpha
    //! default: white
    void setFontColor(const osg::Vec4 &color);

    //! \fn void setPosition(osg::Vec3 p)
    //! \brief Set the coordinate system's Position.
    //! Same as osg::PositionAttitudeTransform::setPosition()
    //! a depth-value of -1000 seems to be good
    //! default: (0,-1000,0)
    void setPosition(const osg::Vec3 &p);

    //! \fn void setAxesColors(osg::Vec4 x, osg::Vec4 y)
    //! \brief set the Color for the axes: Default: Red (1,0,0,1)
    //! default yellow for both
    void setAxesColors(const osg::Vec4 &x, const osg::Vec4 &y);

    //! \fn void setAxesSize(const float x, const float y)
    //! \brief Set the length for the axes in mm. 1mm = 0.01 units. Default. 200units = 0.2m =
    //! default: 200
    void setAxesSize(float x, float y);

    //! \fn void setMinMaxValues(const float xmin, const float xmax, const float ymin, const float ymax)
    //! \brief Set the values, that should be displayed in the coordinate system
    //! default: x(-1,1) y(-1,1)
    void setMinMaxValues(float xmin, float xmax, float ymin, float ymax);

    //! \fn osg::Geode *createWithLinearScaling(void)
    //! \brief This will take all your settings and create a coordinate system with the graphs.
    //!  The axes will be scaled linear, means there is always the same space between
    //!  two marks. For a better overview in the creation algorithm I have decided to
    //!  divide it into three separate functions. This is not a nice OO decision, but
    //!  makes life a lot easier...
    //! \ret 0, if error while creation. Else a valid osg::Group
    osg::ref_ptr<osg::Group> createWithLinearScaling(void);
    // Andreas said, I only need binary and common log...
    osg::ref_ptr<osg::Group> createWithBinaryLogScaling(void);
    osg::ref_ptr<osg::Group> createWithCommonLogScaling(void);

    // you have to call create(), before you do this!!
    // problems:
    // 1) if you have y>=0, but your function values are < 0
    // they are drawn. Maybe I should make the constructor of Graph private
    // and let a method of CoordSystem create a graph. There I can skip the values.
    // Same goes for functions value >ymax
    void addGraph(CoordSystem::Graph *g);
    //FIXME Why const????
    void enableGraph(unsigned int i) const;
    void disableGraph(unsigned int i) const;

protected:
    float calcHeightForX(float tickStepY) const;
    std::string convertNumberToString(float n) const;
    //! \fn osgText::Text *createText(float value)
    //! \brief Take the number and create a text object from it
    osg::ref_ptr<osgText::Text> createText(float value, const osg::Vec3 &pos);
    osg::ref_ptr<osgText::Text> createText(const std::string &s, const osg::Vec3 &pos);

    std::string m_strFontname;
    unsigned int m_charSize;
    osg::Vec4 m_fontColor, m_xColor, m_yColor;
    osg::Vec3 m_position;
    float m_xSize, m_zSize;

    float m_xMin, m_xMax, m_yMin, m_yMax;
    osg::ref_ptr<osg::PositionAttitudeTransform> m_transformNode;

    int m_precision;
    osg::ref_ptr<osg::Group> m_rootNode;
    osg::ref_ptr<osg::Geode> m_subGeode;

    std::vector<CoordSystem::Graph *> m_graphs;

    osg::Vec3 m_cross;

    float m_scalingX, m_scalingZ;
};

#endif
