/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_WIDGET_H_
#define _CUI_WIDGET_H_

// OpenSceneGraph:
#include <osg/MatrixTransform>
#include <osgText/Font>
#ifdef WIN32
#if defined(COMPILE_CUI)
#define CUIEXPORT __declspec(dllexport)
#else
#define CUIEXPORT __declspec(dllimport)
#endif
#elif defined(__GNUC__) && __GNUC__ >= 4
#define CUIEXPORT __attribute__((visibility("default")))
#define CUIIMPORT CUIEXPORT
#else
#define CUIEXPORT
#endif

namespace cui
{

/** This is the parent class of all user interface widgets.
    The coordinate system is the same as OpenGL:
    x=right, y=up, z=towards viewer:
    <pre>
      y
      |
      |
      ------x
    /
   z
    </pre>
  Coordinate units are always millimeters [mm].
  */
class LogFile;

class CUIEXPORT Widget
{
public:
    static const float EPSILON_Z; ///< minimum z distance to be distinguished by z buffer
    static const osg::Vec4 COL_RED;
    static const osg::Vec4 COL_GREEN;
    static const osg::Vec4 COL_BLUE;
    static const osg::Vec4 COL_BLACK;
    static const osg::Vec4 COL_WHITE;
    static const osg::Vec4 COL_LIGHT_GRAY;
    static const osg::Vec4 COL_DARK_GRAY;
    static const osg::Vec4 COL_YELLOW;
    static int getNewGroupID();
    enum AngleType
    {
        X,
        Y,
        Z
    };

    Widget();
    virtual ~Widget();
    virtual void setMatrix(const osg::Matrix &);
    virtual osg::Matrix getMatrix();
    virtual osg::MatrixTransform *getNode();
    virtual bool isHighlighted();
    virtual void setHighlighted(bool);
    virtual bool hasFocus();
    virtual void setFocus(bool);
    virtual void setFont(osgText::Font *);
    virtual void setVisible(bool);
    virtual bool isVisible();
    virtual float angle(osg::Matrix &, AngleType);
    virtual float angleDiff(osg::Matrix &, osg::Matrix &, AngleType);
    virtual void computeEulerAngles(osg::Matrix &, float &, float &, float &);
    virtual float vectorAnglePlane(const osg::Vec3 &, const osg::Vec3 &);
    virtual float vectorAngle(const osg::Vec3 &, const osg::Vec3 &, int);
    virtual bool isInside();
    virtual int getGroupID()
    {
        return _groupID;
    }
    virtual void setGroupID(int id)
    {
        _groupID = id;
    }
    virtual void setLogFile(LogFile *lf)
    {
        _logFile = lf;
    }

protected:
    osg::ref_ptr<osg::MatrixTransform> _node; ///< everything below this widget is attached to this matrix transformation node
    osgText::Font *_font; ///< font; defaults to NULL
    bool _visible; ///< true=visible
    bool _highlighted; ///< true=widget is highlighted
    bool _focus; ///< true=widget has keyboard/mouse input focus
    bool _inside;
    static std::string _resourcePath; ///< path to resources like icon textures etc.
    static int ID_COUNTER;
    int _groupID;
    LogFile *_logFile;
    char _logBuf[128];

    void initResourcePath();
};
}
#endif
