/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _EQUATION_H
#define _EQUATION_H

#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/Material>
#include <osgText/Text>

#include <iostream>

class Equation
{
public:
    Equation();
    ~Equation();

    void setVisible(bool visible);
    void setEquation(std::string e);

    void createArrow();

private:
    std::string equation;

    osg::ref_ptr<osg::Group> group;

    osg::ref_ptr<osg::Material> material;

    osg::ref_ptr<osgText::Text> textDrawable;
    osg::ref_ptr<osg::Geode> textGeode;

    std::vector<osg::ref_ptr<osg::Geode> > helperGeodes;

    osg::ref_ptr<osg::MatrixTransform> arrowTransform;
};

#endif
