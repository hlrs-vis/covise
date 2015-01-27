/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef HfT_osg_Plugin01_Animation_H
#define HfT_osg_Plugin01_Animation_H

#include <osg/AnimationPath>
#include "HfT_osg_Plugin01_Cons.h"

class HfT_osg_Plugin01_Animation : public osg::AnimationPathCallback
{
public:
    // KONSTRUKTOREN:
    HfT_osg_Plugin01_Animation();
    HfT_osg_Plugin01_Animation(int dir);
    HfT_osg_Plugin01_Animation(HfT_osg_Plugin01_Cons *Cons);
    HfT_osg_Plugin01_Animation(HfT_osg_Plugin01_Cons *Cons, double radius);

    // Attribute
    int m_direction;
    double m_radius;
    HfT_osg_Plugin01_Cons *mp_Cons;
    osg::AnimationPath *mp_Animationpath;

    // Methoden
    double getRadius();
    int getDirection();
    HfT_osg_Plugin01_Cons *getCons();

    void createAnimationPath();
    osg::AnimationPath *getAnimationPath();

    // void operator()( osg::Node* node, osg::NodeVisitor* nv );
};
#endif
