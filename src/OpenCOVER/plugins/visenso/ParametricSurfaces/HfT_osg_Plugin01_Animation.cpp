/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Walkpath: immer entlang der Flächenkurve

#include "HfT_osg_Plugin01_Animation.h"

using namespace osg;

HfT_osg_Plugin01_Animation::HfT_osg_Plugin01_Animation()
{
}
HfT_osg_Plugin01_Animation::HfT_osg_Plugin01_Animation(int dir)
{
    m_direction = dir;
}
HfT_osg_Plugin01_Animation::HfT_osg_Plugin01_Animation(HfT_osg_Plugin01_Cons *Cons)
{
    mp_Cons = Cons;
    m_radius = 1;
    createAnimationPath();
    this->setAnimationPath(mp_Animationpath);
}
HfT_osg_Plugin01_Animation::HfT_osg_Plugin01_Animation(HfT_osg_Plugin01_Cons *Cons, double radius)
{
    mp_Cons = Cons;
    m_radius = radius;

    createAnimationPath();
    this->setAnimationPath(mp_Animationpath);
}

double HfT_osg_Plugin01_Animation::getRadius()
{
    return (m_radius);
}
int HfT_osg_Plugin01_Animation::getDirection()
{
    return (m_direction);
}
HfT_osg_Plugin01_Cons *HfT_osg_Plugin01_Animation::getCons()
{
    return (mp_Cons);
}

void HfT_osg_Plugin01_Animation::createAnimationPath()
{
    double t;
    mp_Animationpath = new AnimationPath;
    AnimationPath::ControlPoint cp;
    Vec3Array *varray = mp_Cons->getPoints();
    Vec3Array *narray = mp_Cons->getNormals();

    Vec3 pm, nm, kp;

    for (unsigned int i = 0; i < varray->size(); i++)
    {
        t = (3.0 * i) / varray->size();

        pm = (*varray)[i];
        nm = (*narray)[i];
        kp[0] = pm[0] + m_radius * nm[0];
        kp[1] = pm[1] + m_radius * nm[1];
        kp[2] = pm[2] + m_radius * nm[2];

        cp.setPosition(kp);
        // Animations-Geschwindigkeit mit Faktor 10
        t = (10.0 * i) / varray->size();
        mp_Animationpath->insert(t, cp);
    }
    mp_Animationpath->setLoopMode(osg::AnimationPath::SWING);
    return;
}
osg::AnimationPath *HfT_osg_Plugin01_Animation::getAnimationPath()
{
    return mp_Animationpath;
}
/*muss ausgeklammert sein
void HfT_osg_Plugin01_Animation::operator()( osg::Node* node, osg::NodeVisitor* nv )
{
	int typ,direction;
	double ua,ue,um,va,ve,vm,radius;

	osg::MatrixTransform* mt =
 		dynamic_cast<osg::MatrixTransform*>( node );
	ua = get_ua(); ue= get_ue(); um =get_um();
	va = get_va(); ve= get_ve(); vm =get_vm();
	radius = get_radius();
	direction = get_direction(); typ = get_typ();
	AnimationPath *ap;

    if(direction) // Animation entlang u-Parameterlinie
	{
		
 		 ap = HfT_osg_WPath_u(typ, radius, ua, ue, vm);
	}
	else
	{
		 ap = HfT_osg_WPath_v(typ, radius, va, ve, um);
	}

 	this->setAnimationPath(ap); 
	this->update(*node);

	traverse( node, nv );
}
*/
