/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coPin.h"
#include <virvo/vvtransfunc.h>

#include <cover/coVRPluginSupport.h>

#include <OpenVRUI/osg/OSGVruiTransformNode.h>
#include <OpenVRUI/osg/OSGVruiPresets.h>

#include <osg/CullFace>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>

int coPin::numAlphaPins = 0;
int coPin::numPins = 0;

using namespace osg;
using namespace vrui;
using namespace opencover;

coPin::coPin(Group *root, float Height, float Width, vvTFWidget *myPin,bool bottom)
{
    myX = 0.0;
    A = 0.6;
    B = 0.5;
    H = Height;
    W = Width;
	onBottom = bottom;
    jPin = myPin;
    selected = false;
    createLists();
    myDCS = new OSGVruiTransformNode(new MatrixTransform());
    selectionDCS = new MatrixTransform();
    geode = createLineGeode();
    selectionDCS->addChild(geode.get());
    myDCS->getNodePtr()->asGroup()->addChild(selectionDCS.get());
    myDCS->getNodePtr()->setNodeMask(myDCS->getNodePtr()->getNodeMask() & ~Isect::Intersection);
    root->addChild(myDCS->getNodePtr());
    numPins++;
    id = numPins;
    oldScale = Vec3(1., 1., 1.);
    _handleTrans = 0.;
}

coPin::~coPin()
{
    myDCS->removeAllChildren();
    myDCS->removeAllParents();
    delete myDCS;
}

int coPin::getID()
{
    return id;
}

void coPin::setPos(float x, float minv, float maxv)
{
    myX = (x - minv) / (maxv - minv);
    myDCS->setTranslation(myX * W, 0.0, 0.0);
    jPin->_pos[0] = x;
}

float coPin::getPosValue() const
{
    return jPin->pos()[0];
}

float coPin::getPos01() const
{
    return myX;
}

void coPin::setHandleTrans(float trans)
{
    _handleTrans = trans;
    osg::Matrix mat = selectionDCS->getMatrix();
    mat.setTrans(trans * W, 0., 0.);
    selectionDCS->setMatrix(mat);
}

void coPin::select()
{
    Matrix m = selectionDCS->getMatrix();
    oldScale = m.getScale();
    Matrix sm;
    sm.makeScale(3.0 / oldScale[0], 1.0 / oldScale[1], 3.0 / oldScale[2]);
    //Vec3 x = m.getTrans();
    m.setTrans(0.0, 0.0, 0.2);
    m *= sm;
    selectionDCS->setMatrix(m);
}

void coPin::deSelect()
{
    Matrix m = selectionDCS->getMatrix();
    m.setTrans(0.0, 0.0, 0.0);
    m(0, 0) = oldScale[0];
    m(1, 1) = oldScale[1];
    m(2, 2) = oldScale[2];
    selectionDCS->setMatrix(m);
}

vruiTransformNode *coPin::getDCS()
{
    return myDCS;
}

void coPin::createLists()
{
    color = new Vec4Array(1);
    coord = new Vec3Array(8);
    normal = new Vec3Array(2);

	if(onBottom)
	{
		(*coord)[0].set(B, -(H / 2.0f), 0.0);
		(*coord)[1].set(0.0, -(H / 2.0f), A);
		(*coord)[2].set(0.0, -(H), A);
		(*coord)[3].set(B, -(H), 0.0);

		(*coord)[4].set(0.0, -(H / 2.0f), A);
		(*coord)[5].set(-B, -(H / 2.0f), 0.0);
		(*coord)[6].set(-B, -(H), 0.0);
		(*coord)[7].set(0.0, -(H), A);
	}
	else
	{
		(*coord)[0].set(B, 0.0, 0.0);
		(*coord)[1].set(0.0, 0.0, A);
		(*coord)[2].set(0.0, -(H / 2.0f), A);
		(*coord)[3].set(B, -(H / 2.0f), 0.0);

		(*coord)[4].set(0.0, 0.0, A);
		(*coord)[5].set(-B, 0.0, 0.0);
		(*coord)[6].set(-B, -(H / 2.0f), 0.0);
		(*coord)[7].set(0.0, -(H / 2.0f), A);
	}


#if 0
   coord1 = new Vec3Array(24);
   // alternative coordinates for pyramids
   (*coord1)[0 ].set(B        , 0.0  , 0.0);
   (*coord1)[1 ].set(0.0      , 0.0  , A  );
   (*coord1)[2 ].set(0.0      , -(H) , A  );
   (*coord1)[3 ].set(B        , -(H) , 0.0);

   (*coord1)[4 ].set(0.0      , 0.0  , A  );
   (*coord1)[5 ].set(-B       , 0.0  , 0.0);
   (*coord1)[6 ].set(-B       , -(H) , 0.0);
   (*coord1)[7 ].set(0.0      , -(H) , A  );

   (*coord1)[8 ].set(B        , 0.0  , 0.0);
   (*coord1)[9 ].set(0.0      , 0.0  , A  );
   (*coord1)[10].set(0.0      , -(H) , A  );
   (*coord1)[11].set(B        , -(H) , 0.0);

   (*coord1)[12].set(0.0      , 0.0  , A  );
   (*coord1)[13].set(-B       , 0.0  , 0.0);
   (*coord1)[14].set(-B       , -(H) , 0.0);
   (*coord1)[15].set(0.0      , -(H) , A  );

   (*coord1)[16].set(B        , 0.0  , 0.0);
   (*coord1)[17].set(0.0      , 0.0  , A  );
   (*coord1)[18].set(0.0      , -(H) , A  );
   (*coord1)[19].set(B        , -(H) , 0.0);

   (*coord1)[20].set(0.0      , 0.0  , A  );
   (*coord1)[21].set(-B       , 0.0  , 0.0);
   (*coord1)[22].set(-B       , -(H) , 0.0);
   (*coord1)[23].set(0.0      , -(H) , A  );
#endif

    (*color)[0].set(0.8f, 0.8f, 0.8f, 1.0f);

    (*normal)[0].set(A, 0, B);
    (*normal)[1].set(-A, 0, B);

    (*normal)[0].normalize();
    (*normal)[1].normalize();

    ref_ptr<Material> mtl = new Material();
    mtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    mtl->setAmbient(Material::FRONT_AND_BACK, Vec4(0.1, 0.1, 0.1, 1.0));
    mtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.6, 0.6, 0.6, 1.0));
    mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(1.0, 1.0, 1.0, 1.0));
    mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0, 0.0, 0.0, 1.0));
    mtl->setShininess(Material::FRONT_AND_BACK, 80.0f);

    normalGeostate = new StateSet();
    normalGeostate->setGlobalDefaults();

    normalGeostate->setAttributeAndModes(OSGVruiPresets::getCullFaceBack(), StateAttribute::ON);
    normalGeostate->setAttributeAndModes(mtl.get(), StateAttribute::ON);

    normalGeostate->setMode(GL_LIGHTING, StateAttribute::ON);
    normalGeostate->setMode(GL_BLEND, StateAttribute::ON);
}

Geode *coPin::createLineGeode()
{
    ref_ptr<Geometry> geoset1 = new Geometry();

    geoset1->setColorArray(color.get());
    geoset1->setColorBinding(Geometry::BIND_OVERALL);
    geoset1->setVertexArray(coord.get());
    geoset1->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 8));
    geoset1->setNormalArray(normal.get());
    geoset1->setNormalBinding(Geometry::BIND_OVERALL);

    Geode *myGeode = new Geode();
    myGeode->setStateSet(normalGeostate.get());
    myGeode->addDrawable(geoset1.get());
    return myGeode;
}
