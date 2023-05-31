/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/coInteractor.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRShader.h>

#include <osg/Group>
#include <osg/Node>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/Switch>
#include <osg/TexGenNode>
#include <osg/Geode>
#include <osg/Point>
#include <osg/ShapeDrawable>

#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coNavInteraction.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coFlatPanelGeometry.h>
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coRectButtonGeometry.h>
#include <OpenVRUI/coMouseButtonInteraction.h>
#include <cover/coBillboard.h>
#include <cover/VRVruiRenderInterface.h>

#include <PluginUtil/PluginMessageTypes.h>

#include "Bee.h"

using namespace osg;

Bee *Bee::plugin = NULL;

Bee::Bee()
: coVRPlugin(COVER_PLUGIN_NAME)
{

    float layerOffset = 20;
    offsets = new osg::Vec3Array();
    offsets->resize(8);
    (*offsets)[North].set(0, 5, 0);
    (*offsets)[NorthEast].set(4.33, 2.5, 0);
    (*offsets)[SouthEast].set(4.33, -2.5, 0);
    (*offsets)[South].set(0, -5, 0);
    (*offsets)[SouthWest].set(-4.33, -2.5, 0);
    (*offsets)[NorthWest].set(-4.33, 2.5, 0);
    (*offsets)[Top].set(0, 0, layerOffset);
    (*offsets)[Bottom].set(0, 0, -layerOffset);
}
Bee::~Bee()
{
}

bool Bee::destroy()
{
    cover->getObjectsRoot()->removeChild(geode);
    return true;
}

void Bee::preFrame()
{
    for (std::list<hiveLayer *>::iterator it = hiveLayers.begin(); it != hiveLayers.end(); it++)
    {
        hiveLayer *h = *it;
        h->update();
        if (h->growing.size() < 20)
        {
            int num = 20 - h->growing.size();
            if (num > 5)
                num = 5;
            h->addNew(num);
        }
    }
}

int Bee::addComb(comb *c, position p, comb *newc)
{
    if (c)
    {
        vert->push_back((*vert)[c->number] + (*offsets)[p]);
        // set neighbors
        if (p == Top)
        {
            c->neighbors[Top] = newc;
            newc->neighbors[Bottom] = c;
        }
        else if (p == Bottom)
        {
            c->neighbors[Bottom] = newc;
            newc->neighbors[Top] = c;
        }
        else
        {
            c->neighbors[p] = newc;
            newc->neighbors[(p + 3) % 6] = c;
            for (int i = 1; i < 6; i++)
            {
                comb *nc = c->getNeighborComb(p, i);
                if (nc)
                {
                    nc->neighbors[(p + 6 - i) % 6] = newc;
                    newc->neighbors[(p + 3 - i) % 6] = nc;
                }
            }
        }
    }
    else
    {
        vert->push_back(osg::Vec3(0, 0, 0));
    }

    colArr->push_back(osg::Vec4(0.0, 1, 1, 0.0));
    normalArr->push_back(osg::Vec3(0, 0, 1));
    (*primitives)[0] = vert->size();
	vert->dirty();
	colArr->dirty();
	normalArr->dirty();
	primitives->dirty();
    geom->setColorArray(colArr);
    geom->setNormalArray(normalArr);
    geom->setVertexArray(vert);
    geom->setPrimitiveSet(0, primitives);
    geom->dirtyDisplayList();

    return vert->size() - 1;
}

void Bee::setHeight(int n, float h)
{
    (*colArr)[n].r() = h;
    (*normalArr)[n].x() = h;
}

float Bee::getHeight(int n)
{
    return (*colArr)[n].r();
}

void Bee::setCap(int n, float c)
{
    (*colArr)[n].a() = c;
    (*normalArr)[n].y() = c;
}

comb::comb(comb *c, Bee::position p)
{
    for (int i = 0; i < 8; i++)
    {
        neighbors[i] = NULL;
    }
    number = Bee::plugin->addComb(c, p, this);
    int num = 0;
    targetHeight = 0;
    for (int i = 0; i < 6; i++)
    {
        if (neighbors[i] != NULL)
        {
            num++;
            targetHeight += neighbors[i]->targetHeight;
        }
    }
    if (num == 0)
    {
        targetHeight = 8;
    }
    else
    {
        targetHeight /= (float)num;
    }
    targetHeight += 1 - ((rand() * 2.0) / RAND_MAX);
}

comb::~comb()
{
    Bee::plugin->setHeight(number, 0.0);
    Bee::plugin->setCap(number, 0.0);
}

comb *comb::getNeighborComb(Bee::position p, int n)
{
    comb *c = this;
    int index = (p + 1) % 6;
    for (int i = 0; i < n; i++) // search left
    {
        if (c->neighbors[index] == NULL)
            break;
        c = c->neighbors[index];
        index = (index + 5) % 6;
        if (i == n - 1)
            return c;
    }
    index = p - 1;
    if (index < 0)
        index += 6;
    for (int i = 0; i < n; i++) // search right
    {
        if (c->neighbors[index] == NULL)
            return NULL;
        c = c->neighbors[index];
        index = (index + 2) % 6;
    }
    return c;
}

int comb::getNumNeighbors() const
{
    int numN = 0;
    for (int i = 0; i < 6; i++)
    {
        if (neighbors[i] != NULL)
            numN++;
    }
    return numN;
}
Bee::position comb::freePosition()
{
    if (neighbors[0] == 0)
    {
        for (int i = 1; i < 6; i++)
        {
            if (neighbors[i] != NULL)
            {
                return (Bee::position)(i - 1);
            }
        }
        return Bee::North;
    }
    else
    {
        for (int i = 5; i >= 0; i--)
        {
            if (neighbors[i] == NULL)
            {
                return (Bee::position)(i);
            }
        }
    }
    return Bee::NoFreePosition;
}

void comb::update()
{
    float h = Bee::plugin->getHeight(number);
    if (h < targetHeight)
        h += Bee::plugin->growSpeed;
    if (h > targetHeight)
        h = targetHeight;
    Bee::plugin->setHeight(number, h);
}

hiveLayer::hiveLayer(comb *c, Bee::position p)
{
    nextLayer = NULL;
    root = new comb(c, p);
    all.push_back(root);
    growing.push_back(root);
    outerCombs.push_back(root);
}

hiveLayer::~hiveLayer()
{
    for (std::list<comb *>::iterator it = all.begin(); it != all.end(); it++)
    {
        delete *it;
    }
    root = NULL;
}
void hiveLayer::update()
{
    for (std::list<comb *>::iterator it = growing.begin(); it != growing.end(); it++)
    {
        comb *c = *it;
        c->update();
        float h = Bee::plugin->getHeight(c->number);
        if (h >= c->targetHeight)
        {
            if (it == growing.begin())
            {
                growing.erase(it);
                it = growing.begin();
                if (growing.size() == 0)
                    break;
            }
            else
            {
                std::list<comb *>::iterator itremove = it;
                it--;
                growing.erase(itremove);
                if (growing.size() == 0)
                    break;
            }
        }
    }
}

class coCombCompare
{
public:
    bool operator()(const comb *, const comb *) const;
};

bool coCombCompare::operator()(const comb *p1, const comb *p2) const
{
    return p1->getNumNeighbors() > p2->getNumNeighbors();
}
void hiveLayer::addNew(int num)
{
    /*outerCombs.sort(coCombCompare());
	int i=0;
	for(std::list<comb *>::iterator it = outerCombs.begin();it!=outerCombs.end() && i < num;it++)
	{
		i++;

	}*/

    for (int i = 0; i < num; i++)
    {
        std::list<comb *>::iterator it = outerCombs.end();
        if (it != outerCombs.begin())
        {
            it--;
            Bee::position p = (*it)->freePosition();
            if (p == Bee::NoFreePosition)
            {
                cerr << "this should not happen, this comb should not be in outerCombs" << endl;
                outerCombs.remove(*it);
            }
            else
            {
                comb *c = new comb(*it, p);
                if ((*it)->getNumNeighbors() == 6)
                {
                    outerCombs.remove(*it);
                }
                all.push_back(c);
                growing.push_back(c);
                outerCombs.push_back(c);
            }
        }
    }
}

bool Bee::init()
{

    if (Bee::plugin != NULL)
        return false;
    growSpeed = 0.1;

    Bee::plugin = this;

    geode = new osg::Geode();
    geode->setName("points");
    geom = new osg::Geometry();
    //geom->setUseVertexBufferObjects(true);

    geom->setUseDisplayList(true);
    geom->setUseVertexBufferObjects(false);

    // set up geometry
    vert = new osg::Vec3Array;
    primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::POINTS);
    colArr = new osg::Vec4Array();
    normalArr = new osg::Vec3Array();

    primitives->push_back(0);

    hiveLayers.push_back(new hiveLayer(NULL, Top));
    /* int nx=30;
   int ny=30;
   int nz=2;
   primitives->push_back(nx*ny*nz);
   for(int x=0; x<nx; x++)
   for(int y=0; y<ny; y++)
   for(int z=0; z<nz; z++)
   {
      vert->push_back( osg::Vec3(x*4.33, y*5+(x%2)*2.5, z*10.0));
      colArr->push_back( osg::Vec4((float)x/(float)(nx-1), 1, 1,(float)y/(float)(ny-1)));
   }*/
    geom->setColorArray(colArr);
    geom->setNormalArray(normalArr);
    geom->setVertexArray(vert);
    geom->addPrimitiveSet(primitives);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    geode->addDrawable(geom);

    osg::StateSet *geoState = geode->getOrCreateStateSet();

    osg::Material *mtl = new osg::Material;
    mtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    mtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.0f, 1.0f));
    mtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.0f, 1.0f));
    mtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0f));
    mtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    mtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);

    geoState->setAttributeAndModes(mtl, osg::StateAttribute::ON);

    osg::Point *point = new osg::Point();
    point->setSize(2.5);
    geoState->setAttributeAndModes(point, osg::StateAttribute::ON);

    geode->setStateSet(geoState);
    coVRShader *hc = coVRShaderList::instance()->get("honeycomb");
    hc->apply(geode, geom);

    cover->getObjectsRoot()->addChild(geode);

    return true;
}

COVERPLUGIN(Bee)
