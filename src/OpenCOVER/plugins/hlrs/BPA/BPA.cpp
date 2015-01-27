/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include "BPA.h"
#include <osg/LineWidth>
#include <cover/coVRTui.h>
#include <cover/coVRFileManager.h>

BPAPlugin *BPAPlugin::plugin = NULL;

static const int NUM_HANDLERS = 3;

static const FileHandler handlers[] = {
    { NULL,
      BPAPlugin::SloadBPA,
      BPAPlugin::SloadBPA,
      BPAPlugin::SunloadBPA,
      "nfix" },
    { NULL,
      BPAPlugin::SloadBPA,
      BPAPlugin::SloadBPA,
      BPAPlugin::SunloadBPA,
      "nfi" },
    { NULL,
      BPAPlugin::SloadBPA,
      BPAPlugin::SloadBPA,
      BPAPlugin::SunloadBPA,
      "bpadxf" }
};

BPA::BPA(std::string filename, osg::Group *parent)
{
    fprintf(stderr, "BPA::BPA\n");
    int pos = BPAPlugin::plugin->bpa_map.size() * 3 + 1;

    velocity = new coTUIFloatSlider("velocity", BPAPlugin::plugin->BPATab->getID());
    velocity->setEventListener(this);
    velocity->setValue(6.0); // 6 m/s
    velocity->setMin(1.0);
    velocity->setMax(20.0);
    velocity->setPos(1, 0 + pos);

    velocityLabel = new coTUILabel("velocity", BPAPlugin::plugin->BPATab->getID());
    velocityLabel->setPos(0, 0 + pos);

    length = new coTUIFloatSlider("length", BPAPlugin::plugin->BPATab->getID());
    length->setEventListener(this);
    length->setValue(2.0);
    length->setMin(0.1);
    length->setMax(5.0);
    length->setPos(1, 1 + pos);

    lengthLabel = new coTUILabel("length:", BPAPlugin::plugin->BPATab->getID());
    lengthLabel->setPos(0, 1 + pos);

    lineColorLabel = new coTUILabel("line color", BPAPlugin::plugin->BPATab->getID());
    lineColorLabel->setPos(0, 2 + pos);
    lineColor = new coTUIColorButton("lineColor", BPAPlugin::plugin->BPATab->getID());
    lineColor->setColor(1, 0.9, 1, 1);
    lineColor->setPos(1, 2 + pos);
    lineColor->setEventListener(this);

    originLabel = new coTUILabel("origin:", BPAPlugin::plugin->BPATab->getID());
    originLabel->setPos(2, 2 + pos);

    filenameLabel = new coTUILabel(filename, BPAPlugin::plugin->BPATab->getID());
    filenameLabel->setPos(3, 1 + pos);

    trajectoriesGroup = new osg::Group();
    trajectoriesGroup->setName("BPA_Trajectories");

    int len = filename.length();
    if (len > 4)
    {
        if (strcmp(filename.c_str() + len - 4, ".nfi") == 0)
        {
            loadTxt(filename);
        }
        else if (strcmp(filename.c_str() + len - 5, ".nfix") == 0)
        {
            loadnfix(filename);
        }
        else
        {
            loadDxf(filename);
        }
    }
    if (parent)
        parent->addChild(trajectoriesGroup);
    else
        cover->getObjectsRoot()->addChild(trajectoriesGroup);

    std::list<Trajectory *>::iterator it;
    for (it = trajectories.begin(); it != trajectories.end(); it++)
    {
        if ((*it)->gamma < 0 || (*it)->gamma > M_PI) // sort them into left and right pointing
            right.push_back((*it));
        else
            left.push_back((*it));
    }
    geode = NULL;
    sphere = new osg::Sphere(osg::Vec3(0, 0, 0), 0.1);
    sphereDrawable = new osg::ShapeDrawable(sphere);
    geode = new osg::Geode();
    geode->addDrawable(sphereDrawable);
    osg::StateSet *stateset = geode->getOrCreateStateSet();
    osg::Material *material = new osg::Material;
    material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0, 0.0, 0.0, 1.0));
    material->setAlpha(osg::Material::FRONT_AND_BACK, 0.3);
    stateset->setAttributeAndModes(material, osg::StateAttribute::OVERRIDE);
    stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
    stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    stateset->setNestRenderBins(false);
    sphereTrans = new osg::MatrixTransform();
    sphereTrans->addChild(geode);

    trajectoriesGroup->addChild(sphereTrans);

    calcIntersection();
}

void BPA::calcIntersection()
{
    std::list<Trajectory *>::iterator itl;
    std::list<Trajectory *>::iterator itr;
    int numIntersections = 0;
    osg::Vec3 p;
    osg::Vec3Array *positions = new osg::Vec3Array(left.size() * right.size());
    for (itl = left.begin(); itl != left.end(); itl++)
    {
        for (itr = right.begin(); itr != right.end(); itr++)
        {
            osg::Vec3 tmpP;
            (*itl)->getMinimalDistance((*itr), tmpP);
            positions->at(numIntersections) = tmpP;
            p += tmpP;
            numIntersections++;
        }
    }
    p /= numIntersections;
    double S = 0;
    for (int i = 0; i < numIntersections; i++)
    {
        S += (p - positions->at(i)).length2();
    }
    S = sqrt(S / (numIntersections - 1));
    sphere->setRadius(S);
    sphereDrawable->dirtyDisplayList();
    sphereTrans->setMatrix(osg::Matrix::translate(p));
    char buf[1000];
    sprintf(buf, "Origin: %f %f %f, deviation=%f", p[0], p[1], p[2], (float)S);
    originLabel->setLabel(buf);
}

// this is called if the plugin is removed at runtime
BPA::~BPA()
{
    fprintf(stderr, "BPAPlugin::~BPAPlugin\n");
    delete lengthLabel;
    delete length;
    delete velocityLabel;
    delete velocity;
    delete lineColorLabel;
    delete lineColor;
    delete originLabel;
    delete filenameLabel;
    std::list<Trajectory *>::iterator it;
    for (it = trajectories.begin(); it != trajectories.end(); it++)
    {
        delete (*it);
    }
    trajectories.clear();
    if (trajectoriesGroup->getNumParents())
        trajectoriesGroup->getParent(0)->removeChild(trajectoriesGroup);
}

Trajectory::Trajectory(BPA *b)
{
    bpa = b;
    geode = new osg::Geode();
    //geode->setName(object_name);
    bpa->trajectoriesGroup->addChild(geode);
    bpa->trajectories.push_back(this);
    correctVelocity = false;
    D = 0.0;
    W = 0.0;
    mtl = new osg::Material;
    //mtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    mtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
    mtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
    mtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
    mtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0));

    mtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(bpa->lineColor->getRed(), bpa->lineColor->getGreen(), bpa->lineColor->getBlue(), bpa->lineColor->getAlpha()));
    mtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(bpa->lineColor->getRed(), bpa->lineColor->getGreen(), bpa->lineColor->getBlue(), bpa->lineColor->getAlpha()));
    mtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);

    osg::LineWidth *lw = new osg::LineWidth;
    lw->setWidth(2.0);

    lineState = new osg::StateSet();
    lineState->setAttributeAndModes(mtl, osg::StateAttribute::ON);
    lineState->setAttributeAndModes(lw, osg::StateAttribute::ON);
    lineState->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    lineState->setMode(GL_BLEND, osg::StateAttribute::OFF);
    lineState->setRenderingHint(osg::StateSet::OPAQUE_BIN);
    lineState->setNestRenderBins(false);
}

Trajectory::~Trajectory()
{
}
void Trajectory::setColor(float r, float g, float b, float a)
{
    mtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(r, g, b, a));
    mtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(r, g, b, a));
    lineState->setAttributeAndModes(mtl, osg::StateAttribute::ON);
}

void Trajectory::computeVelocity()
{
    double diff = 1;
    double oldD = 0;
    kappa = bpa->velocity->getValue();

    D = 2 * pow((3 * Vol / pow(10, 9) / (kappa * 4 * M_PI)), 1.0 / 3.0); // from dried volume (Vol) to D, kappa is drying ratio ~0.15

    velo = 5;
    int numIterations = 0;
    while (fabs(diff) > 0.000001 && numIterations < 1000)
    {
        diff = getX(velo);
        velo = velo - diff / getDX(velo);
        numIterations++;
    }
    if (numIterations >= 1000)
    {
        fprintf(stderr, "Could not solve the velocity equation\n");
    }

    double vMax = sqrt(12 / (21.533333333333333333333333333333 * D));
    //double We = 21.533333333333333333333333333333*D*velo*velo;// 1055.0 *D *v*v/0.06
    if (velo > vMax)
    {
        velo = vMax;
        setColor(1, 0, 0, 1);
    }
    else
    {
        setColor(bpa->lineColor->getRed(), bpa->lineColor->getGreen(), bpa->lineColor->getBlue(), bpa->lineColor->getAlpha());
    }
}

double Trajectory::getX(double v)
{
    double sina = sin(alpha);
    double We = 17583.333333333333333333333333333 * D * v * v; // 1055.0 *D *v*v/0.06
    double Re = 211000 * D * v; // 1055*D*v/0.005
    double P = We / pow(Re, 2.0 / 5.0);
    double sqrtp = sqrt(P);
    double powsin = pow(sina, 4.0 / 5.0);
    return ((pow(Re * sina, 1.0 / 5.0) * sqrtp * powsin / (1.24 + (sqrtp * powsin)))) - (W / D);
}

double Trajectory::getDX(double v)
{
    double vsina = v * sin(alpha);
    double We = 17583.333333333333333333333333333 * D * v * v; // 1055.0 *D *v*v/0.06
    double Re = 211000 * D * v; // 1055*D*v/0.005
    double P = We / pow(Re, 2.0 / 5.0);
    double sqrtvP = sqrt(pow(vsina, 8.0 / 5.0) * P);
    return ((-4 * pow(vsina, 4.0 / 5.0) * We / (5 * pow(Re, 1.0 / 5.0) * pow((1.24 + sqrtvP), 2.0))) + (4 * pow(vsina, 4.0 / 5.0) * We / (5 * pow(Re, 1.0 / 5.0) * sqrtvP * (1.24 + sqrtvP))) + (pow(Re, 1.0 / 5.0) * sqrtvP / (5 * pow(vsina, 4.0 / 5.0) * 1.24 + sqrtvP)));
}

void Trajectory::createGeometry()
{
    geom = new osg::Geometry();
    geom->setUseDisplayList(false);
    geom->setUseVertexBufferObjects(false);

    // set up geometry
    vert = new osg::Vec3Array;

    primitives = new osg::DrawArrays(osg::PrimitiveSet::LINE_STRIP, 0, 2);
    recalc();
    geom->addPrimitiveSet(primitives);

    geom->setStateSet(lineState);

    geode->removeDrawables(0, 1);
    geode->addDrawable(geom);
}

float Trajectory::getMinimalDistance(Trajectory *t, osg::Vec3 &p1)
{
    float minDist = 1000;
    int minI;
    int minN;
    for (int i = 0; i < vert->size(); i++)
    {
        for (int n = 0; n < t->vert->size(); n++)
        {
            float dist = (vert->at(i) - t->vert->at(n)).length2();
            if (dist < minDist)
            {
                minDist = dist;
                minI = i;
                minN = n;
            }
        }
    }
    p1 = (vert->at(minI) + t->vert->at(minN)) / 2.0;
    return sqrt(minDist);
}

void Trajectory::recalc()
{
    // set up geometry
    vert->clear();

    float dt = 0.01 / startVelocity.length();
    osg::Vec3 a;
    a.set(0, 0, -9.81);
    float len = 0;
    vert->push_back(startPos);
    osg::Vec3 pos = startPos;
    osg::Vec3 vel = startVelocity;
    bool res = BPAPlugin::plugin->airResistance->getState();
    float vMax = sqrt(12 / (21.533333333333333333333333333333 * D));
    bool Firsttime = true;
    while (len < length && pos[2] > 0.0)
    {
        float v = vel.length();
        if (res)
        {
            vel += ((vel * ((0.5 * 0.48 * 1.292 * v) / ((2.0 / 3.0) * D * 1055.0))) + a) * dt;
            float v = vel.length();

            if (v > vMax)
            {
                v = vMax;
                vel.normalize();
                vel *= v;
                if (Firsttime)
                {
                    setColor(1, 0, 0, 1);
                    Firsttime = false;
                }
            }
        }
        else
        {
            vel += a * dt;
        }
        pos = pos + vel * dt;
        len += (vel * dt).length();
        vert->push_back(pos);
    }
    primitives->setCount(vert->size());

    geom->setVertexArray(vert);
}

BPAPlugin::BPAPlugin()
{
    fprintf(stderr, "BPAPlugin::BPAPlugin\n");

    plugin = this;

    BPAGroup = new osg::Group();
    BPAGroup->setName("BPA_Trajectories");

    BPATab = new coTUITab("BPA", coVRTui::instance()->mainFolder->getID());
    BPATab->setPos(0, 0);

    airResistance = new coTUIToggleButton("Air Resistance", BPATab->getID(), true);
    airResistance->setEventListener(this);
    airResistance->setPos(0, 0);

    OriginComputationType = new coTUIToggleButton("Origin Computation Type", BPATab->getID(), true);
    OriginComputationType->setEventListener(this);
    OriginComputationType->setPos(1, 0);
    for (int index = 0; index < NUM_HANDLERS; index++)
        coVRFileManager::instance()->registerFileHandler(&handlers[index]);
}

// this is called if the plugin is removed at runtime
BPAPlugin::~BPAPlugin()
{
    fprintf(stderr, "BPAPlugin::~BPAPlugin\n");
    for (int index = 0; index < NUM_HANDLERS; index++)
        coVRFileManager::instance()->unregisterFileHandler(&handlers[index]);

    std::map<std::string, BPA *>::iterator it;
    for (it = bpa_map.begin(); it != bpa_map.end(); it++)
    {
        delete (*it).second;
    }
    bpa_map.clear();
    delete airResistance;
    delete OriginComputationType;
    delete BPATab;
    cover->getObjectsRoot()->removeChild(BPAGroup);
}

void BPAPlugin::tabletPressEvent(coTUIElement * /*tUIItem*/)
{
}

void BPAPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == airResistance)
    {
        std::map<std::string, BPA *>::iterator it;
        for (it = bpa_map.begin(); it != bpa_map.end(); it++)
        {
            (*it).second->recalc();
        }
    }
}

void BPA::recalc()
{
    std::list<Trajectory *>::iterator it;
    for (it = trajectories.begin(); it != trajectories.end(); it++)
    {
        (*it)->recalc();
    }

    calcIntersection();
}

void BPA::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == length || tUIItem == velocity)
    {
        std::list<Trajectory *>::iterator it;
        for (it = trajectories.begin(); it != trajectories.end(); it++)
        {
            (*it)->length = length->getValue();
            if (!(*it)->correctVelocity)
            {
                (*it)->startVelocity.normalize();
                (*it)->startVelocity *= velocity->getValue();
            }
            else
            {
                (*it)->computeVelocity();
                (*it)->startVelocity.normalize();
                (*it)->startVelocity *= (*it)->velo;
            }
        }
        recalc();
    }
    if (tUIItem == lineColor)
    {
        std::list<Trajectory *>::iterator it;
        for (it = trajectories.begin(); it != trajectories.end(); it++)
        {
            (*it)->setColor(lineColor->getRed(), lineColor->getGreen(), lineColor->getBlue(), lineColor->getAlpha());
        }
    }
}
void BPA::loadDxf(std::string filename)
{
    FILE *fp = fopen(filename.c_str(), "r");
    if (fp)
    {
        char buf[1000];
        while (!feof(fp))
        {
            if (fgets(buf, 1000, fp) != NULL)
            {
                if (strncmp(buf, "POLYLINE", 8) == 0)
                {
                    int vert = 0;
                    osg::Vec3 v[2];
                    while (fgets(buf, 1000, fp) != NULL)
                    {
                        if (strncmp(buf, "VERTEX", 6) == 0)
                        {
                            if (vert < 2)
                            {
                                float val;
                                fgets(buf, 1000, fp);
                                fgets(buf, 1000, fp);
                                fgets(buf, 1000, fp);
                                fgets(buf, 1000, fp);
                                sscanf(buf, "%f", &val);
                                v[vert][0] = val;
                                fgets(buf, 1000, fp);
                                fgets(buf, 1000, fp);
                                sscanf(buf, "%f", &val);
                                v[vert][1] = val;
                                fgets(buf, 1000, fp);
                                fgets(buf, 1000, fp);
                                sscanf(buf, "%f", &val);
                                v[vert][2] = val;
                                vert++;
                            }
                            else
                            {
                                fprintf(stderr, "too many Vertices\n");
                            }
                        }
                        if (strncmp(buf, "SEQEND", 6) == 0)
                        {
                            Trajectory *t = new Trajectory(this);
                            t->startPos = v[1];
                            t->startVelocity = v[0] - v[1];
                            t->length = length->getValue();
                            t->startVelocity.normalize();
                            t->startVelocity *= velocity->getValue();
                            t->createGeometry();
                            break;
                        }
                    }
                }
            }
        }
        fclose(fp);
    }
}

void BPA::loadnfix(std::string filename)
{
    velocityLabel->setLabel("Kappa");
    velocity->setMin(0.05);
    velocity->setMax(0.3);
    velocity->setValue(0.15);
    FILE *fp = fopen(filename.c_str(), "r");
    if (fp)
    {
        char buf[1000];
        while (!feof(fp))
        {
            if (fgets(buf, 1000, fp) != NULL)
            {
                // 0.564755053	1.4		38.91	-14.63	0	1.789	1.945
                float W, alpha, beta, gamma, x, y, z, Vol;
                sscanf(buf, "%f %f %f %f %f %f %f %f", &Vol, &W, &alpha, &gamma, &beta, &x, &y, &z);
                Trajectory *t = new Trajectory(this);
                osg::Vec3 pos;
                pos.set(x, y, z);
                t->startPos = pos;
                float a = alpha / 180.0 * M_PI;
                float b = beta / 180.0 * M_PI;
                float c = gamma / 180.0 * M_PI;
                osg::Vec3 v;
                v.set(sin(a), -sin(c) * cos(a), -cos(c) * cos(a));
                osg::Matrix m;
                m.makeRotate(b, osg::Vec3(0, 0, 1));
                v = osg::Matrix::transform3x3(m, v);
                v.normalize();
                t->alpha = a;
                t->gamma = c;
                t->W = W / 1000.0;
                t->Vol = Vol;
                t->computeVelocity();
                t->startVelocity = v * t->velo;
                t->length = length->getValue();
                t->correctVelocity = true;
                t->createGeometry();
            }
        }
        fclose(fp);
    }
}
void BPA::loadTxt(std::string filename)
{
    FILE *fp = fopen(filename.c_str(), "r");
    if (fp)
    {
        char buf[1000];
        while (!feof(fp))
        {
            if (fgets(buf, 1000, fp) != NULL)
            {
                // 8.151031135	37.01	-29.35	0	1.4	1.844
                float velo, alpha, beta, gamma, x, y, z;
                sscanf(buf, "%f %f %f %f %f %f %f", &velo, &alpha, &gamma, &beta, &x, &y, &z);
                Trajectory *t = new Trajectory(this);
                osg::Vec3 pos;
                pos.set(x, y, z);
                t->startPos = pos;
                float a = alpha / 180.0 * M_PI;
                float b = beta / 180.0 * M_PI;
                float c = gamma / 180.0 * M_PI;
                osg::Vec3 v;
                v.set(sin(a), -sin(c) * cos(a), -cos(c) * cos(a));
                osg::Matrix m;
                m.makeRotate(beta, osg::Vec3(0, 0, 1));
                v = osg::Matrix::transform3x3(m, v);
                t->startVelocity = v * velo;
                t->length = length->getValue();
                t->correctVelocity = true;
                t->createGeometry();
            }
        }
        fclose(fp);
    }
}

bool BPAPlugin::init()
{
    cover->getObjectsRoot()->addChild(BPAGroup);
    return true;
}

void
BPAPlugin::preFrame()
{
}

int BPAPlugin::SunloadBPA(const char *filename, const char *)
{
    return plugin->unloadBPA(filename);
}

int BPAPlugin::unloadBPA(const char *name)
{
    std::map<std::string, BPA *>::iterator it = bpa_map.find(std::string(name));

    if (it != bpa_map.end())
    {
        BPA *b = bpa_map[(char *)name];
        bpa_map.erase(it);
        delete b;
        return 0;
    }
    return 1;
}

int BPAPlugin::SloadBPA(const char *filename, osg::Group *parent, const char *)
{

    if (filename)
    {

        plugin->loadBPA(filename, parent);
    }

    return 0;
}

int BPAPlugin::loadBPA(const char *filename, osg::Group *parent)
{
    BPA *b = new BPA(filename, parent);
    bpa_map[std::string(filename)] = b;
    return 0;
}

COVERPLUGIN(BPAPlugin)
