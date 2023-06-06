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
      BPAPlugin::SunloadBPA,
      "nfix" },
    { NULL,
      BPAPlugin::SloadBPA,
      BPAPlugin::SunloadBPA,
      "nfi" },
    { NULL,
      BPAPlugin::SloadBPA,
      BPAPlugin::SunloadBPA,
      "bpadxf" }
};

BPA::BPA(std::string filename, osg::Group *parent)
{
    fprintf(stderr, "BPA::BPA\n");
    
    floorHeight = 0;
    int pos = BPAPlugin::plugin->bpa_map.size() * 5 + 1;

    velocity = new coTUIFloatSlider("velocity", BPAPlugin::plugin->BPATab->getID());
    velocity->setEventListener(this);
    velocity->setValue(6.0); // 6 m/s
    velocity->setMin(1.0);
    velocity->setMax(22.0);
    velocity->setPos(1, 0 + pos);

    velocityLabel = new coTUILabel("velocity", BPAPlugin::plugin->BPATab->getID());
    velocityLabel->setPos(0, 0 + pos);
    originVelocityLabel = new coTUILabel("originVelocity", BPAPlugin::plugin->BPATab->getID());
    originVelocityLabel->setPos(3, 0 + pos);

    originVelocity = new coTUIFloatSlider("originVelocity", BPAPlugin::plugin->BPATab->getID());
    originVelocity->setEventListener(this);
    originVelocity->setValue(6.0); // 6 m/s
    originVelocity->setMin(1.0);
    originVelocity->setMax(20.0);
    originVelocity->setPos(4, 0 + pos);

    rhoLabel = new coTUILabel("rho", BPAPlugin::plugin->BPATab->getID());
    rhoLabel->setPos(0, 2 + pos);
    rhoEdit = new coTUIEditFloatField("rhoEdit", BPAPlugin::plugin->BPATab->getID());
    rhoEdit->setPos(1, 2 + pos);
    rhoEdit->setValue(1055.0);
    rhoEdit->setEventListener(this);
    viscosityLabel = new coTUILabel("viscosity", BPAPlugin::plugin->BPATab->getID());
    viscosityLabel->setPos(2, 2 + pos);
    viscosityEdit = new coTUIEditFloatField("viscosityEdit", BPAPlugin::plugin->BPATab->getID());
    viscosityEdit->setPos(3, 2 + pos);
    viscosityEdit->setValue(0.005);
    viscosityEdit->setEventListener(this);
    stLabel = new coTUILabel("surface tension", BPAPlugin::plugin->BPATab->getID());
    stLabel->setPos(4, 2 + pos);
    stEdit = new coTUIEditFloatField("surfaceTensionEdit", BPAPlugin::plugin->BPATab->getID());
    stEdit->setPos(5, 2 + pos);
    stEdit->setValue(0.06);
    stEdit->setEventListener(this);

    length = new coTUIFloatSlider("length", BPAPlugin::plugin->BPATab->getID());
    length->setEventListener(this);
    length->setValue(2.0);
    length->setMin(0.1);
    length->setMax(5.0);
    length->setPos(1, 1 + pos);

    lengthLabel = new coTUILabel("length:", BPAPlugin::plugin->BPATab->getID());
    lengthLabel->setPos(0, 1 + pos);

    lineColorLabel = new coTUILabel("line color", BPAPlugin::plugin->BPATab->getID());
    lineColorLabel->setPos(0, 3 + pos);
    lineColor = new coTUIColorButton("lineColor", BPAPlugin::plugin->BPATab->getID());
    lineColor->setColor(1, 0.9, 1, 1);
    lineColor->setPos(1, 3 + pos);
    lineColor->setEventListener(this);

    originLabel = new coTUILabel("origin:", BPAPlugin::plugin->BPATab->getID());
    originLabel->setPos(2, 3 + pos);

    filenameLabel = new coTUILabel(filename, BPAPlugin::plugin->BPATab->getID());
    filenameLabel->setPos(3, 1 + pos);

    minErrorButton = new coTUIButton("minError", BPAPlugin::plugin->BPATab->getID());
    minErrorButton->setPos(0, 4+pos);
    minErrorButton->setEventListener(this);

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
    float minAngle = 1000;
    float maxAngle = -1000;
    for (it = trajectories.begin(); it != trajectories.end(); it++)
    {
        if(!(*it)->correctVelocity)
        {
            float angle = atan2((*it)->startVelocity[1],(*it)->startVelocity[0]);
            if(angle< minAngle)
                minAngle = angle;
            if(angle> maxAngle)
                maxAngle = angle;
        }
    }
    float midAngle = minAngle + ((maxAngle - minAngle)/2.0);

    for (it = trajectories.begin(); it != trajectories.end(); it++)
    {
        if((*it)->correctVelocity)
        {
            if ((*it)->gamma < 0 || (*it)->gamma > M_PI) // sort them into left and right pointing
                right.push_back((*it));
            else
                left.push_back((*it));
        }
        else
        {
            float angle = atan2((*it)->startVelocity[1],(*it)->startVelocity[0]);
            if (angle < midAngle) // sort them into left and right pointing
                right.push_back((*it));
            else
                left.push_back((*it));
        }
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




void BPA::intersectLines(osg::Vec3 p0,osg::Vec3 p1,osg::Vec3 d0,osg::Vec3 d1,osg::Vec3 &c0, osg::Vec3 &c1)
{
    float a = d0 * d0;
    float b = d0 * d1;
    float c = d1 * d1;
    osg::Vec3 w0 = p0 - p1;
    float d = d0 * w0;
    float e = d1 * w0;
    float tmp = ((a*c) - (b*b));
    float s0 = ((b*e)-(c*d)) / tmp;
    float s1 = ((a*e)-(b*d)) / tmp;
    c0 = p0 + (d0*s0);
    c1 = p1 + (d1*s1);
}
/*float BPA::distancePointLine(osg::Vec3 p0, osg::Vec3 d0, osg::Vec3 p)
{
    return(d0 ^ (p0-p)).length()/d0.length();
}*/

void BPA::calcIntersection()
{
    std::list<Trajectory *>::iterator itl;
    std::list<Trajectory *>::iterator itr;
    int numIntersections = 0;
    osg::Vec3 p;
    osg::Vec3Array *positions=NULL;
    if(BPAPlugin::plugin->allToAll->getState())
    {
        float angleThreshold = BPAPlugin::plugin->angleEdit->getValue() * M_PI / 180.0;
        positions = new osg::Vec3Array(trajectories.size() * trajectories.size());
        for (itl = trajectories.begin(); itl != trajectories.end(); itl++)
        {
            for (itr = trajectories.begin(); itr != trajectories.end(); itr++)
            {
                if(*itl != *itr)
                {
                    osg::Vec3 s1 = (*itl)->startVelocity;
                    osg::Vec3 s2 = (*itr)->startVelocity;
                    s1.normalize();
                    s2.normalize();
                    float angle = acos(s1 * s2);
                    if(angle > angleThreshold)
                    {
                        osg::Vec3 tmpP;
                        osg::Vec3 tmpP2;
                        intersectLines((*itl)->startPos,(*itr)->startPos,s1,s2,tmpP,tmpP2);
                        float d = (*itl)->getMinimalDistance((*itr), tmpP);
                        if (d > 0)
                        {
                            positions->at(numIntersections) = tmpP;
                            p += tmpP;
                            numIntersections++;
                        }
                    }
                }
            }
        }
    }
    else
    {
        positions = new osg::Vec3Array(left.size() * right.size());
        for (itl = left.begin(); itl != left.end(); itl++)
        {
            osg::Vec3 s1 = (*itl)->startVelocity;
            for (itr = right.begin(); itr != right.end(); itr++)
            {

                osg::Vec3 s2 = (*itr)->startVelocity;
                osg::Vec3 tmpP;
                osg::Vec3 tmpP2;
                intersectLines((*itl)->startPos,(*itr)->startPos,s1,s2,tmpP,tmpP2);
                float d = (*itl)->getMinimalDistance((*itr), tmpP);
                if (d > 0)
                {
                    positions->at(numIntersections) = tmpP;
                    p += tmpP;
                    numIntersections++;
                }
            }
        }
    }
    if(numIntersections >0)
    {
    p /= numIntersections;
    standardDeviation = 0;
    for (int i = 0; i < numIntersections; i++)
    {
        standardDeviation += (p - positions->at(i)).length2();
    }
    standardDeviation = sqrt(standardDeviation / (numIntersections - 1));
    sphere->setRadius(standardDeviation);
    sphereDrawable->dirtyDisplayList();
    sphereTrans->setMatrix(osg::Matrix::translate(p));
    Origin = p;
    char buf[1000];
    sprintf(buf, "Origin: %f %f %f, deviation=%f, numIntersections:%d", p[0], p[1], p[2], (float)standardDeviation,numIntersections);
    //fprintf(stderr, "%s\n",buf);
    originLabel->setLabel(buf);
    }
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
    delete rhoLabel;
    delete rhoEdit;
    delete viscosityLabel;
    delete viscosityEdit;
    delete stLabel;
    delete stEdit;
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
    recalcVelocities = true;
    geode = new osg::Geode();
    //geode->setName(object_name);
    bpa->trajectoriesGroup->addChild(geode);
    bpa->trajectories.push_back(this);
    correctVelocity = false;
    D = 0.0;
    W = 0.0;
    mtl = new osg::Material;
    mtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
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
    kappa = bpa->velocity->getValue();
    Rho = bpa->rhoEdit->getValue();
    viscosity = bpa->viscosityEdit->getValue();
    surfacetension = bpa->stEdit->getValue();

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
}

double Trajectory::getX(double v)
{
    double sina = sin(alpha);
    double We = (Rho / surfacetension) * D * v * v; // 1055.0 *D *v*v/0.06
    double Re = (Rho / viscosity) * D * v; // 1055*D*v/0.005
    double P = We / pow(Re, 2.0 / 5.0);
    double sqrtp = sqrt(P);
    double powsin = pow(sina, 4.0 / 5.0);
    return ((pow(Re * sina, 1.0 / 5.0) * sqrtp * powsin / (1.24 + (sqrtp * powsin)))) - (W / D);
}

double Trajectory::getDX(double v)
{
    double vsina = v * sin(alpha);
    double We = (Rho / surfacetension) * D * v * v; // 1055.0 *D *v*v/0.06
    double Re = (Rho / viscosity) * D * v; // 1055*D*v/0.005
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
    velos = new osg::Vec3Array;
    colors = new osg::Vec4Array;

    primitives = new osg::DrawArrays(osg::PrimitiveSet::LINE_STRIP, 0, 2);
    recalc();
    geom->addPrimitiveSet(primitives);

    geom->setStateSet(lineState);

    geode->removeDrawables(0, 1);
    geode->addDrawable(geom);
}

float Trajectory::distance(osg::Vec3 &p,osg::Vec3 &p0,osg::Vec3 &p1)
    // computes distance between p and the line between p0 and p1
    //returns < -100000 if p is behind p0 and > 100000 if p is further than p1
{
    osg::Vec3 a = p1 - p0;
    osg::Vec3 b = p - p0;
    float la = a.length();
    float sprod = (a*b)/la;
    if(sprod < 0)
        return -100000 + sprod;
    if(sprod > 1)
        return -100000+sprod;
    return((a ^ b).length()/la);
}

float Trajectory::distance(Trajectory *t,int gi,int git)
    // computes distance between p and the line between p0 and p1this trajectory at index gi and t at git
    //returns 100000 if gi or git are out of range;
{
    if((gi > vert->size()-2)||(git > t->vert->size()-2))
        return 100000;
    osg::Vec3 tmpP;
    osg::Vec3 tmpP2;
    BPA::intersectLines(vert->at(gi),t->vert->at(git),vert->at(gi+1)-vert->at(gi),t->vert->at(git+1)-t->vert->at(git),tmpP,tmpP2);
    return (tmpP-tmpP2).length2();

}
void Trajectory::setStartVelocity(BPA *bpa, float vel)
{
    if (recalcVelocities)
    { // get an initial estimate of the origin with the velocity set at the wall point of the trajectory
        startVelocity *= bpa->velocity->getValue();
        recalc();
    }
    if(recalcVeloDiff)
    {
        // now we can compute the velocity difference between the origin and the wall point.
        computeVeloDiff(bpa->Origin);

        startVelocity.normalize();
    }
    if((vel - veloDiff) < 0)
    {
        vel = veloDiff + 0.01;
    }
    startVelocity *= vel - veloDiff;
}
void Trajectory::computeVeloDiff(osg::Vec3 &origin)
{
    float oldDist = 1000000;
    float currentDist=0;
    float startVelo=0;
    veloDiff = 0.0;
    if(vert->size()>0)
    {
        startVelo = (*velos)[0].length();
        for (int i = 0; i < vert->size(); i++)
        {
            currentDist = (vert->at(i) - origin).length();
            if (oldDist < currentDist)
            {
                veloDiff = velos->at(i - 1).length() - startVelo;
                recalcVeloDiff = false;
                return;
            }
            oldDist = currentDist;
        }
    }
    fprintf(stderr, "origin velocity not found\n");
}
float Trajectory::getMinimalDistance(Trajectory *t, osg::Vec3 &p1) // p1 is a start estimate and also returns the intersection point
{
    float minDist = 100000;

    if (vert->size() > 0 && t->vert->size() > 0)
    {
        int gi1=0,gi2=0;
        float len2 = (p1 - vert->at(0)).length();
        float l2=0;
        for (size_t i = 1; i < vert->size(); i++)
        {
            l2+= (vert->at(i) - vert->at(i-1)).length();
            if(len2 < l2)
            {
                gi1 = i-1;
                break;
            }

        }
        l2=0;
        for (size_t i = 1; i < t->vert->size(); i++)
        {
            l2+= (t->vert->at(i) - t->vert->at(i-1)).length();
            if(len2 < l2)
            {
                gi2 = i-1;
                break;
            }

        }
        float lastDistance = 1000000000.0;
        //float dist = distance(vert->at(gi1),t->vert->at(gi2),t->vert->at(gi2+1));

        float d[7];
        d[0] = distance(t,gi1  ,gi2  );
        d[1] = distance(t,gi1+1,gi2  );
        d[2] = distance(t,gi1+1,gi2+1);
        d[3] = distance(t,gi1  ,gi2+1);
        d[4] = distance(t,gi1-1,gi2  );
        d[5] = distance(t,gi1-1,gi2-1);
        d[6] = distance(t,gi1  ,gi2-1);
        while(lastDistance > minDist)
        {
            lastDistance = minDist;
            minDist = 100000;
            int mini = -1;
            for(int m=0;m<7;m++)
            {
              
                if(d[m]<minDist)
                {
                    minDist = d[m];
                    mini = m;
                }
            }
            if(mini == 0) // we are done, we have the closest pair
            {
                osg::Vec3 tmpP;
                osg::Vec3 tmpP2;
                BPA::intersectLines(vert->at(gi1),t->vert->at(gi2),vert->at(gi1+1)-vert->at(gi1),t->vert->at(gi2+1)-t->vert->at(gi2),tmpP,tmpP2);
                p1 = (tmpP + tmpP2)/2.0;
                return sqrt(minDist);
            }
            else if(mini == 1)
            {
                gi1++;
                d[0]=d[1];
                d[3] = d[2];
                d[4] = d[0];
                d[5] = d[6];
                d[1] = distance(t,gi1+1,gi2  );
                d[2] = distance(t,gi1+1,gi2+1);
                d[6] = distance(t,gi1  ,gi2-1);
            }
            else if(mini == 2)
            {
                gi1++; gi2++;
                d[6] = d[0];
                d[5] = d[0];
                d[0] = d[2];
                d[4] = d[3];
                d[1] = distance(t,gi1+1,gi2  );
                d[2] = distance(t,gi1+1,gi2+1);
                d[3] = distance(t,gi1  ,gi2+1);
            }
            else if(mini == 3)
            {
                gi2++;
                d[6] = d[0];
                d[0] = d[3];
                d[1] = d[2];
                d[5] = d[4];
                d[2] = distance(t,gi1+1,gi2+1);
                d[3] = distance(t,gi1  ,gi2+1);
                d[4] = distance(t,gi1-1,gi2  );
            }
            else if(mini == 4)
            {
                gi1--;
                d[1] = d[0];
                d[0] = d[4];
                d[2] = d[3];
                d[6] = d[5];
                d[3] = distance(t,gi1  ,gi2+1);
                d[4] = distance(t,gi1-1,gi2  );
                d[5] = distance(t,gi1-1,gi2-1);
            }
            else if(mini == 5)
            {
                gi1--; gi2--;
                d[2] = d[0];
                d[0] = d[5];
                d[1] = d[6];
                d[3] = d[4];
                d[4] = distance(t,gi1-1,gi2  );
                d[5] = distance(t,gi1-1,gi2-1);
                d[6] = distance(t,gi1  ,gi2-1);
            }
            else if(mini == 6)
            {
                gi2--;
                d[3] = d[0];
                d[0] = d[6];
                d[2] = d[1];
                d[4] = d[5];
                d[1] = distance(t,gi1+1,gi2  );
                d[5] = distance(t,gi1-1,gi2-1);
                d[6] = distance(t,gi1  ,gi2-1);
            }
        }
    }
    return -1;
}

float Trajectory::getMinimalDistanceSlow(Trajectory *t, osg::Vec3 &p1) 
{
    float minDist = 1000;
    int minI;
    int minN;

    if (vert->size() > 0 && t->vert->size() > 0)
    {
        for (size_t i = 0; i < vert->size(); i++)
        {
            for (size_t n = 0; n < t->vert->size(); n++)
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
    return -1;
}

void Trajectory::recalc()
{
    // set up geometry
    vert->clear();
    if (recalcVelocities)
    {
        velos->clear();
    }
    colors->clear();

    float dt = 0.01 / startVelocity.length();
    bool ignoreUpward = BPAPlugin::plugin->ignoreUpward->getState();
    if (startVelocity[2] > 0 && ignoreUpward)
    {
        primitives->setCount(0);
        geom->setVertexArray(vert);
        geom->setColorArray(colors);
        geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
        return;
    }
    double vMax = sqrt(12 / ((1.292 / surfacetension) * D));
    float v = startVelocity.length();
    if (v >= vMax)
    {
        colors->push_back(osg::Vec4(1, 0, 0, 1));
        if (recalcVelocities)
        {
            velos->push_back(startVelocity);
        }
    }
    else
    {
        colors->push_back(osg::Vec4(bpa->lineColor->getRed(), bpa->lineColor->getGreen(), bpa->lineColor->getBlue(), bpa->lineColor->getAlpha()));
        if (recalcVelocities)
        {
            velos->push_back(startVelocity);
        }
    }

    osg::Vec3 a;
    a.set(0, 0, -9.81);
    float len = 0;
    vert->push_back(startPos);
    osg::Vec3 pos = startPos;
    osg::Vec3 vel = startVelocity;
    bool res = BPAPlugin::plugin->airResistance->getState();
    while (len < length && pos[2] >= bpa->floorHeight)
    {
        float v = vel.length();
        if (res)
        {
            //vel += ((vel * ((0.5 * 0.48 * 1.292 * v) / ((2.0 / 3.0) * D * Rho))) + a) * dt;
            vel += ((vel * ((3.0 * 0.48 * 1.292 * v) / (4.0 * D * Rho))) + a) * dt;
        }
        else
        {
            vel += a * dt;
        }

        v = vel.length();
        if (v >= vMax)
        {
            v = vMax;
            vel.normalize();
            vel *= v;
            colors->push_back(osg::Vec4(1, 0, 0, 1));
        }
        else
        {
            colors->push_back(osg::Vec4(bpa->lineColor->getRed(), bpa->lineColor->getGreen(), bpa->lineColor->getBlue(), bpa->lineColor->getAlpha()));
        }
        pos = pos + vel * dt;
        len += (vel * dt).length();
        vert->push_back(pos);
        if (recalcVelocities)
        {
            velos->push_back(vel);
        }
    }
    //double We = 21.533333333333333333333333333333*D*velo*velo;// 1055.0 *D *v*v/0.06
    primitives->setCount(vert->size());
	vert->dirty();
	colors->dirty();

    geom->setVertexArray(vert);
    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    recalcVelocities = false;
}

BPAPlugin::BPAPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
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

    ignoreUpward = new coTUIToggleButton("Ignore upward", BPATab->getID(), true);
    ignoreUpward->setEventListener(this);
    ignoreUpward->setPos(2, 0);
    ignoreUpward->setState(false);
    
    allToAll = new coTUIToggleButton("All to all", BPATab->getID(), true);
    allToAll->setEventListener(this);
    allToAll->setPos(3, 0);
    allToAll->setState(true);
    
    angleLabel = new coTUILabel("Angle", BPATab->getID());
    angleLabel->setPos(4, 0);

    angleEdit = new coTUIEditFloatField("angleEdit", BPATab->getID(), 5);
    angleEdit->setEventListener(this);
    angleEdit->setPos(5, 0);
    angleEdit->setValue(20);


    writeButton = new coTUIButton("wrteResults", BPATab->getID());
    writeButton->setPos(6, 0);
    writeButton->setEventListener(this);
    originVeloEdit = new coTUIEditFloatField("originVelocity", BPATab->getID(), 19);
    originVeloEdit->setEventListener(this);
    originVeloEdit->setPos(7, 0);
    originVeloEdit->setValue(19);

    for (int index = 0; index < NUM_HANDLERS; index++)
        coVRFileManager::instance()->registerFileHandler(&handlers[index]);
}

// this is called if the plugin is removed at runtime
BPAPlugin::~BPAPlugin()
{
    fprintf(stderr, "BPAPlugin::~BPAPlugin\n");
    for (int index = 0; index < NUM_HANDLERS; index++)
        coVRFileManager::instance()->unregisterFileHandler(&handlers[index]);

    for (auto it = bpa_list.begin(); it != bpa_list.end(); it++)
    {
        delete (*it);
    }
    bpa_map.clear();
    bpa_list.clear();
    delete airResistance;
    delete OriginComputationType;
    delete ignoreUpward;
    delete BPATab;
    cover->getObjectsRoot()->removeChild(BPAGroup);
}

void BPAPlugin::tabletPressEvent(coTUIElement * tUIItem)
{
    if (tUIItem == writeButton)
    {
        char fileName[100];
        sprintf(fileName, "res%d.csv", (int)originVeloEdit->getValue());
        FILE* res = fopen(fileName, "w");
        if (res != NULL)
        {
            fprintf(res, "Origin positions and origin velocity\n");
            for (auto it = bpa_list.begin(); it != bpa_list.end(); it++)
            {
                fprintf(res, "%s;%f;%f;%f;%f\n", (*it)->filenameLabel->getName().c_str(),(*it)->originVelocity->getValue(), (*it)->Origin[0], (*it)->Origin[1], (*it)->Origin[2]);
            }
            fprintf(res, "wall impact velocities\n");
            fprintf(res, "file name;min Velocity;max Velocity;average Velocity;values\n");
            for (auto it = bpa_list.begin(); it != bpa_list.end(); it++)
            {
                fprintf(res, "%s", (*it)->filenameLabel->getName().c_str());
                float avgVelo = 0;
                float vmin = 1000000;
                float vmax = -1000000;
                for (auto tit = (*it)->trajectories.begin(); tit != (*it)->trajectories.end(); tit++)
                {
                    float velo = (*tit)->startVelocity.length();
                    avgVelo += velo;
                    if (velo < vmin)
                        vmin = velo;
                    if (velo > vmax)
                        vmax = velo;
                }
                avgVelo /= (*it)->trajectories.size();
                fprintf(res, ";%f", vmin);
                fprintf(res, ";%f", vmax);
                fprintf(res, ";%f", avgVelo);

                for (auto tit = (*it)->trajectories.begin(); tit != (*it)->trajectories.end(); tit++)
                {
                    float velo = (*tit)->startVelocity.length();
                    fprintf(res, ";%f", velo);
                }

                fprintf(res, "\n");
            }
            fclose(res);
        }
        else
        {
            fprintf(stderr, "could not open results.csv in current directory\n");
        }
    }
}

void BPAPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == originVeloEdit)
    {
        for (auto it = bpa_map.begin(); it != bpa_map.end(); it++)
        {
            it->second->setOriginVelocity(originVeloEdit->getValue());
        }
        
    }

    if (tUIItem == airResistance || tUIItem == ignoreUpward|| tUIItem == allToAll|| tUIItem == angleEdit)
    {
        for (auto it = bpa_map.begin(); it != bpa_map.end(); it++)
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
    if(BPAPlugin::plugin->OriginComputationType->getState())
    {
	calcIntersection();
    }
}

void BPA::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == minErrorButton)
    {
        for (auto it = trajectories.begin(); it != trajectories.end(); it++)
        {
            (*it)->length = length->getValue();
        }
        double minDev = 100000;
        double maxDev = -100000;
        float minv;
        std::list<pair<float, double>> deviations;
        for (float vel = 3.5; vel < 21.5; vel += 0.05)
        {
            for (auto it = trajectories.begin(); it != trajectories.end(); it++)
            {
                (*it)->setStartVelocity(this, vel);
            }
            float maxVdiff = 0;
            for (auto it = trajectories.begin(); it != trajectories.end(); it++)
            {
                if ((*it)->veloDiff > maxVdiff)
                {
                    maxVdiff = (*it)->veloDiff;
                }
            }
            originVelocity->setMin(velocity->getValue()-maxVdiff);
            originVelocity->setMax((velocity->getValue() - maxVdiff) + 20.0);
            recalc();
            deviations.push_back(std::make_pair(vel, standardDeviation));
            if (standardDeviation < minDev)
            {
                minDev = standardDeviation;
                minv = vel;
            }
            if (standardDeviation > maxDev)
            {
                maxDev = standardDeviation;
            }
        }

        for (auto it = trajectories.begin(); it != trajectories.end(); it++)
        {
            (*it)->startVelocity.normalize();
            (*it)->startVelocity *= minv;
        }
        velocity->setValue(minv);
        recalc();

        fprintf(stderr, "Origin: %f %f %f, standardDeviation=%f, minv=%f\n", Origin[0], Origin[1], Origin[2], (float)standardDeviation, minv);

    }
}
void BPA::setOriginVelocity(float ov)
{
    for (auto it = trajectories.begin(); it != trajectories.end(); it++)
    {
        (*it)->length = length->getValue();
        (*it)->startVelocity.normalize();
        originVelocity->setValue(ov);
        (*it)->setStartVelocity(this, ov);
    }
    float maxVdiff = 0;
    for (auto it = trajectories.begin(); it != trajectories.end(); it++)
    {
        if ((*it)->veloDiff > maxVdiff)
        {
            maxVdiff = (*it)->veloDiff;
        }
    }
    originVelocity->setMin(velocity->getValue() - maxVdiff);
    originVelocity->setMax((velocity->getValue() - maxVdiff) + 20.0);
    recalc();

    fprintf(stderr, "Origin: %f;%f;%f, standardDeviation=%f\n", Origin[0], Origin[1], Origin[2], (float)standardDeviation);
}
void BPA::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == length || tUIItem == velocity || tUIItem == originVelocity || tUIItem == rhoEdit || tUIItem == viscosityEdit || tUIItem == stEdit)
    {
        for (auto it = trajectories.begin(); it != trajectories.end(); it++)
        {
            (*it)->length = length->getValue();
            if (!(*it)->correctVelocity)
            {
                (*it)->startVelocity.normalize();
                if (tUIItem == velocity)
                {
                    (*it)->recalcVelocities = true;
                    (*it)->recalcVeloDiff = true;
                    (*it)->startVelocity *= velocity->getValue();
                }
                else
                {
                    (*it)->setStartVelocity(this,originVelocity->getValue());
                }
            }
            else
            {
                (*it)->computeVelocity();
                (*it)->startVelocity.normalize();
                (*it)->startVelocity *= (*it)->velo;
            }
        }
        float maxVdiff = 0;
        for (auto it = trajectories.begin(); it != trajectories.end(); it++)
        {
            if ((*it)->veloDiff > maxVdiff)
            {
                maxVdiff = (*it)->veloDiff;
            }
        }
        originVelocity->setMin(velocity->getValue() - maxVdiff);
        originVelocity->setMax((velocity->getValue() - maxVdiff) + 20.0);
        recalc();

        fprintf(stderr, "Origin: %f;%f;%f, standardDeviation=%f\n", Origin[0], Origin[1], Origin[2], (float)standardDeviation);
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
    floorHeight = -2;
    BPAPlugin::plugin->airResistance->setState(true);
    FILE *fp = fopen(filename.c_str(), "r");
    if (fp)
    {
        char buf[1000];
        char lastText=' ';
        char *useTexts = new char[1000];
        useTexts[0] = '\0';
        int numChars = 0;
        while (!feof(fp))
        {
            if (fgets(buf, 1000, fp) != NULL)
            {
                if (strncmp(buf, "USELINES", 8) == 0)
                {
                    strncpy(useTexts, buf + 9, 1000);
                    numChars = strlen(useTexts);
                }
                if (strncmp(buf, "TEXT", 4) == 0)
                {
                    if (fgets(buf, 1000, fp) != NULL)
                    {
                        if (fgets(buf, 1000, fp) != NULL)
                        {
                            lastText = buf[0];
                        }
                    }
                }
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
                            bool found = false;
                            for(int i=0;i<numChars;i++)
                            {
                                if (lastText == useTexts[i])
                                {
                                    found = true;
                                    break;
                                }
                            }
                            if(numChars == 0 || found )
                            {
                            Trajectory *t = new Trajectory(this);
                            t->startPos = v[1];
                            t->startVelocity = v[0] - v[1];
                            t->length = length->getValue();
                            t->startVelocity.normalize();
                            t->startVelocity *= velocity->getValue();
			    t->viscosity= 0.005;
			    t->surfacetension=0.06;
			    t->gamma = 0.0;
			    t->alpha = 0.0;    
			    t->W = 3.8 / 1000.0;
			    t->Vol = 0.11842;
			    t->kappa = 0.15;
			    t->Rho = 1055.0;
			    t->D = 2 * pow((3 * t->Vol / pow(10, 9) / (t->kappa * 4 * M_PI)), 1.0 / 3.0); // from dried volume (Vol) to D, kappa is drying ratio ~0.15
                            t->createGeometry();
                            }
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
    
    floorHeight = 0;
    BPAPlugin::plugin->OriginComputationType->setState(true);
    velocityLabel->setLabel("Kappa");
    velocity->setMin(0.05);
    velocity->setMax(0.3);
    velocity->setValue(0.15);
    bool flat=false;
    FILE *fp = fopen(filename.c_str(), "r");
    if (fp)
    {
        char buf[1000];
        while (!feof(fp))
        {
            if (fgets(buf, 1000, fp) != NULL)
            {
                if(buf[0] =='#')
                {
                    if(strstr(buf,"flat")!=NULL)
                    {
                        flat = true;
                    }
                }
                else
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
                    if(flat)
                    {
                        v.set(sin(a+M_PI_2), -sin(c) * cos(a+M_PI_2), -cos(c) * cos(a+M_PI_2));
                    }
                    else
                    {
                        v.set(sin(a), -sin(c) * cos(a), -cos(c) * cos(a));
                    }
                    osg::Matrix m;
                    m.makeRotate(b, osg::Vec3(0, 0, 1));
                    v = osg::Matrix::transform3x3(m, v);
                    v.normalize();
                    t->alpha = a;
                    t->gamma = c;
                    t->W = W / 1000.0;
                    t->Vol = Vol;
                    t->correctVelocity = true;
                    t->computeVelocity();
                    t->startVelocity = v * t->velo;
                    t->length = length->getValue();
                    t->createGeometry();
                }
            }
        }
        fclose(fp);
    }
}
void BPA::loadTxt(std::string filename)
{
    floorHeight = 0;
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
    auto it = bpa_map.find(std::string(name));

    if (it != bpa_map.end())
    {
        BPA *b = bpa_map[(char *)name];

        for (auto it = bpa_list.begin(); it != bpa_list.end(); it++)
        {
            bpa_list.erase(it);
            break;
        }
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
    bpa_list.push_back(b);
    return 0;
}

COVERPLUGIN(BPAPlugin)
