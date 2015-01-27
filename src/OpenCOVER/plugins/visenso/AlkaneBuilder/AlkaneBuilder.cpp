/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <osg/Geometry>
#include <osg/StateSet>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/Matrix>

#include "AtomBallInteractor.h"

#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRConfig.h>
#include <cover/coVRNavigationManager.h>
#include <cover/coHud.h>
#include <cover/coVRFileManager.h>

#include "Atom.h"
#include "Carbon.h"

#include <config/CoviseConfig.h>

#include "AlkaneBuilder.h"

#include "cover/coTranslator.h"

//constructor
AlkaneBuilder::AlkaneBuilder()
{
    if (opencover::cover->debugLevel(0))
        fprintf(stderr, "---AlkaneBuilder::AlkaneBuilder\n");

    currentAlkane_ = NULL;
    group_ = new osg::Group();
    group_->ref();
    opencover::cover->getObjectsRoot()->addChild(group_);

    size_ = opencover::cover->getSceneSize() / 50;
    //fprintf(stderr,"opencover::cover->getSceneSize()/50=%f\n", size_);
    size_ = covise::coCoviseConfig::getFloat("COVER.IconSize", size_);
    //fprintf(stderr,"COVER.IconSize=%f\n", size_);

    mode_ = false; //show alkane
    osg::Matrix m;

    // create atoms, position it left and right of the table
    for (int i = 0; i < MAX_CARBONS; i++)
    {
        std::vector<osg::Vec3> connections;
        connections.push_back(osg::Vec3(1, 1, -1));
        connections.push_back(osg::Vec3(-1, -1, -1));
        connections.push_back(osg::Vec3(-1, 1, 1));
        connections.push_back(osg::Vec3(1, -1, 1));
        char name[1024];
        sprintf(name, "Carbon_%d", i);
        //fprintf(stderr,"creating C atom %s\n", name );

        float w = opencover::coVRConfig::instance()->screens[0].hsize;
        float h = opencover::coVRConfig::instance()->screens[0].vsize;
        float x = 0.28 * w;
        float y = -0.3 * h * sin(M_PI * 0.25) + i * 0.6 * h / (MAX_CARBONS - 1) * sin((M_PI * 0.25));
        float z = -0.3 * h * sin(M_PI * 0.25) + i * 0.6 * h / (MAX_CARBONS - 1) * sin((M_PI * 0.25));
        m.makeTranslate(x, y, z);
        atoms_.push_back(new Carbon("C", name, m, size_, connections, osg::Vec4(0.3, 0.3, 0.3, 1)));
    }
    m.makeTranslate(HYDROGEN_POS_X, HYDROGEN_POS_Y, HYDROGEN_POS_Z);
    for (int i = 0; i < MAX_HYDROGENS; i++)
    {
        std::vector<osg::Vec3> connections;
        connections.push_back(osg::Vec3(0, 0, 1));
        char name[1024];
        sprintf(name, coTranslator::coTranslate("Hydrogen_%d").c_str(), i);
        //fprintf(stderr,"creating H atom %s\n", name );
        float w = opencover::coVRConfig::instance()->screens[0].hsize;
        float h = opencover::coVRConfig::instance()->screens[0].vsize;
        float x = -0.28 * w;
        float y = -0.3 * h * sin(M_PI * 0.25) + i * 0.6 * h / (MAX_HYDROGENS - 1) * sin((M_PI * 0.25));
        float z = -0.3 * h * sin(M_PI * 0.25) + i * 0.6 * h / (MAX_HYDROGENS - 1) * sin((M_PI * 0.25));
        m.makeTranslate(x, y, z);
        atoms_.push_back(new Atom("H", name, m, size_, connections, osg::Vec4(0.8, 0.8, 0.8, 1)));
    }

    makeDescription("-", "-", 0.5 * size_, osg::Vec3(DESCRIPTION_POS_X, DESCRIPTION_POS_Y, DESCRIPTION_POS_Z));

    createPlane();

    // create error panel
    hud_ = opencover::coHud::instance();
    forwardOk_ = true;
    //showHud_=0;
    hudTime_ = 0.0;

    if (opencover::cover->debugLevel(0))
        fprintf(stderr, "---AlkaneBuilder::AlkaneBuilder done\n");
}

// destructor
AlkaneBuilder::~AlkaneBuilder()
{
    if (opencover::cover->debugLevel(3))
        fprintf(stderr, "AlkaneBuilder::~AlkaneBuilder\n");
}

// set new element
void AlkaneBuilder::setAlkane(Alkane al)
{
    if (opencover::cover->debugLevel(0))
        fprintf(stderr, "\tAlkaneBuilder::setAlkane [%s]\n", al.name.c_str());
    if (al.name == "ERROR")
    {
        currentAlkane_ = NULL;
        updateDescription("-", "-");
        reset();
        forwardOk_ = true;
    }
    else
    {
        currentAlkane_ = new Alkane(al.formula, al.name, al.linear, al.carbons, al.hydrogens);
        updateDescription(al.name, al.formula);
        if (!mode_)
        {
            buildAlkane(currentAlkane_->carbons, currentAlkane_->hydrogens, currentAlkane_->linear);
            forwardOk_ = true;
        }
        else // first build the alkane manually
            forwardOk_ = false;
    }
}

// update called every frame
void AlkaneBuilder::update()
{
    if (opencover::cover->debugLevel(3))
        fprintf(stderr, "AlkaneBuilder::update\n");

    for (int i = 0; i < atoms_.size(); i++)
    {
        atoms_[i]->preFrame();

        // check if the running atom has free connections and is in connect position to other atoms
        for (int j = 0; j < atoms_.size(); j++)
        {
            if (!atoms_[i]->isIdle())
                atoms_[i]->checkNear(atoms_[j]);
        }

        if (atoms_[i]->wasStopped())
        {
            //fprintf(stderr,"--Atom %s wasStopped\n", atoms_[i]->atomBall_->getInteractorName());
            //if the running atom has fixed connections, we have to update them
            //check if the running atom has free connections and is in connect position to other atoms
            for (int j = 0; j < atoms_.size(); j++)
            {
                if (atoms_[i] != atoms_[j])
                {
                    if (atoms_[i]->checkNear(atoms_[j]))
                    {
                        atoms_[i]->snap(atoms_[i]->getMySnapAtomStick(), atoms_[j], atoms_[i]->getOtherSnapAtomStick());
                        //check();
                        break;
                    }
                    else
                    {
                        atoms_[i]->moveToPlane(plane_);
                    }
                }
            }
            check();
        }
    }

    if (hudTime_ > 2.5f)
    {
        hudTime_ = 0.0f;
        hud_->hide();
        fprintf(stderr, "AlkaneBuilder::update hiding hud\n");
    }
    else if (hudTime_ > 0.0f)
    {
        hudTime_ += opencover::cover->frameDuration();
        fprintf(stderr, "AlkaneBuilder::update hud_Time_=%f\n", hudTime_);
    }

    if (opencover::cover->debugLevel(3))
        fprintf(stderr, "AlkaneBuilder::update done\n");
}

void AlkaneBuilder::show(bool value)
{
    fprintf(stderr, "AlkaneBuilder::show %d\n", value);
    if (value)
    {
        if (!group_)
            fprintf(stderr, "group???\n");
        if (group_->getNumParents() == 0)
        {
            if (!opencover::cover->getObjectsRoot())
                fprintf(stderr, "!opencover::cover->getObjectsRoot()\n");
            opencover::cover->getObjectsRoot()->addChild(group_);
        }
    }
    else
    {
        if (group_->getNumParents())
        {
            opencover::cover->getObjectsRoot()->removeChild(group_);
        }
    }

    for (int i = 0; i < atoms_.size(); i++)
    {
        if (value)
        {
            // show geometry
            atoms_[i]->show(true);
            atoms_[i]->enableIntersection(true);
        }
        else
        {
            //hide geometry
            atoms_[i]->show(false);
            atoms_[i]->enableIntersection(false);
        }
    }
}

void
AlkaneBuilder::makeDescription(std::string name, std::string formula, float fontSize, osg::Vec3 pos)
{
    fprintf(stderr, "\tAlkaneBuilder::makeDescription\n");
    anweisungGeode_ = new osg::Geode();
    anweisungGeode_->ref();
    anweisungText_ = new osgText::Text();
    anweisungText_->setCharacterSize(fontSize);
    anweisungText_->setAlignment(osgText::Text::LEFT_TOP);
    anweisungText_->setAxisAlignment(osgText::Text::XZ_PLANE);
    anweisungText_->setFont(opencover::coVRFileManager::instance()->getFontFile(NULL));
    anweisungText_->setPosition(pos);

    char t[2000];
    if (mode_) // build mode
    {
        sprintf(t, coTranslator::coTranslate("Bauen Sie das Alkan: %s\n").c_str(), name.c_str());
    }
    else
        sprintf(t, coTranslator::coTranslate("Sie sehen das Alkan: %s\n").c_str(), name.c_str());

    osgText::String ot(std::string(t), osgText::String::ENCODING_UTF8);
    anweisungText_->setText(ot);
    anweisungGeode_->addDrawable(anweisungText_);

    statusGeode_ = new osg::Geode();
    statusGeode_->ref();
    statusText_ = new osgText::Text();
    statusText_->setCharacterSize(fontSize);
    statusText_->setAlignment(osgText::Text::LEFT_TOP);
    statusText_->setAxisAlignment(osgText::Text::XZ_PLANE);
    statusText_->setFont(opencover::coVRFileManager::instance()->getFontFile(NULL));
    pos[2] -= fontSize;
    statusText_->setPosition(pos);
    sprintf(t, " ");
    osgText::String ott(std::string(t), osgText::String::ENCODING_UTF8);
    statusText_->setText(ott);
    statusGeode_->addDrawable(statusText_);

    //group_->addChild(descrGeode);
    //opencover::cover->getScene()->addChild(descrGeode_);

    // Color
    osg::Vec4 textColor(1.0f, 1.0f, 1.0f, 1.0f);
    osg::Material *textMaterial = new osg::Material;
    textMaterial->setAmbient(osg::Material::FRONT_AND_BACK, textColor);
    textMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, textColor);
    textMaterial->setEmission(osg::Material::FRONT_AND_BACK, textColor);
    anweisungGeode_->getOrCreateStateSet()->setAttributeAndModes(textMaterial);
    statusGeode_->getOrCreateStateSet()->setAttributeAndModes(textMaterial);
}

void
AlkaneBuilder::updateDescription(std::string name, std::string formula)
{
    fprintf(stderr, "\tAlkaneBuilder::updateDescription %s\n", name.c_str());
    char t[2000];
    statusText_->setText(" ");
    if (mode_) // build mode
    {
        sprintf(t, coTranslator::coTranslate("Bauen Sie das Alkan: %s\n").c_str(), name.c_str());
    }
    else
    {
        sprintf(t, coTranslator::coTranslate("Sie sehen das Alkan: %s\n").c_str(), name.c_str());
    }
    osgText::String ot(std::string(t), osgText::String::ENCODING_UTF8);
    anweisungText_->setText(ot);
}

void
AlkaneBuilder::showInstructionText(bool show)
{
    fprintf(stderr, "\tAlkaneBuilder::showDescription value=%d\n", show);
    if (show)
    {
        if (anweisungGeode_->getNumParents() == 0)
            opencover::cover->getScene()->addChild(anweisungGeode_);
    }
    else
    {
        if (anweisungGeode_->getNumParents() > 0)
            opencover::cover->getScene()->removeChild(anweisungGeode_);
    }
}
void
AlkaneBuilder::showStatusText(bool show)
{
    fprintf(stderr, "\tAlkaneBuilder::showDescription value=%d\n", show);
    if (show)
    {
        if (statusGeode_->getNumParents() == 0)
            opencover::cover->getScene()->addChild(statusGeode_);
    }
    else
    {
        if (statusGeode_->getNumParents() > 0)
            opencover::cover->getScene()->removeChild(statusGeode_);
    }
}
void AlkaneBuilder::check()
{

    //fprintf(stderr,"--AlkaneBuilder::check\n");
    carbons_.clear();
    // liste anlegen mit verbundenen kohlenstoffen
    for (int i = 0; i < MAX_CARBONS; i++)
    {
        //fprintf(stderr,"checking if carbon %d is completely connected\n",i);
        if (((Carbon *)atoms_[i])->allConnectionsConnected(NULL))
        {
            //fprintf(stderr,"carbon %d is completely connected\n", i);
            carbons_.push_back((Carbon *)atoms_[i]);
        }
    }
    // jetzt enthaelt die Liste nur noch Kohlenstoffe, die Alkane sein können
    int nh = 0;
    for (int i = 0; i < carbons_.size(); i++)
    {
        nh += carbons_[i]->getNumAtoms("H");
    }
    //fprintf(stderr,"have %d completely connected carbons with %d hydrogens\n", carbons_.size(), nh);

    // jetzt rausfinden, ob das überhaupt ein Alkane ist oder ein unfertiges Teil
    Alkane alkane = AlkaneDatabase::Instance()->findByAtoms(carbons_.size(), nh);
    if (alkane.name != "ERROR")
    {
        // wenn was gebaut wurde, ist es linear?
        bool isLinear = false;
        // methan, ethan, propan gibt es nur linear
        if (carbons_.size() <= 3)
        {
            isLinear = true;
        }
        else
        {
            for (int i = 0; i < carbons_.size(); i++)
            {
                if (((Carbon *)carbons_[i])->isLinearAlkane(alkane.carbons))
                {
                    isLinear = true;
                    break;
                }
            }
        }
        // jetzt nochmals den genauen Namen rausfinden
        alkane = AlkaneDatabase::Instance()->findByAtoms(carbons_.size(), nh, isLinear);
    }

    // hier kommt der eigentliche check
    if (currentAlkane_) // soll überhaupt was gecheckt werden?
    {
        updateDescription(currentAlkane_->name, currentAlkane_->formula);
        if (alkane.name == "ERROR")
        {
            forwardOk_ = false;
            showStatusText(false);
            statusText_->setText(" ");
        }
        else if (alkane.name == currentAlkane_->name)
        {
            string t;
            t = coTranslator::coTranslate("Status: Sie haben das Alkan ");
            t.append(alkane.name);
            t.append(coTranslator::coTranslate(" zusammengebaut!"));
            osgText::String ot(std::string(t), osgText::String::ENCODING_UTF8);
            statusText_->setText(ot);
            forwardOk_ = true;
            showStatusText(true);
        }
    }
    else
    {
        updateDescription("-", "-");
        showStatusText(false);
        forwardOk_ = true;
    }

    fprintf(stderr, "--AlkaneBuilder::check done forwardOk_=%d\n", forwardOk_);
}

void
AlkaneBuilder::buildAlkaneStart(Carbon *c, Atom *h0, Atom *h1, Atom *h2)
{
    //fprintf(stderr,"AlkaneBuilder::buildAlkaneStart\n");
    osg::Matrix m;
    m.makeTranslate(-100, 0, 0);
    c->updateTransform(m);

    h0->snap(h0->atomSticks_[0], c, c->atomSticks_[0]);
    h1->snap(h1->atomSticks_[0], c, c->atomSticks_[1]);
    h2->snap(h2->atomSticks_[0], c, c->atomSticks_[2]);
}
void
AlkaneBuilder::buildAlkaneEnd13(Carbon *c, Carbon *prevC, Atom *h0, Atom *h1, Atom *h2)
{
    //fprintf(stderr,"AlkaneBuilder::buildAlkaneEnd13\n");
    c->snap(c->atomSticks_[1], prevC, prevC->atomSticks_[3]);
    h0->snap(h0->atomSticks_[0], c, c->atomSticks_[0]);
    h1->snap(h1->atomSticks_[0], c, c->atomSticks_[2]);
    h2->snap(h2->atomSticks_[0], c, c->atomSticks_[3]);
}
void
AlkaneBuilder::buildAlkaneEnd20(Carbon *c, Carbon *prevC, Atom *h0, Atom *h1, Atom *h2)
{
    //fprintf(stderr,"AlkaneBuilder::buildAlkaneEnd20\n");
    c->snap(c->atomSticks_[2], prevC, prevC->atomSticks_[0]);
    h0->snap(h0->atomSticks_[0], c, c->atomSticks_[0]);
    h1->snap(h1->atomSticks_[0], c, c->atomSticks_[1]);
    h2->snap(h2->atomSticks_[0], c, c->atomSticks_[3]);
}
void
AlkaneBuilder::buildAlkaneMiddle13(Carbon *c, Carbon *prevC, Atom *h0, Atom *h1)
{
    //fprintf(stderr,"AlkaneBuilder::buildAlkaneMiddle\n");
    c->snap(c->atomSticks_[1], prevC, prevC->atomSticks_[3]);
    h0->snap(h0->atomSticks_[0], c, c->atomSticks_[2]);
    h1->snap(h1->atomSticks_[0], c, c->atomSticks_[3]);
    //fprintf(stderr,"AlkaneBuilder::buildAlkaneMiddle done\n");
}
void
AlkaneBuilder::buildAlkaneMiddle20(Carbon *c, Carbon *prevC, Atom *h0, Atom *h1)
{
    //fprintf(stderr,"AlkaneBuilder::buildAlkaneMiddle\n");
    c->snap(c->atomSticks_[2], prevC, prevC->atomSticks_[0]);
    h0->snap(h0->atomSticks_[0], c, c->atomSticks_[0]);
    h1->snap(h1->atomSticks_[0], c, c->atomSticks_[1]);
    //fprintf(stderr,"AlkaneBuilder::buildAlkaneMiddle done\n");
}

void
AlkaneBuilder::buildAlkane(int nc, int nh, bool linear)
{
    forwardOk_ = true;
    fprintf(stderr, "\tAlkaneBuilder::buildAlkane nc=%d nh=%d linear=%d\n", nc, nh, linear);
    reset();

    if (currentAlkane_)
    {
        if (linear)
        {
            if (nc == 1 && nh == 4) //ethan
            {
                buildAlkaneStart((Carbon *)atoms_[0], atoms_[MAX_CARBONS + 0], atoms_[MAX_CARBONS + 1], atoms_[MAX_CARBONS + 2]);
                atoms_[MAX_CARBONS + 3]->snap(atoms_[MAX_CARBONS + 3]->atomSticks_[0], atoms_[0], atoms_[0]->atomSticks_[3]);
            }
            else
            {
                int ic = 0;
                int ih = 0;
                buildAlkaneStart((Carbon *)atoms_[ic], atoms_[MAX_CARBONS + ih], atoms_[MAX_CARBONS + (ih + 1)], atoms_[MAX_CARBONS + ih + 2]);
                ic++;
                ih += 3;
                for (int i = 1; i < nc - 1; i++)
                {
                    if (i % 2 == 1) //ungerades C
                    {
                        buildAlkaneMiddle13((Carbon *)atoms_[ic], (Carbon *)atoms_[ic - 1], atoms_[MAX_CARBONS + ih], atoms_[MAX_CARBONS + ih + 1]);
                        ic++;
                        ih += 2;
                    }
                    else
                    {
                        buildAlkaneMiddle20((Carbon *)atoms_[ic], (Carbon *)atoms_[ic - 1], atoms_[MAX_CARBONS + ih], atoms_[MAX_CARBONS + ih + 1]);
                        ic++;
                        ih += 2;
                    }
                }
                if ((nc - 1) % 2 == 1) // ungerade
                    buildAlkaneEnd13((Carbon *)atoms_[ic], (Carbon *)atoms_[ic - 1], atoms_[MAX_CARBONS + ih], atoms_[MAX_CARBONS + ih + 1], atoms_[MAX_CARBONS + ih + 2]);
                else
                    buildAlkaneEnd20((Carbon *)atoms_[ic], (Carbon *)atoms_[ic - 1], atoms_[MAX_CARBONS + ih], atoms_[MAX_CARBONS + ih + 1], atoms_[MAX_CARBONS + ih + 2]);
            }
        }
        else
        {
            osgText::String ot(coTranslator::coTranslate("Die automatische Darstellung von i-Alkanen ist nicht implementiert"), osgText::String::ENCODING_UTF8);
            anweisungText_->setText(ot);
        }
    }
}
void
AlkaneBuilder::createPlane()
{
    osg::ref_ptr<osg::Vec3Array> coordLine, coordLine1;
    osg::ref_ptr<osg::Vec3Array> coordPoly, coordPoly1;
    osg::ref_ptr<osg::Vec3Array> polyNormal;
    osg::ref_ptr<osg::Geometry> geometryLine, geometryLine1; ///< Geometry object
    osg::ref_ptr<osg::Geometry> geometryPoly, geometryPoly1; ///< Geometry object
    osg::Vec4Array *colorLine, *colorPoly;

    float w = opencover::coVRConfig::instance()->screens[0].hsize;
    float h = opencover::coVRConfig::instance()->screens[0].vsize;
    float dx = 0.28 * w;
    float dy = 0.3 * h * sin(0.25 * M_PI);
    float dz = 0.3 * h * sin(0.25 * M_PI);

    colorLine = new osg::Vec4Array(1);
    (*colorLine)[0].set(1, 1, 1, 1);

    colorPoly = new osg::Vec4Array(1);
    (*colorPoly)[0].set(0.7, 0.4, 0.3, 0.5);

    coordLine = new osg::Vec3Array(5);
    (*coordLine)[0].set(0.3 * w, 0, -0.3 * h);
    (*coordLine)[1].set(0.3 * w, 0, 0.3 * h);
    (*coordLine)[2].set(-0.3 * w, 0, 0.3 * h);
    (*coordLine)[3].set(-0.3 * w, 0, -0.3 * h);
    (*coordLine)[4].set(0.3 * w, 0, -0.3 * h);

    coordPoly = new osg::Vec3Array(4);
    (*coordPoly)[0].set(0.3 * w, 0, -0.3 * h);
    (*coordPoly)[1].set(0.3 * w, 0, 0.3 * h);
    (*coordPoly)[2].set(-0.3 * w, 0, 0.3 * h);
    (*coordPoly)[3].set(-0.3 * w, 0, -0.3 * h);

    coordPoly1 = new osg::Vec3Array(4);
    (*coordPoly1)[0].set(dx, dy, -dz);
    (*coordPoly1)[1].set(dx, dy, dz);
    (*coordPoly1)[2].set(-dx, dy, dz);
    (*coordPoly1)[3].set(-dx, dy, -dz);

    polyNormal = new osg::Vec3Array(1);
    (*polyNormal)[0].set(0, -1, 0);

    geometryLine = new osg::Geometry();
    geometryLine->setColorArray(colorLine);
    geometryLine->setColorBinding(osg::Geometry::BIND_OVERALL);
    geometryLine->setVertexArray(coordLine.get());
    geometryLine->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_STRIP, 0, 5));
    geometryLine->setUseDisplayList(false);
    geometryLine->setStateSet(opencover::VRSceneGraph::instance()->loadDefaultGeostate());

    geometryPoly = new osg::Geometry();
    geometryPoly->setColorArray(colorPoly);
    geometryPoly->setColorBinding(osg::Geometry::BIND_OVERALL);
    geometryPoly->setVertexArray(coordPoly.get());
    geometryPoly->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POLYGON, 0, 4));
    geometryPoly->setUseDisplayList(false);
    osg::StateSet *stateSet = opencover::VRSceneGraph::instance()->loadTransparentGeostate();
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    geometryPoly->setStateSet(stateSet);

    planeTransformNode_ = new osg::MatrixTransform();
    planeTransformNode_->ref();
    osg::Matrix r;
    r.makeRotate(-0.25 * M_PI, 1, 0, 0);
    planeTransformNode_->setMatrix(r);

    planeNode_ = new osg::Geode();
    planeNode_->addDrawable(geometryLine.get());
    planeNode_->addDrawable(geometryPoly.get());
    planeNode_->setName("AlkaneBuilderPlane");
    planeNode_->setNodeMask(planeNode_->getNodeMask() & (~opencover::Isect::Intersection) & (~opencover::Isect::Pick));

    //group_->addChild(planeTransformNode_.get());
    planeTransformNode_->addChild(planeNode_.get());

    plane_ = new opencover::coPlane(osg::Vec3(0, -1, 1), osg::Vec3(0, 0, 0));
}

void
AlkaneBuilder::reset()
{
    fprintf(stderr, "\tAlkaneBuilder::reset\n");
    for (int i = 0; i < atoms_.size(); i++)
        atoms_[i]->reset();
}

void
AlkaneBuilder::setModeBuild(bool m)
{
    fprintf(stderr, "\tAlkaneBuilder::setModeBuild mode=%d\n", m);
    mode_ = m;
    reset();

    if (currentAlkane_)
        updateDescription(currentAlkane_->name, currentAlkane_->formula);
    else
        updateDescription(coTranslator::coTranslate("Üben Sie die Bedienung"), coTranslator::coTranslate("-"));

    /*
   if (mode_) // manually build
     showStatusText(true);
   else
      showStatusText(false);
   */
}
void
AlkaneBuilder::enableIntersection(bool enable)
{

    fprintf(stderr, "\tAlkaneBuilder::enableIntersection value=%d\n", enable);
    for (int i = 0; i < (MAX_CARBONS + MAX_HYDROGENS); i++)
    {
        if (enable)
            atoms_[i]->enableIntersection(true);
        else
            atoms_[i]->enableIntersection(false);
    }
}

void
AlkaneBuilder::showPlane(bool show)
{
    fprintf(stderr, "\tAlkaneBuilder::showPlane %d\n", show);
    if (show)
    {
        if (planeTransformNode_->getNumParents() == 0)
        {
            group_->addChild(planeTransformNode_);
        }
    }
    else //hide
    {
        if (planeTransformNode_->getNumParents() > 0)
        {
            group_->removeChild(planeTransformNode_);
        }
    }

    /*
   for (int i = 0; i < MAX_CARBONS; i++) 
   {
      osg::Matrix m;
      m.makeTranslate(CARBON_POS_X, CARBON_POS_Y, CARBON_POS_Z);
      atoms_[i]->updateTransform(m);
   }
   for (int i = MAX_CARBONS; i < MAX_CARBONS+MAX_HYDROGENS; i++) 
   {
      osg::Matrix m;
      m.makeTranslate(HYDROGEN_POS_X, HYDROGEN_POS_Y, HYDROGEN_POS_Z);
      atoms_[i]->updateTransform(m);
   }
   */
}

void AlkaneBuilder::showErrorPanel()
{
    //fprintf(stderr,"Die Aufgabe ist falsch geloest!\n");
    hud_->setText1(coTranslator::coTranslate("Das Alkan ist noch nicht korrekt! \nVersuchen Sie es weiter.").c_str());
    hud_->show();
    hud_->redraw();
    //if (showHud_==0)
    //	showHud_++;
    if (hudTime_ == 0.0f)
        hudTime_ = 0.001f;
}
