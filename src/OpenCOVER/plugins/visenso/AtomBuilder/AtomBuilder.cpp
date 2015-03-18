/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <osg/Geometry>
#include <osg/StateSet>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/Matrix>

#include <config/CoviseConfig.h>

#include "cover/coTranslator.h"

#include "AtomBuilder.h"
#include "ElementaryParticleInteractor.h"
#include "ElectronInteractor.h"
#include "NucleonInteractor.h"
#include "ElementDatabase.h"
#include "CheckButton.h"

#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coInteractor.h>
#include <PluginUtil/coVR2DTransInteractor.h>
#include <cover/coInteractor.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRFileManager.h>
#include <cover/coHud.h>

using namespace opencover;
using namespace covise;
//constructor
AtomBuilder::AtomBuilder()
{
    if (cover->debugLevel(0))
        fprintf(stderr, "AtomBuilder::AtomBuilder\n");

    group_ = new osg::Group();
    group_->ref();

    // initialize variables
    currentElement_ = NULL;

    // read positions and sizes from config
    osg::Vec3 protonStartPos, neutronStartPos, posElectron;
    sizeNucleon_ = cover->getSceneSize() / 50;
    sizeNucleon_ = coCoviseConfig::getFloat("COVER.IconSize", sizeNucleon_);

    px_ = coCoviseConfig::getFloat("x", "COVER.Plugin.AtomBuilder.ProtonsPosition", PROTONS_POS_X);
    py_ = coCoviseConfig::getFloat("y", "COVER.Plugin.AtomBuilder.ProtonsPosition", PROTONS_POS_Y);
    pz_ = coCoviseConfig::getFloat("z", "COVER.Plugin.AtomBuilder.ProtonsPosition", PROTONS_POS_Z);
    nx_ = coCoviseConfig::getFloat("x", "COVER.Plugin.AtomBuilder.NeutronsPosition", NEUTRONS_POS_X);
    ny_ = coCoviseConfig::getFloat("y", "COVER.Plugin.AtomBuilder.NeutronsPosition", NEUTRONS_POS_Y);
    nz_ = coCoviseConfig::getFloat("z", "COVER.Plugin.AtomBuilder.NeutronsPosition", NEUTRONS_POS_Z);
    ex_ = coCoviseConfig::getFloat("x", "COVER.Plugin.AtomBuilder.ElectronsPosition", ELECTRONS_POS_X);
    ey_ = coCoviseConfig::getFloat("y", "COVER.Plugin.AtomBuilder.ElectronsPosition", ELECTRONS_POS_Y);
    ez_ = coCoviseConfig::getFloat("z", "COVER.Plugin.AtomBuilder.ElectronsPosition", ELECTRONS_POS_Z);
    atomNucleusRadius_ = coCoviseConfig::getFloat("n", "COVER.Plugin.AtomBuilder.AtomSizes", ATOM_NUCLEUS_RADIUS);
    atomKShellRadius_ = coCoviseConfig::getFloat("k", "COVER.Plugin.AtomBuilder.AtomSizes", ATOM_KSHELL_RADIUS);
    atomLShellRadius_ = coCoviseConfig::getFloat("l", "COVER.Plugin.AtomBuilder.AtomSizes", ATOM_LSHELL_RADIUS);
    atomMShellRadius_ = coCoviseConfig::getFloat("m", "COVER.Plugin.AtomBuilder.AtomSizes", ATOM_MSHELL_RADIUS);

    // make description
    //makeText("OOOOOOOOOOOOOO", 0.5*sizeNucleon_, osg::Vec3(px_, py_, pz_));
    nN_ = nP_ = nKE_ = nLE_ = nME_ = 0;
    makeDescription(coTranslator::coTranslate("Eingebaute Teilchen:"), nN_, nP_, nKE_ + nLE_ + nME_, 0.5 * sizeNucleon_, osg::Vec3(px_ - 0.6 * sizeNucleon_, py_, pz_ + 4 * sizeNucleon_));

    // positions in nucleus
    float dn = 1.5 * sizeNucleon_;
    protonPositions_.push_back(osg::Vec3(0, 0, 0)); //p 0

    neutronPositions_.push_back(osg::Vec3(1, 0, 0) * dn); // n0
    protonPositions_.push_back(osg::Vec3(-1, 0, 0) * dn); // p1
    neutronPositions_.push_back(osg::Vec3(0, 0, 1) * dn); // n1
    protonPositions_.push_back(osg::Vec3(0, 0, -1) * dn); // p2
    neutronPositions_.push_back(osg::Vec3(-0.7071, 0, 0.7071) * dn); //n2
    protonPositions_.push_back(osg::Vec3(0.7071, 0, 0.7071) * dn); // p3
    neutronPositions_.push_back(osg::Vec3(-0.7071, 0, -0.7071) * dn); //n3
    protonPositions_.push_back(osg::Vec3(0.7071, 0, -0.7071) * dn); //p4
    //nucleusPositions_.push_back(osg::Vec3(0,-1, 0)*dn);//n4
    neutronPositions_.push_back(osg::Vec3(0.35, -0.7071, 0.35) * dn); //p5
    protonPositions_.push_back(osg::Vec3(-0.35, -0.7071, 0.35) * dn); //n5
    neutronPositions_.push_back(osg::Vec3(-0.35, -0.7071, -0.35) * dn); //p6
    protonPositions_.push_back(osg::Vec3(0.35, -0.7071, -0.35) * dn); //n6
    //nucleusPositions_.push_back(osg::Vec3(0,1, 0)*dn);
    neutronPositions_.push_back(osg::Vec3(0.35, 0.7071, 0.35) * dn); //p7
    protonPositions_.push_back(osg::Vec3(0.35, 0.7071, -0.35) * dn); //n7
    neutronPositions_.push_back(osg::Vec3(-0.35, 0.7071, 0.35) * dn); //p 8
    protonPositions_.push_back(osg::Vec3(-0.35, 0.7071, -0.35) * dn); //n8

    dn = 3 * sizeNucleon_;
    neutronPositions_.push_back(osg::Vec3(1, 0, 0) * dn); // n9
    protonPositions_.push_back(osg::Vec3(-1, 0, 0) * dn); // p9
    neutronPositions_.push_back(osg::Vec3(0, 0, 1) * dn); // n10
    protonPositions_.push_back(osg::Vec3(0, 0, -1) * dn); // p10
    neutronPositions_.push_back(osg::Vec3(-0.7071, 0, 0.7071) * dn); //n11
    protonPositions_.push_back(osg::Vec3(0.7071, 0, 0.7071) * dn); // p11
    neutronPositions_.push_back(osg::Vec3(-0.7071, 0, -0.7071) * dn); //n12
    protonPositions_.push_back(osg::Vec3(0.7071, 0, -0.7071) * dn); //p12
    //nucleusPositions_.push_back(osg::Vec3(0,-1, 0)*dn);//n13
    neutronPositions_.push_back(osg::Vec3(0.35, -0.7071, 0.35) * dn); //p13
    protonPositions_.push_back(osg::Vec3(-0.35, -0.7071, 0.35) * dn); //n14
    neutronPositions_.push_back(osg::Vec3(-0.35, -0.7071, -0.35) * dn); //p14
    protonPositions_.push_back(osg::Vec3(0.35, -0.7071, -0.35) * dn); //n15
    //nucleusPositions_.push_back(osg::Vec3(0,1, 0)*dn);
    neutronPositions_.push_back(osg::Vec3(0.35, 0.7071, 0.35) * dn); //p15
    protonPositions_.push_back(osg::Vec3(0.35, 0.7071, -0.35) * dn); //n16
    neutronPositions_.push_back(osg::Vec3(-0.35, 0.7071, 0.35) * dn); //p16
    protonPositions_.push_back(osg::Vec3(-0.35, 0.7071, -0.35) * dn); //n17

    normal_.set(0, -1, 0);

    osg::Node *pg = coVRFileManager::instance()->loadIcon("atombaukasten/glas");
    if (pg)
    {
        pg->setNodeMask(pg->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));

        // add matrix transform to modify file geometry
        osg::MatrixTransform *t = new osg::MatrixTransform();
        group_->addChild(t);
        osg::Matrix m1, m2, m3, m;
        m1.makeScale(0.5, 0.5, 0.1);
        m2.makeRotate(45.0, osg::Vec3(0, 0, 1));
        m.mult(m1, m2);
        m3.makeTranslate(px_, py_, pz_ - 0.5 * sizeNucleon_);
        m.mult(m, m3);
        t->setMatrix(m);
        t->addChild(pg);
    }
    float d = 1.5 * sizeNucleon_;
    makeText(coTranslator::coTranslate("Protonen"), 0.5 * sizeNucleon_, osg::Vec3(px_ - 0.6 * sizeNucleon_, py_, pz_ - 0.6 * sizeNucleon_));
    for (int i = 0; i < MAX_PROTONS; i++)
    {
        //protonStartPos.set(px_+(float)i*d, py_, pz_);
        protonStartPos.set(px_, py_, pz_);
        protons_.push_back(new NucleonInteractor(protonStartPos, normal_, sizeNucleon_, "atombaukasten/p_plus", protonPositions_[i], atomNucleusRadius_));
        protons_[protons_.size() - 1]->disableIntersection();
        protons_[protons_.size() - 1]->hide();
        d -= 0.1 * d;
    }

    osg::Node *ng = coVRFileManager::instance()->loadIcon("atombaukasten/glas");
    if (ng)
    {
        ng->setNodeMask(ng->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));

        // add matrix transform to modify file geometry
        osg::MatrixTransform *t = new osg::MatrixTransform();
        group_->addChild(t);
        osg::Matrix m1, m2, m3, m;
        m1.makeScale(0.5, 0.5, 0.1);
        m2.makeRotate(45.0, osg::Vec3(0, 0, 1));
        m.mult(m1, m2);
        m3.makeTranslate(nx_, ny_, nz_ - 0.5 * sizeNucleon_);
        m.mult(m, m3);
        t->setMatrix(m);
        t->addChild(ng);
    }
    d = 1.5 * sizeNucleon_;
    makeText(coTranslator::coTranslate("Neutronen"), 0.5 * sizeNucleon_, osg::Vec3(nx_ - 0.6 * sizeNucleon_, ny_, nz_ - 0.6 * sizeNucleon_));
    for (int i = 0; i < MAX_NEUTRONS; i++)
    {
        //neutronStartPos.set(nx_+(float)i*d, ny_, nz_);
        neutronStartPos.set(nx_, ny_, nz_);
        neutrons_.push_back(new NucleonInteractor(neutronStartPos, normal_, sizeNucleon_, "atombaukasten/n", neutronPositions_[i], atomNucleusRadius_));
        neutrons_[neutrons_.size() - 1]->disableIntersection();
        neutrons_[neutrons_.size() - 1]->hide();
        d -= 0.05 * d;
    }

    osg::Node *eg = coVRFileManager::instance()->loadIcon("atombaukasten/glas");
    if (eg)
    {
        eg->setNodeMask(eg->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));

        // add matrix transform to modify file geometry
        osg::MatrixTransform *t = new osg::MatrixTransform();
        group_->addChild(t);
        osg::Matrix m1, m2, m3, m;
        m1.makeScale(0.5, 0.5, 0.1);
        m2.makeRotate(45.0, osg::Vec3(0, 0, 1));
        m.mult(m1, m2);
        m3.makeTranslate(ex_, ey_, ez_ - 0.5 * sizeNucleon_);
        m.mult(m, m3);
        t->setMatrix(m);
        t->addChild(eg);
    }
    d = 1.5 * 0.75 * sizeNucleon_;
    makeText(coTranslator::coTranslate("Elektronen"), 0.5 * sizeNucleon_, osg::Vec3(ex_ - 0.6 * sizeNucleon_, ey_, ez_ - 0.6 * sizeNucleon_));
    for (int i = 0; i < MAX_ELECTRONS; i++)
    {
        //posElectron.set(ex_+(float)i*d, ey_, ez_);
        posElectron.set(ex_, ey_, ez_);
        electrons_.push_back(new ElectronInteractor(posElectron, normal_, 0.75 * sizeNucleon_, "atombaukasten/e_minus", atomNucleusRadius_, atomKShellRadius_, atomLShellRadius_, atomMShellRadius_));
        electrons_[electrons_.size() - 1]->disableIntersection();
        electrons_[electrons_.size() - 1]->hide();
        d -= 0.05 * d;
    }
    // Visualize atom nucleus and shells

    osg::Node *node = coVRFileManager::instance()->loadIcon("atombaukasten/gesamt");
    if (node)
    {
        node->setNodeMask(node->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));

        // add matrix transform to modify file geometry
        osg::MatrixTransform *t = new osg::MatrixTransform();
        group_->addChild(t);
        osg::Matrix m1, m2, m3, m;
        t->addChild(node);
    }

    // create error panel
    hud_ = coHud::instance();
    forwardOk_ = true;
    //showHud_=0;
    hudTime_ = 0.0;

    oldElectronsInKShellSize_ = 0;
    oldElectronsInLShellSize_ = 0;
    oldElectronsInMShellSize_ = 0;

    checkButton_ = new CheckButton(osg::Vec3(nx_ - 1.5 * sizeNucleon_, ny_, pz_ + 3 * sizeNucleon_), 1.5 * sizeNucleon_);
    checkButton_->setVisible(false);
    checkButton_->disableIntersection();
    //showCheck_=0;
    checkTime_ = 0.0f;
}

// destructor
AtomBuilder::~AtomBuilder()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "AtomBuilder::~AtomBuilder\n");
}

// set new element
void AtomBuilder::setElement(Element el)
{
    if (cover->debugLevel(0))
        fprintf(stderr, "AtomBuilder::setElement [%s]\n", el.name.c_str());

    if (currentElement_)
        delete currentElement_;
    if (el.name == "ERROR")
    {
        currentElement_ = NULL;
        checkButton_->setVisible(false);
        checkButton_->disableIntersection();
    }
    else
    {
        currentElement_ = new Element(el.number, el.symbol, el.name, el.protons, el.neutrons, el.electrons[0], el.electrons[1], el.electrons[2], el.electrons[3]);
        resetParticles();
        checkButton_->setVisible(true);
        checkButton_->enableIntersection();
        checkButton_->setButtonState(BUTTON_STATE_CHECK);
    }
}

// update called every frame
void AtomBuilder::update()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "AtomBuilder::update\n");

    nN_ = nP_ = nKE_ = nLE_ = nME_ = 0;
    // if nucleon was stopped, move animated to final position
    for (size_t i = 0; i < protons_.size(); i++)
    {
        protons_[i]->preFrame();
        if ((protons_[i]->wasStopped()) || (protons_[i]->isIdle()))
        {
            if (((NucleonInteractor *)protons_[i])->insideNucleus())
                nP_++;
        }
        if (protons_[i]->wasStopped())
        {
            if (((NucleonInteractor *)protons_[i])->insideNucleus())
            {

                protons_[i]->startAnimation(((NucleonInteractor *)protons_[i])->getFinalPosition());

                /* rearrange the ouside protons
            int J=0;
            float d=1.5*sizeNucleon_;
            for (int j=0; j<protons_.size(); j++)
            {

               if ( protons_[j]!=protons_[i]) 
               {
                  fprintf(stderr,"Proton %d not the released one\n", j);
                  if (   ! ((NucleonInteractor*)protons_[j])->insideNucleus() )
                  {      
                     fprintf(stderr,"Proton %d outside  ... animate [%f %f %f]\n", j, px_+(float)J*d, py_, pz_);
                     osg::Vec3 pos(px_+(float)J*d, py_, pz_);
                     protons_[j]->startAnimation(pos);
                     protons_[j]->setInitialPosition(pos);
                     d-=0.1*d;
                     J++;
                  }
               }

            }
            */
            }
            else
                protons_[i]->startAnimation(protons_[i]->getInitialPosition());
        }
    }
    for (size_t i = 0; i < neutrons_.size(); i++)
    {
        neutrons_[i]->preFrame();
        if ((neutrons_[i]->wasStopped()) || (neutrons_[i]->isIdle()))
        {
            if (((NucleonInteractor *)neutrons_[i])->insideNucleus())
                nN_++;
        }
        if (neutrons_[i]->wasStopped())
        {
            if (((NucleonInteractor *)neutrons_[i])->insideNucleus())
            {
                neutrons_[i]->startAnimation(((NucleonInteractor *)neutrons_[i])->getFinalPosition());
            }
            else
                neutrons_[i]->startAnimation(neutrons_[i]->getInitialPosition());
        }
    }

    // if electron is released inside a shell
    // move it to the nearest position on radius, compute the angle of this electron
    // insert it in the list sorted by angle
    // compute delta angle and angle of first electron in list
    // rearrange the other electrons

    // clear lists
    electronsInKShell_.clear();
    electronsInLShell_.clear();
    electronsInMShell_.clear();

    bool stoppedInside = false;
    bool stoppedOutside = false;

    float releasedAngle;
    ElectronInteractor *releasedElectron = NULL;
    int releasedPos;

    if (cover->debugLevel(5))
        fprintf(stderr, "--sorting electrons\n");
    for (int i = 0; i < ssize_t(electrons_.size()); i++)
    {
        if (cover->debugLevel(5))
            fprintf(stderr, "--electron %d\n", i);
        electrons_[i]->preFrame();

        //int pos=0;
        float a = ((ElectronInteractor *)electrons_[i])->getAngle();
        if (a > 2 * M_PI)
            a = a - 2 * M_PI;

        // append inside electrons to list, insert the just stopped electron at begin
        if (((ElectronInteractor *)electrons_[i])->insideKShell())
        {
            if (cover->debugLevel(5))
                fprintf(stderr, "---electron %d in kshell\n", i);

            if (electrons_[i]->isIdle() || electrons_[i]->wasStopped())
            {
                nKE_++;
                // list empty
                if (electronsInKShell_.size() == 0)
                {
                    electronsInKShell_.push_back(electrons_[i]);
                    //fprintf(stderr,"adding electron %d in empty list at pos %d with angle %f\n", i, pos, ((ElectronInteractor*)electrons_[i])->getAngle()*180/M_PI);
                }
                // list not empty
                else
                {
                    std::list<ElementaryParticleInteractor *>::iterator it;
                    for (it = electronsInKShell_.begin(); it != electronsInKShell_.end(); it++)
                    {
                        if (a < ((ElectronInteractor *)(*it))->getAngle())
                        {
                            electronsInKShell_.insert(it, electrons_[i]);
                            //fprintf(stderr,"inserting electron %d at pos %d with angle %f\n", i, pos, ((ElectronInteractor*)electrons_[i])->getAngle()*180/M_PI);
                            break;
                        }
                        //else
                        //   pos++;
                    }
                    if (it == electronsInKShell_.end())
                    {
                        electronsInKShell_.push_back(electrons_[i]);
                        //fprintf(stderr,"adding electron %d at pos %d with angle %f\n", i, pos, ((ElectronInteractor*)electrons_[i])->getAngle()*180/M_PI);
                        //pos=electronsInKShell_.size()-1;
                    }
                }
            }
            if (electrons_[i]->wasStopped())
            {
                if (cover->debugLevel(5))
                    fprintf(stderr, "---electron %d stopped in kshell\n", i);
                releasedElectron = (ElectronInteractor *)electrons_[i];
                releasedAngle = a;
                stoppedInside = true;

                // snap to radius
                float r = atomNucleusRadius_ + 0.5 * (atomKShellRadius_ - atomNucleusRadius_);
                osg::Vec3 vector = electrons_[i]->getPosition();
                vector.normalize();
                electrons_[i]->startAnimation(vector * r);
            }
        }

        else if (((ElectronInteractor *)electrons_[i])->insideLShell())
        {
            if (cover->debugLevel(5))
                fprintf(stderr, "---electron %d in Lshell\n", i);

            if (electrons_[i]->isIdle() || electrons_[i]->wasStopped())
            {
                nLE_++;
                // list empty
                if (electronsInLShell_.size() == 0)
                {
                    electronsInLShell_.push_back(electrons_[i]);
                    //fprintf(stderr,"adding electron %d in empty list at pos %d with angle %f\n", i, pos, ((ElectronInteractor*)electrons_[i])->getAngle()*180/M_PI);
                }
                // list not empty
                else
                {
                    std::list<ElementaryParticleInteractor *>::iterator it;
                    for (it = electronsInLShell_.begin(); it != electronsInLShell_.end(); it++)
                    {
                        if (a < ((ElectronInteractor *)(*it))->getAngle())
                        {
                            electronsInLShell_.insert(it, electrons_[i]);
                            //fprintf(stderr,"inserting electron %d at pos %d with angle %f\n", i, pos, ((ElectronInteractor*)electrons_[i])->getAngle()*180/M_PI);
                            break;
                        }
                        //else
                        //   pos++;
                    }
                    if (it == electronsInLShell_.end())
                    {
                        electronsInLShell_.push_back(electrons_[i]);
                        //fprintf(stderr,"adding electron %d at pos %d with angle %f\n", i, pos, ((ElectronInteractor*)electrons_[i])->getAngle()*180/M_PI);
                        //pos=electronsInLShell_.size()-1;
                    }
                }
            }
            if (electrons_[i]->wasStopped())
            {
                if (cover->debugLevel(5))
                    fprintf(stderr, "---electron %d stopped in lshell\n", i);
                releasedElectron = (ElectronInteractor *)electrons_[i];
                releasedAngle = a;
                stoppedInside = true;

                // snap to radius
                float r = atomKShellRadius_ + 0.5 * (atomLShellRadius_ - atomKShellRadius_);
                osg::Vec3 vector = electrons_[i]->getPosition();
                vector.normalize();
                electrons_[i]->startAnimation(vector * r);
            }
        }

        else if (((ElectronInteractor *)electrons_[i])->insideMShell())
        {

            if (cover->debugLevel(5))
                fprintf(stderr, "---electron %d in mshell\n", i);

            if (electrons_[i]->isIdle() || electrons_[i]->wasStopped())
            {
                nME_++;
                // list empty
                if (electronsInMShell_.size() == 0)
                {
                    electronsInMShell_.push_back(electrons_[i]);
                    //fprintf(stderr,"adding electron %d in empty list at pos %d with angle %f\n", i, pos, ((ElectronInteractor*)electrons_[i])->getAngle()*180/M_PI);
                }
                // list not empty
                else
                {
                    std::list<ElementaryParticleInteractor *>::iterator it;
                    for (it = electronsInMShell_.begin(); it != electronsInMShell_.end(); it++)
                    {
                        if (a < ((ElectronInteractor *)(*it))->getAngle())
                        {
                            electronsInMShell_.insert(it, electrons_[i]);
                            //fprintf(stderr,"inserting electron %d at pos %d with angle %f\n", i, pos, ((ElectronInteractor*)electrons_[i])->getAngle()*180/M_PI);
                            break;
                        }
                        //else
                        //   pos++;
                    }
                    if (it == electronsInMShell_.end())
                    {
                        electronsInMShell_.push_back(electrons_[i]);
                        //fprintf(stderr,"adding electron %d at pos %d with angle %f\n", i, pos, ((ElectronInteractor*)electrons_[i])->getAngle()*180/M_PI);
                        //pos=electronsInLShell_.size()-1;
                    }
                }
            }
            if (electrons_[i]->wasStopped())
            {
                if (cover->debugLevel(5))
                    fprintf(stderr, "---electron %d stopped in mshell\n", i);
                releasedElectron = (ElectronInteractor *)electrons_[i];
                releasedAngle = a;
                stoppedInside = true;

                // snap to radius
                float r = atomLShellRadius_ + 0.5 * (atomMShellRadius_ - atomLShellRadius_);
                osg::Vec3 vector = electrons_[i]->getPosition();
                vector.normalize();
                electrons_[i]->startAnimation(vector * r);
            }
        }

        else // not inside a shell
        {
            if (electrons_[i]->isIdle() || electrons_[i]->wasStopped())
            {
                protonsOutside_.push_back(electrons_[i]);
            }
            if (electrons_[i]->wasStopped())
            {
                if (cover->debugLevel(5))
                    fprintf(stderr, "---electron %d stopped outside\n", i);

                stoppedInside = false;
                stoppedOutside = true;
                electrons_[i]->startAnimation(electrons_[i]->getInitialPosition());
                releasedElectron = (ElectronInteractor *)electrons_[i];
            }
        }
    }

    if (stoppedInside || stoppedOutside)
    {
        //fprintf(stderr,"stoppedIn/Outside releasedAngle=%f electronsInKShell_.size()=%d \n", releasedAngle*180/M_PI, electronsInKShell_.size());

        std::list<ElementaryParticleInteractor *>::iterator it;
        int p;
        float da, a, r;

        // rearrange kshell
        if (((oldElectronsInKShellSize_ != ssize_t(electronsInKShell_.size())) && (electronsInKShell_.size() != 0)) || (releasedElectron && releasedElectron->insideKShell()))
        {
            if (cover->debugLevel(5))
                fprintf(stderr, "---rearrange khell\n");

            p = 0;
            for (it = electronsInKShell_.begin(); it != electronsInKShell_.end(); ++it)
            {
                if (((ElectronInteractor *)(*it)) == releasedElectron)
                {

                    break;
                }
                else
                    p++;
            }
            releasedPos = p;
            //fprintf(stderr,"electronsInKShell_=%d releasedpos=%d releasedangle %f\n", (int)electronsInKShell_.size(), p, releasedAngle*180/M_PI);
            //sleep(2);

            // compute delta angle for distributing electron
            da = 2 * M_PI / electronsInKShell_.size();

            if (releasedPos == ssize_t(electronsInKShell_.size()))
            {
                a = ((ElectronInteractor *)(*electronsInKShell_.begin()))->getAngle();
            }
            else
            {
                a = releasedAngle - da * releasedPos;
            }
            //fprintf(stderr,"---releasedAngle:%f - da:%f * releasedPos:%d = starta:%f\n",releasedAngle*180/M_PI, da*180/M_PI, releasedPos, a*180/M_PI);

            if (a > 2 * M_PI)
                a = a - 2 * M_PI;
            r = atomNucleusRadius_ + 0.5 * (atomKShellRadius_ - atomNucleusRadius_);
            for (it = electronsInKShell_.begin(); it != electronsInKShell_.end(); it++)
            {
                //(*it)->startRadiusAnimation(osg::Vec3(r*sin(a),0, r*cos(a)));
                (*it)->startRadiusAnimation(r, ((ElectronInteractor *)(*it))->getAngle(), a);
                a += da;
                if (a > 2 * M_PI)
                    a = a - 2 * M_PI;
            }
        }

        if (cover->debugLevel(5))
            fprintf(stderr, "---rearrange khell done\n");

        // rearrange lshell
        if (((oldElectronsInLShellSize_ != ssize_t(electronsInLShell_.size())) && (electronsInLShell_.size() != 0)) || (releasedElectron && releasedElectron->insideLShell()))
        {
            if (cover->debugLevel(5))
                fprintf(stderr, "---rearrange lshell\n");

            p = 0;
            for (it = electronsInLShell_.begin(); it != electronsInLShell_.end(); ++it)
            {
                //fprintf(stderr,"angle=%f\n", ( (ElectronInteractor*)(*it) )->getAngle()*180/M_PI);
                if (((ElectronInteractor *)(*it)) == releasedElectron)
                {

                    break;
                }
                else
                    p++;
            }
            releasedPos = p;

            //fprintf(stderr,"---releasedpos=%d releasedangle %f\n", p, releasedAngle*180/M_PI);
            //sleep(2);

            // compute delta angle for distributing electrons
            da = 2 * M_PI / electronsInLShell_.size();

            if (releasedPos == ssize_t(electronsInLShell_.size()))
            {
                a = ((ElectronInteractor *)(*electronsInLShell_.begin()))->getAngle();
            }
            else
            {
                a = releasedAngle - da * releasedPos;
            }
            if (a > 2 * M_PI)
                a = a - 2 * M_PI;

            //fprintf(stderr,"da=%f starta = %f\n", da*180/M_PI, a*180/M_PI);
            r = atomKShellRadius_ + 0.5 * (atomLShellRadius_ - atomKShellRadius_);
            for (it = electronsInLShell_.begin(); it != electronsInLShell_.end(); it++)
            {
                //(*it)->startRadiusAnimation(osg::Vec3(r*sin(a),0, r*cos(a)));
                (*it)->startRadiusAnimation(r, ((ElectronInteractor *)(*it))->getAngle(), a);
                a += da;
                if (a > 2 * M_PI)
                    a = a - 2 * M_PI;
            }
        }

        // rearrange mshell
        if (((oldElectronsInMShellSize_ != ssize_t(electronsInMShell_.size())) && (electronsInMShell_.size() != 0)) || (releasedElectron && releasedElectron->insideMShell()))
        {
            if (cover->debugLevel(5))
                fprintf(stderr, "---rearrange mshell\n");

            p = 0;
            for (it = electronsInMShell_.begin(); it != electronsInMShell_.end(); ++it)
            {
                //fprintf(stderr,"angle=%f\n", ( (ElectronInteractor*)(*it) )->getAngle()*180/M_PI);
                if (((ElectronInteractor *)(*it)) == releasedElectron)
                {

                    break;
                }
                else
                    p++;
            }
            releasedPos = p;
            //fprintf(stderr,"---releasedpos=%d releasedangle %f\n", p, releasedAngle*180/M_PI);
            //sleep(2);

            // compute delta angle for distributing electrons
            da = 2 * M_PI / electronsInMShell_.size();

            if (releasedPos == ssize_t(electronsInMShell_.size()))
            {
                a = ((ElectronInteractor *)(*electronsInMShell_.begin()))->getAngle();
            }
            else
            {
                a = releasedAngle - da * releasedPos;
            }
            if (a > 2 * M_PI)
                a = a - 2 * M_PI;
            //fprintf(stderr,"da=%f starta = %f\n", da*180/M_PI, a*180/M_PI);
            r = atomLShellRadius_ + 0.5 * (atomMShellRadius_ - atomLShellRadius_);
            for (it = electronsInMShell_.begin(); it != electronsInMShell_.end(); it++)
            {
                //(*it)->startRadiusAnimation(osg::Vec3(r*sin(a),0, r*cos(a)));
                (*it)->startRadiusAnimation(r, ((ElectronInteractor *)(*it))->getAngle(), a);
                a += da;
                if (a > 2 * M_PI)
                    a = a - 2 * M_PI;
            }
        }

        // rearrange Outside electrons

        oldElectronsInKShellSize_ = electronsInKShell_.size();
        oldElectronsInLShellSize_ = electronsInLShell_.size();
        oldElectronsInMShellSize_ = electronsInMShell_.size();

        if (cover->debugLevel(5))
            fprintf(stderr, "--electron done\n");
    }
    if (cover->debugLevel(5))
        fprintf(stderr, "--sorting electrons done\n");

    bool protonsOk, neutronsOk, electronsKShellOk, electronsLShellOk, electronsMShellOk;
    protonsOk = neutronsOk = electronsKShellOk = electronsLShellOk = electronsMShellOk = false;
    if (currentElement_)
    {
        if (cover->debugLevel(5))
            fprintf(stderr, "---check\n");

        // check if protons, netrons and electrons are correct for the current element
        protonsOk = check(protons_, currentElement_->protons, atomNucleusRadius_);
        neutronsOk = check(neutrons_, currentElement_->neutrons, atomNucleusRadius_);
        electronsKShellOk = check(electrons_, currentElement_->electrons[0], atomNucleusRadius_, atomKShellRadius_);
        electronsLShellOk = check(electrons_, currentElement_->electrons[1], atomKShellRadius_, atomLShellRadius_);
        electronsMShellOk = check(electrons_, currentElement_->electrons[2], atomLShellRadius_, atomMShellRadius_);
        //fprintf(stderr,"%s: protonsOk=%d neutronsOk=%d, electronsKShellOk=%d, electronsLShellOk=%d, electronsMShellOk=%d\n", currentElement_->name.c_str(), protonsOk, neutronsOk, electronsKShellOk, electronsLShellOk, electronsMShellOk);

        // if atom is build correctly allow presentationstepforward
        if (protonsOk && neutronsOk && electronsKShellOk && electronsLShellOk && electronsMShellOk)
        {
            forwardOk_ = true;
        }
        else
            forwardOk_ = false;
    }
    //fprintf(stderr,"AtomBuilder::update forwardOk_=%d\n", forwardOk_);

    // check if hud panel has to be hided
    //if (showHud_>40)
    //{
    //	showHud_ = 0;
    //	hud_->hide();
    //}
    //else if (showHud_>0)
    //	showHud_++;

    if (hudTime_ > 2.5f)
    {
        hudTime_ = 0.0f;
        hud_->hide();
    }
    else if (hudTime_ > 0.0f)
    {
        hudTime_ += cover->frameDuration();
    }

    //fprintf(stderr,"NP=%d, NN=%d, NE=%d\n", nP_, nN_, nE_);
    if (cover->debugLevel(5))
        fprintf(stderr, "---preFrame done\n");

    if (checkButton_->wasClicked())
    {
        //showCheck_++;
        if (checkTime_ == 0.0f)
            checkTime_ = 0.001f;
        checkButton_->disableIntersection();

        if (forwardOk_)
            checkButton_->setButtonState(BUTTON_STATE_OK);
        else
            checkButton_->setButtonState(BUTTON_STATE_NOTOK);
    }

    if (checkTime_ > 2.5f)
    {
        checkTime_ = 0.0f;
        checkButton_->setButtonState(BUTTON_STATE_CHECK);
        checkButton_->enableIntersection();
    }
    else if (checkTime_ > 0.0f)
    {
        checkTime_ += cover->frameDuration();
    }

    updateDescription(coTranslator::coTranslate("Eingebaute Teilchen:"), nP_, nN_, nKE_ + nLE_ + nME_);
}

// check if inside radius
bool
AtomBuilder::check(std::vector<ElementaryParticleInteractor *> particles, int numParticles, float radius)
{

    bool correct = false;
    int numParticlesInside = 0;
    //fprintf(stderr,"AtomBuilder::check radius=%f\n", radius);
    for (size_t i = 0; i < particles.size(); i++)
    {
        osg::Vec3 pos = particles[i]->getPosition();
        float d = pos.length();

        if (d <= radius)
        {
            //particles[i]->setInsidePosition();
            numParticlesInside++;
            //fprintf(stderr,"AtomBuilder::check nucleus numParticlesInside=%d/%d \n", numParticlesInside, numParticles);
        }
    }
    if (numParticles == numParticlesInside)
    {
        correct = true;
    }
    return correct;
}

// check if between radius1 and radius 2
bool
AtomBuilder::check(std::vector<ElementaryParticleInteractor *> particles, int numParticles, float radius1, float radius2)
{

    bool correct = false;
    int numParticlesInside = 0;
    for (size_t i = 0; i < particles.size(); i++)
    {
        osg::Vec3 pos = particles[i]->getPosition();
        float d = pos.length();

        if ((d >= radius1) && (d <= radius2))
        {
            numParticlesInside++;
        }
    }
    if (numParticles == numParticlesInside)
    {
        correct = true;
    }
    return correct;
}

void AtomBuilder::showErrorPanel()
{
    //fprintf(stderr,"Die Aufgabe ist falsch gel�st!\n");
    hud_->setText1(coTranslator::coTranslate("Der Atombau ist noch nicht korrekt! \nVersuchen Sie es weiter.").c_str());
    hud_->show();
    hud_->redraw();
    //if (showHud_==0)
    //	showHud_++;
    if (hudTime_ == 0.0f)
        hudTime_ = 0.001f;
}
void AtomBuilder::show(bool value)
{
    if (value)
    {
        if (!group_)
            fprintf(stderr, "group???\n");

        if (group_->getNumParents() == 0)
        {
            //fprintf(stderr,"num parents ==0");
            if (!cover->getObjectsRoot())
                fprintf(stderr, "!cover->getObjectsRoot()\n");
            cover->getObjectsRoot()->addChild(group_);
        }
    }
    else
    {
        if (group_->getNumParents())
        {
            cover->getObjectsRoot()->removeChild(group_);
        }
    }

    for (size_t i = 0; i < protons_.size(); i++)
    {
        if (value)
        {
            // show geometry
            protons_[i]->show();
            protons_[i]->enableIntersection();
        }
        else
        {
            //hide geometry
            protons_[i]->hide();
            protons_[i]->disableIntersection();
        }
    }
    for (size_t i = 0; i < neutrons_.size(); i++)
    {
        if (value)
        {
            // show geometry
            neutrons_[i]->show();
            neutrons_[i]->enableIntersection();
        }
        else
        {
            //hide geometry
            neutrons_[i]->hide();
            neutrons_[i]->disableIntersection();
        }
    }
    for (size_t i = 0; i < electrons_.size(); i++)
    {
        if (value)
        {
            // show geometry
            electrons_[i]->show();
            electrons_[i]->enableIntersection();
        }
        else
        {
            //hide geometry
            electrons_[i]->hide();
            electrons_[i]->disableIntersection();
        }
    }
}

void
AtomBuilder::resetParticles()
{
    for (size_t i = 0; i < protons_.size(); i++)
    {
        protons_[i]->resetPosition();
    }
    for (size_t i = 0; i < neutrons_.size(); i++)
    {
        neutrons_[i]->resetPosition();
    }
    for (size_t i = 0; i < electrons_.size(); i++)
    {
        electrons_[i]->resetPosition();
    }
}
void
AtomBuilder::makeText(const std::string &t, float s, osg::Vec3 p)
{
    // Text
    osgText::Text *textDrawable = new osgText::Text();
    osg::Geode *textGeode = new osg::Geode();
    textGeode->addDrawable(textDrawable);
    group_->addChild(textGeode);

    textDrawable->setCharacterSize(s);
    textDrawable->setAlignment(osgText::Text::LEFT_TOP);
    textDrawable->setAxisAlignment(osgText::Text::XZ_PLANE);
    textDrawable->setFont(coVRFileManager::instance()->getFontFile(NULL));
    osgText::String ot(std::string(t), osgText::String::ENCODING_UTF8);
    textDrawable->setText(ot);
    textDrawable->setPosition(p);

    // Color
    osg::Vec4 textColor(1.0f, 1.0f, 1.0f, 1.0f);
    osg::Material *textMaterial = new osg::Material;
    textMaterial->setAmbient(osg::Material::FRONT_AND_BACK, textColor);
    textMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, textColor);
    textMaterial->setEmission(osg::Material::FRONT_AND_BACK, textColor);
    textGeode->getOrCreateStateSet()->setAttributeAndModes(textMaterial);
}

void
AtomBuilder::makeDescription(const std::string &heading, int numP, int numN, int numE, float s, osg::Vec3 p)
{
    osg::Geode *descrGeode = new osg::Geode();
    descrText_ = new osgText::Text();
    descrText_->setCharacterSize(s);
    descrText_->setAlignment(osgText::Text::LEFT_TOP);
    descrText_->setAxisAlignment(osgText::Text::XZ_PLANE);
    descrText_->setFont(coVRFileManager::instance()->getFontFile(NULL));

    descrText_->setPosition(p);

    updateDescription(heading, numP, numN, numE);

    descrGeode->addDrawable(descrText_);
    group_->addChild(descrGeode);

    // Color
    osg::Vec4 textColor(1.0f, 1.0f, 1.0f, 1.0f);
    osg::Material *textMaterial = new osg::Material;
    textMaterial->setAmbient(osg::Material::FRONT_AND_BACK, textColor);
    textMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, textColor);
    textMaterial->setEmission(osg::Material::FRONT_AND_BACK, textColor);
    descrGeode->getOrCreateStateSet()->setAttributeAndModes(textMaterial);
}

void AtomBuilder::updateDescription(const std::string &heading, int numP, int numN, int numE)
{

    char t[1024];
    sprintf(t, coTranslator::coTranslate("%s\nProtonen: %d\nNeutronen: %d\nElektronen: %d\n").c_str(), heading.c_str(), numP, numN, numE);
    osgText::String ot(std::string(t), osgText::String::ENCODING_UTF8);
    descrText_->setText(ot);
}
