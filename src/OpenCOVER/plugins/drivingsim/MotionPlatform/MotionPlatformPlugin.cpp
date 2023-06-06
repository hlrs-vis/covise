/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Template Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "MotionPlatformPlugin.h"
#include <cover/coVRTui.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

MotionPlatformPlugin::MotionPlatformPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "MotionPlatformPlugin::MotionPlatformPlugin\n");

    seteps(1e-6);
    setnmax(100);
    filedirpath = coCoviseConfig::getEntry("value", "COVER.Plugin.MotionPlatform.Filepath", "/data/Fahrsimulator/OpenSceneGraph");
    // dL = hubhoehe der drei Linearmotoren; // 400mm
    dL = 0.4;
    yawAxis = osg::Vec3(0.f, 0.f, 1.f);
    pitchAxis = osg::Vec3(1.f, 0.f, 0.f);
    rollAxis = osg::Vec3(0.f, 1.f, 0.f);
    // tmp-Werte der Linearmotoren
    wTml = osg::Matrix::translate(0.675, 0.887, 0.228);
    wTmr = osg::Matrix::translate(0.675, -0.386, 0.228);
    wTmh = osg::Matrix::translate(-0.482, 0.251, 0.228);
    tmp = 2.0;
    // Nullhoehen der Linearmotoren;
    hl0 = wTml(3, 2);
    hr0 = wTmr(3, 2);
    hh0 = wTmh(3, 2);
    // tmp-Werte der Kugelkoepfe
    wTkl = osg::Matrix::identity();
    wTkr = osg::Matrix::identity();
    wTkh = osg::Matrix::identity();
    //tmp-Werte der Querstange
    wTq = osg::Matrix::identity();
    //tmp-Werte der Zange
    wZ = osg::Vec3(0.245, 0.251, 0.03);
    wTz = osg::Matrix::translate(wZ);
    //tmp-Werte der Linearfuehrung
    wTl = osg::Matrix::identity();
    zL = osg::Vec3(0.85, 0.0, 0.19);
    //tmp-Werte des Aufbaus
    wTa = osg::Matrix::identity();

    X0 = boost::numeric::ublas::vector<double>(2);
    X = boost::numeric::ublas::vector<double>(2);
    // Zeit
    t = 0.0;
    counter = 0;

    // Create the Group root node.
    osg::ref_ptr<osg::Group> root = new osg::Group;
    root->setName("Root Node");
    // Data variance is STATIC because we won't modify it.
    root->setDataVariance(osg::Object::DYNAMIC);
    // Load Gestell.
    std::string gestellname = this->filedirpath;
    gestellname.append("/Gestell_Geo.ive");
    osg::Node *node = osgDB::readNodeFile(gestellname);
    if (node)
    {
        osg::ref_ptr<osg::MatrixTransform> mtGestell = dynamic_cast<osg::MatrixTransform *>(node);
        mtGestell->setName("Gestell_Trans");
        mtGestell->setDataVariance(osg::Object::STATIC);
        mtGestell->setMatrix(osg::Matrix::translate(0.405, 0.251, 0.0));

        Aufbau = InsertSceneElement(root.get(), "Aufbau_Geo.ive", "Aufbau_Trans");
        Quertraeger = InsertSceneElement(root.get(), "Quertraeger_Geo.ive", "Quertraeger_Trans");
        LinMot_L = InsertSceneElement(root.get(), "Linearmotor_links_Geo.ive", "Linearmotor_links_Trans");
        LinMot_R = InsertSceneElement(root.get(), "Linearmotor_rechts_Geo.ive", "Linearmotor_rechts_Trans");
        LinMot_H = InsertSceneElement(root.get(), "Linearmotor_hinten_Geo.ive", "Linearmotor_hinten_Trans");
        KK_L = InsertSceneElement(root.get(), "Kugelkopf_links_geo.ive", "Kugelkopf_links_Trans");
        KK_R = InsertSceneElement(root.get(), "Kugelkopf_rechts_geo.ive", "Kugelkopf_rechts_Trans");
        KK_H = InsertSceneElement(root.get(), "Kugelkopf_hinten_geo.ive", "Kugelkopf_hinten_Trans");
        Zange = InsertSceneElement(root.get(), "zange_Geo.ive", "zange_Geo_Trans");
        LinFuehr = InsertSceneElement(root.get(), "linearfuehrung_Geo.ive", "linearfuehrung_Geo_Trans");

        // Add Scene Elements
        root->addChild(mtGestell.get());
    }

    cover->getObjectsRoot()->addChild(root.get());

    //Startwerte fuer Newtonverfahren
    //lambda0
    X0[0] = 0.42;
    //alpha0
    X0[1] = 0;

    MotionPlatformTab = new coTUITab("Annotations", coVRTui::instance()->mainFolder->getID());
    MotionPlatformTab->setPos(0, 0);

    animateButton = new coTUIToggleButton("Animate", MotionPlatformTab->getID());
    animateButton->setPos(0, 0);
    animateButton->setEventListener(this);
    leftSlider = new coTUIFloatSlider("left", MotionPlatformTab->getID());
    leftSlider->setValue(hl0);
    leftSlider->setTicks(1000);
    leftSlider->setMin(hl0);
    leftSlider->setMax(dL + hl0);
    leftSlider->setPos(0, 1);
    leftSlider->setEventListener(this);
    rightSlider = new coTUIFloatSlider("right", MotionPlatformTab->getID());
    rightSlider->setValue(hl0);
    rightSlider->setTicks(1000);
    rightSlider->setMin(hl0);
    rightSlider->setMax(dL + hl0);
    rightSlider->setPos(0, 2);
    rightSlider->setEventListener(this);
    backSlider = new coTUIFloatSlider("back", MotionPlatformTab->getID());
    backSlider->setValue(hl0);
    backSlider->setTicks(1000);
    backSlider->setMin(hl0);
    backSlider->setMax(dL + hl0);
    backSlider->setPos(0, 3);
    backSlider->setEventListener(this);
}
osg::ref_ptr<osg::MatrixTransform> MotionPlatformPlugin::InsertSceneElement(osg::Group *parent,
                                                                            std::string filepath,
                                                                            std::string nodename)
{
    // Load Aufbau.
    std::string tmppath = this->filedirpath;
    tmppath.append("/");
    tmppath.append(filepath);
    std::cout << tmppath << std::endl;
    osg::Node *fn = osgDB::readNodeFile(tmppath);
    // Data variance is STATIC because we won't modify it.
    fn->setDataVariance(osg::Object::STATIC);
    // Create a MatrixTransform
    osg::ref_ptr<osg::MatrixTransform> mtfn = new osg::MatrixTransform;
    mtfn->setName(nodename);
    // Set data variance to DYNAMIC to let OSG know that we
    // will modify this node during the update traversal.
    mtfn->setDataVariance(osg::Object::DYNAMIC);
    mtfn->addChild(fn);
    parent->addChild(mtfn.get());
    return mtfn.get();
}

// this is called if the plugin is removed at runtime
MotionPlatformPlugin::~MotionPlatformPlugin()
{
    fprintf(stderr, "MotionPlatformPlugin::~MotionPlatformPlugin\n");
}

void
MotionPlatformPlugin::preFrame()
{
    if (animateButton->getState())
    {
        wTml(3, 2) = hl0 + dL * (1.0 + 0.5 * sin(2.0 * M_PI * t));
        wTmr(3, 2) = hr0 + dL * (1.0 + 0.5 * sin(4.0 * M_PI * t));
        wTmh(3, 2) = hh0 + dL * (1.0 + 0.5 * sin(2.5 * M_PI * t));
    }
    else
    {
        wTml(3, 2) = leftSlider->getValue();
        wTmr(3, 2) = rightSlider->getValue();
        wTmh(3, 2) = backSlider->getValue();
    }

    osg::Vec3 wQ = ((wTml.getTrans() + wTmr.getTrans()) * 0.5);

    osg::Vec3 lx = wQ - wTmh.getTrans();
    lx.normalize();
    osg::Vec3 ly = wTml.getTrans() - wTmr.getTrans();
    ly.normalize();
    osg::Vec3 lz = lx ^ ly;
    lz.normalize();

    // Vorsicht!! OSG nimmt "verdrehte" Matrizen (Zeilenvektoren!!)
    wTq.set(lx[0], lx[1], lx[2], 0.0,
            ly[0], ly[1], ly[2], 0.0,
            lz[0], lz[1], lz[2], 0.0,
            wQ[0], wQ[1], wQ[2], 1.0);

    ly = lz ^ lx;
    wTa.set(lx[0], lx[1], lx[2], 0.0,
            ly[0], ly[1], ly[2], 0.0,
            lz[0], lz[1], lz[2], 0.0,
            wQ[0], wQ[1], wQ[2], 1.0);

    wTkl = wTq;
    wTkl.setTrans(wTml.getTrans());
    wTkr = wTq;
    wTkr.setTrans(wTmr.getTrans());
    wTkh = wTq;
    wTkh.setTrans(wTmh.getTrans());

    this->solve_4O(X0, X);

    double lambda = X[0];
    double alpha = X[1];

    if (counter % 60 == 0)
    {
        counter = 0;
        std::cout << "alpha : " << alpha << "; lambda: " << lambda << std::endl;
    }

    //osg::Quat quat(0.0f, 1.0f, 0.0f, alpha);
    osg::Quat quat(0.0f, yawAxis, 0.0f, pitchAxis, alpha, rollAxis);
    wTz.setRotate(quat);

    osg::Vec3 aL(lambda, 0.0f, 0.0f);
    aL = wTq.getTrans() + osg::Matrix::transform3x3(aL, wTq);

    wTl.set(
        wTq(0, 0), 0.0f, wTq(0, 2), 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        wTq(2, 0), 0.0f, wTq(2, 2), 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f);

    wTl.setTrans(aL);

    Aufbau->setMatrix(wTa);
    Quertraeger->setMatrix(wTq);
    LinMot_L->setMatrix(wTml);
    LinMot_R->setMatrix(wTmr);
    LinMot_H->setMatrix(wTmh);
    KK_L->setMatrix(wTkl);
    KK_R->setMatrix(wTkr);
    KK_H->setMatrix(wTkh);
    Zange->setMatrix(wTz);
    LinFuehr->setMatrix(wTl);

    X0 = X;

    t += 0.001667;
    counter++;
}

COVERPLUGIN(MotionPlatformPlugin)
