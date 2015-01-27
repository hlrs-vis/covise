/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Probe.h"
#include <alg/coIsoSurface.h>
#include <cover/VRSceneGraph.h>

#include "../../covise/COVISE/VRCoviseGeometryManager.h"

#include <cover/coTranslator.h>

#include "ChargedObjectHandler.h"
#include "ElectricFieldPlugin.h"

Probe::Probe()
    : GenericGuiObject("Probe")
{
    p_visible = addGuiParamBool("Visible", false);
    p_position = addGuiParamVec3("Position", osg::Vec3(0.01, 0.0, 0.0));
    p_showField = addGuiParamBool("LabelShowsField", true);
    p_showPotential = addGuiParamBool("LabelShowsPotential", true);
    p_showIsoSurface = addGuiParamBool("ShowIsoSurface", false);
    p_showArrow = addGuiParamBool("ShowArrow", true);

    geode = new osg::Geode();
    cylinder = new osg::Cylinder(osg::Vec3(0.0, 0.0, 0.0), 0.01, 0.0);
    cylinderD = new osg::ShapeDrawable(cylinder.get());
    geode->addDrawable(cylinderD.get());
    cone = new osg::Cone(osg::Vec3(0.0, 0.0, 0.0), 0.03, 0.05);
    coneD = new osg::ShapeDrawable(cone.get());
    geode->addDrawable(coneD.get());
    transform = new osg::MatrixTransform();
    transform->addChild(geode.get());
    cover->getObjectsRoot()->addChild(geode.get());
    label = new coVRLabel("", 24, 100.0, osg::Vec4(0.5451, 0.7020, 0.2431, 1.0), osg::Vec4(0.0, 0.0, 0.0, 0.8));
    interactor = new coVR3DTransInteractor(osg::Vec3(0.01, 0.0, 0.0), 0.06, coInteraction::ButtonA, "hand", "ProbeInteractor", coInteraction::Medium);
    interactor->enableIntersection();

    menuItemVisible = new coCheckboxMenuItem(coTranslator::coTranslate("Zeige Messpunkt mit Pfeil"), true, NULL);
    menuItemVisible->setMenuListener(this);
    ElectricFieldPlugin::plugin->getObjectsMenu()->insert(menuItemVisible, 0);

    menuItemAequipVisible = new coCheckboxMenuItem(coTranslator::coTranslate("Zeige Messpunkt mit Aequipotentialflaeche"), true, NULL);
    menuItemAequipVisible->setMenuListener(this);
    ElectricFieldPlugin::plugin->getObjectsMenu()->insert(menuItemAequipVisible, 0);

    updateMenuItem();
}

Probe::~Probe()
{
}

void Probe::preFrame()
{
    osg::Vec3 position = interactor->getPos();

    if (interactor->isRunning())
    {
        float min = ChargedObjectHandler::Instance()->getGridMin();
        float max = ChargedObjectHandler::Instance()->getGridMax();
        // reposition interactor if outside
        if (position[0] > max)
            position[0] = max;
        else if (position[0] < min)
            position[0] = min;
        if (position[1] > max)
            position[1] = max;
        else if (position[1] < min)
            position[1] = min;
        if (position[2] > max)
            position[2] = max;
        else if (position[2] < min)
            position[2] = min;

        interactor->updateTransform(position);
        update();
    }
    label->setPosition(position * cover->getBaseMat()); // always do this -> needs update when camera changes

    if (interactor->wasStopped())
    {
        p_position->setValue(position);
    }
}

void Probe::menuEvent(coMenuItem *menuItem)
{
    if (menuItem == menuItemVisible)
    {
        p_visible->setValue(menuItemVisible->getState());
        /*
      if (p_visible->getValue())
      {
         if (p_showIsoSurface->getValue())
         {
            p_visible->setValue(false);
            p_showIsoSurface->setValue(false);
         } else {
            p_showIsoSurface->setValue(true);
         }
      } else {
         p_visible->setValue(true);
         p_showIsoSurface->setValue(false);
      }
      updateMenuItem();*/
    }
    if (menuItem == menuItemAequipVisible)
        p_showIsoSurface->setValue(menuItemAequipVisible->getState());
    // if any parameter was changed, just update
    update();
}

void Probe::guiParamChanged(GuiParam *guiParam)
{
    if (guiParam == p_position)
        interactor->updateTransform(p_position->getValue());
    /*if ((guiParam == p_visible) || (guiParam == p_showIsoSurface))
      updateMenuItem();*/
    if (guiParam == p_visible)
        menuItemVisible->setState(p_visible->getValue());
    if (guiParam == p_showIsoSurface)
        menuItemAequipVisible->setState(p_showIsoSurface->getValue());
    // if any parameter was changed, just update
    update();
}

void Probe::updateMenuItem()
{
    menuItemVisible->setState(p_visible->getValue());
    menuItemAequipVisible->setState(p_showIsoSurface->getValue());
    /*if (p_visible->getValue())
   {
      if (p_showIsoSurface->getValue())
      {
         menuItemVisible->setName("Zeige Messpunkt mit Aequip.");
      } else {
         menuItemVisible->setName("Zeige Messpunkt ohne Aequip.");
      }
   } else {
      menuItemVisible->setName("Zeige Messpunkt");
   }*/
}

void Probe::update()
{
    updateProbe();
    updateIsoSurface();
}

void Probe::updateProbe()
{
    if (p_visible->getValue())
    {
        interactor->show();
        interactor->enableIntersection();

        if (ChargedObjectHandler::Instance()->fieldIsValid())
        {

            if (p_showArrow->getValue())
            {
                if (!cover->getObjectsRoot()->containsNode(geode.get()))
                    cover->getObjectsRoot()->addChild(geode.get());
            }
            else
            {
                cover->getObjectsRoot()->removeChild(geode.get());
            }
            if (p_showField->getValue() || p_showPotential->getValue())
            {
                label->show();
            }
            else
            {
                label->hide();
            }

            //fprintf(stderr,"Probe::update\n");
            osg::Vec3 position = interactor->getPos();
            osg::Vec3 field = ChargedObjectHandler::Instance()->getFieldAt(position);
            float potential = ChargedObjectHandler::Instance()->getPotentialAt(position);

            // constant length
            osg::Vec3 vector = field;
            vector.normalize();
            vector *= 0.2;

            float length = vector.length();
            osg::Vec3 lineCenter = position + vector * 0.5;
            osg::Vec3 lineEnd = position + vector;
            osg::Matrix m;
            m.makeRotate(osg::Vec3(0.0, 0.0, length), vector);

            cylinder->setCenter(lineCenter);
            cylinder->setHeight(length);
            cylinder->setRotation(m.getRotate());
            cylinderD->dirtyDisplayList();
            cone->setCenter(lineEnd);
            cone->setRotation(m.getRotate());
            coneD->dirtyDisplayList();

            char buffer[50];
            if (p_showField->getValue() && p_showPotential->getValue())
            {
                sprintf(buffer, "%.2f V/m   %.2f V", field.length(), potential);
            }
            else if (p_showField->getValue() && !p_showPotential->getValue())
            {
                sprintf(buffer, "%.2f V/m", field.length());
            }
            else if (!p_showField->getValue() && p_showPotential->getValue())
            {
                sprintf(buffer, "%.2f V", potential);
            }
            label->setString(buffer);
        }
        else
        {
            cover->getObjectsRoot()->removeChild(geode.get());
            label->hide();
        }
    }
    else
    {
        interactor->hide();
        interactor->disableIntersection();
        cover->getObjectsRoot()->removeChild(geode.get());
        label->hide();
    }
}

void Probe::updateIsoSurface()
{
    if (isoPlane_.get() && cover->getObjectsRoot()->containsNode(isoPlane_.get()))
    {
        cover->getObjectsRoot()->removeChild(isoPlane_.get());
    }

    if (/*p_visible->getValue() &&*/ p_showIsoSurface->getValue() && ChargedObjectHandler::Instance()->fieldIsValid())
    {
        interactor->show();
        interactor->enableIntersection();
        if (p_showField->getValue() || p_showPotential->getValue())
            label->show();
        else
            label->hide();

        int steps_ = ChargedObjectHandler::Instance()->getGridSteps();
        float min_ = ChargedObjectHandler::Instance()->getGridMin();
        float max_ = ChargedObjectHandler::Instance()->getGridMax();
        float *s_ = ChargedObjectHandler::Instance()->getFieldPotential();
        float pot = ChargedObjectHandler::Instance()->getPotentialAt(interactor->getPos());

        // create IsoPlane
        UNI_IsoPlane *uplane = new UNI_IsoPlane((steps_ - 1) * (steps_ - 1) * (steps_ - 1), steps_ * steps_ * steps_, 1,
                                                min_, max_, min_, max_, min_, max_,
                                                steps_, steps_, steps_,
                                                NULL, s_, NULL, NULL, NULL, pot,
                                                false, NULL);
        uplane->createIsoPlane();

        int num_triangles = uplane->getNumTriangles();
        int *pl = new int[num_triangles];
        for (int i = 0; i < num_triangles; i++)
        {
            pl[i] = i * 3;
        }
        //fprintf(stderr,"create iso surface with %d triangles, %d, %d\n", num_triangles, uplane->getNumVertices(),uplane->getNumCoords() );

        float color[3] = { 0.5, 0.5, 0.5 };
        coMaterial *material = new coMaterial("iso", color, color, color, color, 1.0, 0.5);

        float trans = 0.;
        isoPlane_ = GeometryManager::instance()->addPolygon("isoPlaneElectricField", num_triangles, uplane->getNumVertices(),
                                                            uplane->getNumCoords(), uplane->getXout(), uplane->getYout(), uplane->getZout(),
                                                            uplane->getVerticeList(), pl,
                                                            0, 0, 0, NULL, NULL, NULL, NULL,
                                                            0, 0, NULL, NULL, NULL, trans,
                                                            2, material, 0, 0, 0, NULL, 0, NULL, NULL, osg::Texture::REPEAT, osg::Texture::LINEAR, osg::Texture::LINEAR, 0, NULL, NULL, NULL, false);
        isoPlane_->setNodeMask(isoPlane_->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
        cover->getObjectsRoot()->addChild(isoPlane_.get());

        delete[] pl;
        delete uplane;
    }
}
