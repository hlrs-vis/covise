/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                            (C)2006 HLRS  **
 **                                                                          **
 ** Description: Parallel Rendering using the HP parallel compositing        **
 ** libary                                                                   **
 **                                                                          **
 ** Author: Andreas Kopecki                                                  **
 **                                                                          **
\****************************************************************************/

#include "SortLastSlave.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRMSController.h>
#include <cover/coVRConfig.h>

#include <config/coConfig.h>

#include <iostream>

#include <osg/Geode>
#include <osg/Group>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Vec4f>
#include <osg/Matrix>
#include <osg/PolygonMode>
#include <osg/StateSet>

#include <mpi.h>
//#define MPI_BCAST

#ifndef CO_MPI_SEND
#define CO_MPI_SEND MPI_Ssend
#endif

//#define SL_DEMO_MODE

#define LOG_CERR(x)            \
    {                          \
        if (std::cerr.bad())   \
            std::cerr.clear(); \
        std::cerr << x;        \
    }

static int operator_toInt(const std::string &value)
{
    return atoi(value.c_str());
}

SortLastSlave::SortLastSlave(const std::string &nodename, int session)
    : SortLastImplementation(nodename, session)
    , index(0)
    , inFrame(false)
    , group(0)
{

    std::cerr << "SortLastSlave::<init> info: starting plugin" << std::endl;

    pixels = 0;
    depth = 0;
    width = 0;
    height = 0;
}

SortLastSlave::~SortLastSlave()
{
}

bool SortLastSlave::initialiseAsSlave()
{
    return true;
}

bool SortLastSlave::init()
{
    return true;
}

bool SortLastSlave::createContext(const std::list<std::string> &hostlist, int)
{

    static const osg::Vec4f colors[] = {
        osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f), osg::Vec4f(0.0f, 1.0f, 0.0f, 1.0f),
        osg::Vec4f(0.0f, 0.0f, 1.0f, 1.0f), osg::Vec4f(1.0f, 1.0f, 0.0f, 1.0f),
        osg::Vec4f(0.0f, 1.0f, 1.0f, 1.0f), osg::Vec4f(1.0f, 0.0f, 1.0f, 1.0f)
    };

    if (hostlist.empty())
        return false;

    this->hostlist.resize(hostlist.size());
    std::transform(hostlist.begin(), hostlist.end(), this->hostlist.begin(), operator_toInt);

    for (int ctr = 1; ctr < hostlist.size(); ++ctr)
    {
        if (this->hostlist[ctr] == opencover::coVRMSController::instance()->getID())
        {
            this->index = ctr;
            break;
        }
    }

    if (covise::coConfig::getInstance()->isOn("COVER.Parallel.SortLast.ColourizeSlaves", false))
    {
        osg::Material *material = new osg::Material();
        material->setDiffuse(osg::Material::FRONT_AND_BACK, colors[this->index]);
        material->setAmbient(osg::Material::FRONT_AND_BACK, colors[this->index]);
        material->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.1f, 0.1f, 0.1f, 1.0f));
        material->setShininess(osg::Material::FRONT_AND_BACK, 10.f);
        material->setColorMode(osg::Material::OFF);

        osg::StateSet *ss = opencover::cover->getObjectsRoot()->getOrCreateStateSet();
        osg::PolygonMode *polymode = new osg::PolygonMode();
        polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::FILL);
        ss->setAttribute(material, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED);
        ss->setAttributeAndModes(polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED | osg::StateAttribute::ON);
    }

#ifdef SL_DEMO_MODE
    {
        osg::ref_ptr<osg::Box> box = new osg::Box(osg::Vec3(0.0f, 0.0f, 0.0f), 300.f);

        if (this->group.valid())
        {
            opencover::cover->getObjectsRoot()->removeChild(this->group.get());
        }

        this->group = new osg::MatrixTransform();
        this->group->setMatrix(osg::Matrix::translate(350.0f * (this->index - 1.0f), 0.0f, 0.0f));
        osg::ref_ptr<osg::Geode> geode = new osg::Geode();
        osg::ref_ptr<osg::ShapeDrawable> drawable = new osg::ShapeDrawable(box);

        if (this->index < 6)
        {
            osg::ref_ptr<osg::Material> material = new osg::Material();
            material->setDiffuse(osg::Material::FRONT_AND_BACK, colors[this->index]);
            drawable->getOrCreateStateSet()->setAttributeAndModes(material.get(), osg::StateAttribute::ON);
        }

        geode->addDrawable(drawable.get());
        this->group->addChild(geode.get());
        opencover::cover->getObjectsRoot()->addChild(this->group.get());

        //   QString labelText = QString("SortLast #%1").arg(hostid);

        //   text = new osgText::Text();
        //   text->setDataVariance(Object::DYNAMIC);
        //   text->setFont(vruiRendererInterface::the()->getName("fonts/arial.ttf"));
        //   text->setDrawMode(Text::TEXT);
        //   text->setCharacterSize(20);
        //   text->setText(labelText.toUtf8().data(), osgText::String::ENCODING_UTF8);
        //   text->setAxisAlignment(Text::XY_PLANE);
    }
#endif

    MPI_Status status;
    MPI_Recv(&this->frame, sizeof(Frame), MPI_BYTE, this->hostlist[0],
             opencover::coVRMSController::AppTag, opencover::coVRMSController::instance()->getAppCommunicator(),
             &status);

    delete[] this->pixels;
    delete[] this->depth;

    this->pixels = new GLubyte[this->frame.width * this->frame.height * 3];
    this->depth = new GLfloat[this->frame.width * this->frame.height];

    return true;
}

void SortLastSlave::preSwapBuffers(int)
{

    if (opencover::coVRConfig::instance()->windows[0].sx != width || opencover::coVRConfig::instance()->windows[0].sy != height)
    {
        width = opencover::coVRConfig::instance()->windows[0].sx;
        height = opencover::coVRConfig::instance()->windows[0].sy;
        LOG_CERR("SortLastSlave::preSwapBuffers info: resizing to ["
                 << width << "," << height << "]" << std::endl);
        delete[] pixels;
        delete[] depth;
        pixels = new GLubyte[width * height * 4];
        depth = new GLfloat[width * height];
    }

    glReadBuffer(GL_BACK);

    glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, pixels);
    glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, depth);

    CO_MPI_SEND(pixels, width * height * 3, MPI_BYTE, this->hostlist[0],
                opencover::coVRMSController::AppTag, opencover::coVRMSController::instance()->getAppCommunicator());
    CO_MPI_SEND(depth, width * height, MPI_FLOAT, this->hostlist[0],
                opencover::coVRMSController::AppTag, opencover::coVRMSController::instance()->getAppCommunicator());

    this->inFrame = false;
}
