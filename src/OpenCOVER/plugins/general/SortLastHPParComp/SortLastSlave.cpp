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

#define LOG_CERR(x)            \
    {                          \
        if (std::cerr.bad())   \
            std::cerr.clear(); \
        std::cerr << x;        \
    }

SortLastSlave::SortLastSlave(const std::string &nodename, int session)
    : SortLastImplementation(nodename, session)
    , connected(false)
    , inFrame(false)
    , group(0)
{

    std::cerr << "SortLastSlave::<init> info: starting plugin" << std::endl;

#ifndef USE_HP_READBACK
    pixels = 0;
    depth = 0;
    width = 0;
    height = 0;
#endif

    //    coConfigInt hostID = coConfig::getInstance()->getInt("COVER.Plugin.SortLastHPParComp.HostID");
    //
    //    if (!hostID.hasValidValue())
    //    {
    //       std::cerr << "SortLastSlave::<init> err: no HostID specified in the configuration" << std::endl;
    //       exit(-1);
    //    }
}

SortLastSlave::~SortLastSlave()
{
    callPcFunc(pcContextDestroy(context), "SortLastSlave::<dest>", __LINE__);
    callPcFunc(pcSessionDestroy(), "SortLastSlave::<dest>", __LINE__);
    callPcFunc(pcSystemFinalize(), "SortLastSlave::<dest>", __LINE__);
}

bool SortLastSlave::initialiseAsSlave()
{
    char *initString = getenv("CEI_PC_LIBPATH");

    LOG_CERR("SortLastMaster::initialiseAsMaster info: initialising library ("
             << initString << ")" << std::endl);

    callPcFunc(pcSystemInitialize(initString), "SortLastSlave::initialiseAsSlave", __LINE__);
    callPcFunc(pcSessionCreate(this->session), "SortLastSlave::initialiseAsSlave", __LINE__);
    return true;
}

bool SortLastSlave::init()
{
    return true;
}

bool SortLastSlave::createContext(const std::list<std::string> &hostlist, int)
{

    this->index = 0;

    if (hostlist.empty())
        return false;

    for (std::list<std::string>::const_iterator h = hostlist.begin(); h != hostlist.end(); ++h)
    {
        if (*h == this->nodename)
        {
            break;
        }
        else
        {
            ++this->index;
        }
    }

    assert(this->index < hostlist.size());

    if (this->connected)
    {
        callPcFunc(pcContextDestroy(context), "SortLastSlave::createContext", __LINE__);
    }

#ifdef SL_DEMO_MODE
    static const osg::Vec4f colors[] = {
        osg::Vec4f(1.0f, 0.0f, 0.0f, 1.0f), osg::Vec4f(0.0f, 1.0f, 0.0f, 1.0f),
        osg::Vec4f(0.0f, 0.0f, 1.0f, 1.0f), osg::Vec4f(1.0f, 1.0f, 0.0f, 1.0f),
        osg::Vec4f(0.0f, 1.0f, 1.0f, 1.0f), osg::Vec4f(1.0f, 0.0f, 1.0f, 1.0f)
    };

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

#endif

    connectToCompositor();

    return true;
}

void SortLastSlave::preSwapBuffers(int windowNumber)
{

    (void)windowNumber;

    if (!this->connected)
        return;

    LOG_CERR("\rSortLastMaster::compositeSimpleShader info: compositing frame ");
    callPcFunc(pcFrameBegin(context, &id, 1, 0, 0, 0, 0, 0), "SortLastSlave::preSwapBuffers", __LINE__);
    LOG_CERR(" (B");

#ifndef USE_HP_READBACK

    callPcFunc(pcContextGetInteger(context, PC_FRAME_WIDTH, PC_ID_DEFAULT, &dx), "SortLastSlave::preSwapBuffers");
    callPcFunc(pcContextGetInteger(context, PC_FRAME_HEIGHT, PC_ID_DEFAULT, &dy), "SortLastSlave::preSwapBuffers");

    if (cover->windows[0].sx != width || cover->windows[0].sy != height)
    {
        width = cover->windows[0].sx;
        height = cover->windows[0].sy;
        LOG_CERR("SortLastSlave::preSwapBuffers info: resizing to ["
                 << width << "," << height << "]" << std::endl);
        delete[] pixels;
        delete[] depth;
        pixels = new GLubyte[width * height * 4];
        depth = new GLint[width * height];
    }
#endif

    glReadBuffer(GL_BACK);
    LOG_CERR("R");

#ifdef USE_HP_READBACK
    callPcFunc(pcFrameAddGLFrameletEXT(context, id, 0, 0), "SortLastSlave::preSwapBuffers", __LINE__);
    LOG_CERR("A");
#else
    glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, pixels);
    glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, depth);
    callPcFunc(pcFrameAddFramelet(context, id, pixels, depth), "SortLastSlave::preSwapBuffers", __LINE__);
#endif

    callPcFunc(pcFrameEnd(context, id), "SortLastSlave::preSwapBuffers", __LINE__);
    LOG_CERR("E)");

    this->inFrame = false;
}

bool SortLastSlave::connectToCompositor()
{

    if (this->connected)
        return false;

    usleep(5000000);

    LOG_CERR("SortLastSlave::connectToCompositor info: connecting as server " << this->nodename);
    callPcFunc(pcContextCreate(PC_ID_DEFAULT, const_cast<char *>(this->nodename.c_str()), &context),
               "SortLastSlave::connectToCompositor", __LINE__);

    PCint nhosts, hostidx;

    LOG_CERR(".");
    callPcFunc(pcContextGetInteger(context, PC_HOSTINDEX, PC_LOCALHOST_INDEX, &hostidx),
               "SortLastSlave::connectToCompositor", __LINE__);

    LOG_CERR(".");
    callPcFunc(pcContextGetInteger(context, PC_NUM_HOSTS, 0, &nhosts),
               "SortLastSlave::connectToCompositor", __LINE__);

    this->connected = true;

    LOG_CERR(". -> " << context << ":" << hostidx << ":" << nhosts << std::endl);

    return true;
}

void SortLastSlave::callPcFunc(PCerr error, const char *location, int line)
{
    if (error)
    {
        if (cerr.bad())
            cerr.clear();
        cerr << location << ":" << line << " err: " << pcGetErrorString(error) << endl;
        exit(1);
    }
}
