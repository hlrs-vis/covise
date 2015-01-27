/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <GL/glew.h>

#include "SortLastMaster.h"

#include <iostream>

#define COMPOSITOR_TEX_SIZE 2048

#include <config/coConfig.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cstdlib>
#include <climits>

#include <osg/Geode>
#include <osg/Group>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Vec4f>
#include <osg/Matrix>

#define SL_DEPTH_TEXTURE_MODE_F32
//#define SL_DEPTH_TEXTURE_MODE_I32
//#define SL_DEPTH_TEXTURE_MODE_I24

#define LOG_CERR(x)            \
    {                          \
        if (std::cerr.bad())   \
            std::cerr.clear(); \
        std::cerr << x;        \
    }

using namespace std;

SortLastMaster::SortLastMaster(const std::string &nodename, int session)
    : SortLastImplementation(nodename, session)
    , hostlist(0)
    , frameCtr(0)
    , session(0)
    , initPending(true)
{
}

SortLastMaster::~SortLastMaster()
{
    callPcFunc(pcContextDestroy(context), "SortLastMaster::<dest>", __LINE__);
    callPcFunc(pcSessionDestroy(), "SortLastMaster::<dest>", __LINE__);
    callPcFunc(pcSystemFinalize(), "SortLastMaster::<dest>", __LINE__);
}

bool SortLastMaster::initialiseAsMaster()
{

    covise::coConfig *config = covise::coConfig::getInstance();
    this->channelWidth = config->getInt("width", "COVER.ChannelConfig.Channel:0", 1024);
    this->channelHeight = config->getInt("height", "COVER.ChannelConfig.Channel:0", 1024);
    this->channelLeft = config->getInt("left", "COVER.ChannelConfig.Channel:0", 0);
    this->channelBottom = config->getInt("bottom", "COVER.ChannelConfig.Channel:0", 0);

    this->frameWidth = config->getInt("width", "COVER.WindowConfig.Window:0", 1024);
    this->frameHeight = config->getInt("height", "COVER.WindowConfig.Window:0", 1024);
    this->frameLeft = config->getInt("left", "COVER.WindowConfig.Window:0", 0);
    this->frameBottom = config->getInt("bottom", "COVER.WindowConfig.Window:0", 0);

    LOG_CERR("SortLastMaster::<init> info: setting size to ["
             << frameLeft << "," << frameBottom << " | " << frameWidth << "x" << frameHeight << "]"
             << ", VP ["
             << channelLeft << "," << channelBottom << " | " << channelWidth << "x" << channelHeight << "]"
             << std::endl);

    char *initString = ::getenv("CEI_PC_LIBPATH");

    LOG_CERR("SortLastMaster::initialiseAsMaster info: initialising library ("
             << initString << ")" << std::endl);

    callPcFunc(pcSystemInitialize(initString), "SortLastMaster::initialiseAsMaster", __LINE__);
    callPcFunc(pcSessionCreate(this->session), "SortLastMaster::initialiseAsMaster", __LINE__);

    return true;
}

bool SortLastMaster::createContext(const std::list<std::string> &hostlist, int groupIdentifier)
{

    LOG_CERR("SortLastMaster::createContext info: creating master context");

    int ctr = 0;

    this->session = groupIdentifier;
    this->hostlist = hostlist;

    const char **tmpHostlist = new const char *[hostlist.size() + 1];
    std::list<std::string>::const_iterator host = hostlist.begin();

    tmpHostlist[0] = host->c_str();

    LOG_CERR("SortLastMaster::createContext info: using hosts [ /" << tmpHostlist[0] << "/");
    for (++host; host != hostlist.end(); ++host)
    {
        tmpHostlist[++ctr] = host->c_str();
        std::cerr << ", " << tmpHostlist[ctr];
    }
    tmpHostlist[hostlist.size()] = 0;
    LOG_CERR(" ]" << std::endl);

    PCint attributes[100];

    int i = 0;
    attributes[i++] = PC_COMPOSITE_TYPE;
    attributes[i++] = PC_COMP_DEPTH;
#if defined SL_DEPTH_TEXTURE_MODE_I24
    attributes[i++] = PC_PIXEL_FORMAT;
    attributes[i++] = PC_PF_BGR8 | PC_PF_Z24I;
#elif defined SL_DEPTH_TEXTURE_MODE_I32
    attributes[i++] = PC_PIXEL_FORMAT;
    attributes[i++] = PC_PF_BGR8 | PC_PF_Z32I;
#elif defined SL_DEPTH_TEXTURE_MODE_F32
    attributes[i++] = PC_PIXEL_FORMAT;
    attributes[i++] = PC_PF_BGR8 | PC_PF_Z32F;
#else
#error No depth mode defined
#endif
    attributes[i++] = PC_NETWORK_ID;
    attributes[i++] = PC_ID_DEFAULT;
    attributes[i++] = PC_OUTPUT_DEPTH;
    attributes[i++] = 1;
    attributes[i++] = PC_PROPERTY_END;

    LOG_CERR("SortLastMaster::createContext info: creating global context");
    callPcFunc(pcContextCreateMaster(attributes, const_cast<char **>(tmpHostlist),
                                     const_cast<char *>(tmpHostlist[0]), &context),
               "SortLastMaster::createContext", __LINE__);
    LOG_CERR(".");

    delete[] tmpHostlist;

    PCint hostidx, numHosts;

    callPcFunc(pcContextGetInteger(context, PC_HOSTINDEX, PC_LOCALHOST_INDEX, &hostidx), "SortLastMaster::createContext", __LINE__);
    LOG_CERR(".");
    callPcFunc(pcContextGetInteger(context, PC_NUM_HOSTS, 0, &numHosts), "SortLastMaster::createContext", __LINE__);
    LOG_CERR(". -> " << context << ":" << hostidx << ":" << numHosts << std::endl);

#ifdef SL_DEMO_MODE
    static const osg::Vec4f colors[] = {
        osg::Vec4f(1.0f, 0.0f, 0.0f, 1.0f), osg::Vec4f(0.0f, 1.0f, 0.0f, 1.0f),
        osg::Vec4f(0.0f, 0.0f, 1.0f, 1.0f), osg::Vec4f(1.0f, 1.0f, 0.0f, 1.0f),
        osg::Vec4f(0.0f, 1.0f, 1.0f, 1.0f), osg::Vec4f(1.0f, 0.0f, 1.0f, 1.0f)
    };

    // Master is always first...
    int hostid = 0;
    osg::ref_ptr<osg::Box> box = new osg::Box(osg::Vec3(0.0f, 0.0f, 0.0f), 300.f);

    osg::ref_ptr<osg::MatrixTransform> group = new osg::MatrixTransform();
    group->setMatrix(osg::Matrix::translate(350.0f * (hostid - 1), 0.0f, 0.0f));
    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    osg::ref_ptr<osg::ShapeDrawable> drawable = new osg::ShapeDrawable(box);

    if (hostid < 6)
    {
        osg::ref_ptr<osg::Material> material = new osg::Material();
        material->setDiffuse(osg::Material::FRONT_AND_BACK, colors[hostid]);
        drawable->getOrCreateStateSet()->setAttributeAndModes(material.get(), osg::StateAttribute::ON);
    }

    geode->addDrawable(drawable.get());
    group->addChild(geode.get());
    opencover::cover->getObjectsRoot()->addChild(group.get());
#endif

    return true;
}

bool SortLastMaster::init()
{
    return true;
}

void SortLastMaster::preSwapBuffers(int window)
{

    (void)window;

    callPcFunc(pcContextSetInteger(context, PC_FRAME_WIDTH, 0, frameWidth), "SortLastMaster::preSwapBuffers", __LINE__);
    callPcFunc(pcContextSetInteger(context, PC_FRAME_HEIGHT, 0, frameHeight), "SortLastMaster::preSwapBuffers", __LINE__);
    callPcFunc(pcContextSetInteger(context, PC_OUTPUT_X_OFFSET, PC_MASTER_INDEX, 0), "SortLastMaster::preSwapBuffers", __LINE__);
    callPcFunc(pcContextSetInteger(context, PC_OUTPUT_Y_OFFSET, PC_MASTER_INDEX, 0), "SortLastMaster::preSwapBuffers", __LINE__);
    callPcFunc(pcContextSetInteger(context, PC_OUTPUT_WIDTH, PC_MASTER_INDEX, frameWidth), "SortLastMaster::preSwapBuffers", __LINE__);
    callPcFunc(pcContextSetInteger(context, PC_OUTPUT_HEIGHT, PC_MASTER_INDEX, frameHeight), "SortLastMaster::preSwapBuffers", __LINE__);

    //compositeSimpleReadback();
    compositeSimpleShader();
}

void SortLastMaster::initTextures(PCchannel *frameBuffer, PCchannel *depthBuffer)
{
    if (this->initPending)
    {

        this->initPending = false;

        glGenTextures(2, textures);

        for (int ctr = 0; ctr < 2; ++ctr)
        {
            glBindTexture(GL_TEXTURE_2D, textures[ctr]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        }

        if (frameBuffer != 0)
        {
            glBindTexture(GL_TEXTURE_2D, textures[0]);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

            GLubyte *pixels = new GLubyte[COMPOSITOR_TEX_SIZE * COMPOSITOR_TEX_SIZE * 4];
            memset(pixels, 255, sizeof(GLubyte) * COMPOSITOR_TEX_SIZE * COMPOSITOR_TEX_SIZE * 4);

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, COMPOSITOR_TEX_SIZE, COMPOSITOR_TEX_SIZE, 0,
                         GL_BGR, GL_UNSIGNED_BYTE, pixels);
        }

        if (depthBuffer != 0)
        {
            glBindTexture(GL_TEXTURE_2D, textures[1]);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

#if defined SL_DEPTH_TEXTURE_MODE_I32
            GLubyte *pixels = new GLubyte[COMPOSITOR_TEX_SIZE * COMPOSITOR_TEX_SIZE * 4];
            memset(pixels, 255, sizeof(GLubyte) * COMPOSITOR_TEX_SIZE * COMPOSITOR_TEX_SIZE * 4);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, COMPOSITOR_TEX_SIZE, COMPOSITOR_TEX_SIZE, 0,
                         GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, pixels);
#elif defined SL_DEPTH_TEXTURE_MODE_I24
#error Depth mode 24I not working (yet?)
#elif defined SL_DEPTH_TEXTURE_MODE_F32
            GLfloat *pixels = new GLfloat[COMPOSITOR_TEX_SIZE * COMPOSITOR_TEX_SIZE];
            for (int ctr = 0; ctr < COMPOSITOR_TEX_SIZE * COMPOSITOR_TEX_SIZE; ++ctr)
            {
                pixels[ctr] = 1.0f;
            }
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, COMPOSITOR_TEX_SIZE, COMPOSITOR_TEX_SIZE, 0,
                         GL_DEPTH_COMPONENT, GL_FLOAT, pixels);
#else
#error No depth mode defined
#endif
        }
    }
}

GLuint SortLastMaster::makeShader(GLuint program, GLuint type, const char *source)
{

    GLuint shader = glCreateShader(type);
    GLint length = strlen(source);
    glShaderSource(shader, 1, &source, &length);
    glCompileShader(shader);
    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);

    if (compiled != GL_TRUE)
    {
        GLint laux = 0;
        length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        GLchar *logString = new GLchar[length];
        glGetShaderInfoLog(shader, length, &laux, logString);
        LOG_CERR("SortLastMaster::makeShader err: failed to compile shader "
                 << length << "(" << compiled << ") "
                 << " " << logString << std::endl);
        delete[] logString;
    }

    glAttachShader(program, shader);

    return shader;
}

void SortLastMaster::compositeSimpleShader()
{

    static GLuint program = 0;

    if (this->initPending)
    {

        glewInit();
        program = glCreateProgram();

        const GLchar *fSource = "uniform sampler2D depth;"
                                "uniform sampler2D frame;"
                                "varying vec2 frameCoords;"
                                "void main() {\n"
                                "  gl_FragColor = texture2D(frame, frameCoords);\n"
                                "  gl_FragDepth = texture2D(depth, frameCoords).r;\n"
                                "}";

        const GLchar *vSource = "varying vec2 frameCoords;"
                                "varying vec2 depthCoords;"
                                "void main() {\n"
                                "  gl_Position = ftransform();\n"
                                "  frameCoords = gl_MultiTexCoord0.st;\n"
                                "}";

        makeShader(program, GL_FRAGMENT_SHADER, fSource);
        makeShader(program, GL_VERTEX_SHADER, vSource);

        glLinkProgram(program);

        GLint linked = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &linked);

        if (linked != GL_TRUE)
        {
            GLint laux = 0;
            GLint length = 0;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
            GLchar *logString = new GLchar[length];
            glGetProgramInfoLog(program, length, &laux, logString);
            LOG_CERR("SortLastMaster::compositeSimpleShader err: failed to link shader "
                     << length << "(" << linked << ") "
                     << " " << logString << std::endl);
            delete[] logString;
        }
    }

    PCid id;
    ++frameCtr;
    //LOG_CERR("\rSortLastMaster::compositeSimpleShader info: compositing frame " << frameCtr);

    callPcFunc(pcFrameBegin(context, &id, 0, 0, 0, 0, 0, 0), "SortLastMaster::compositeSimpleShader", __LINE__);
    //LOG_CERR(" (B");

    callPcFunc(pcFrameEnd(context, id), "SortLastMaster::compositeSimpleShader", __LINE__);
    //LOG_CERR("E");

    callPcFunc(pcFrameResultChannel(context, id, 0, 0, frameWidth, frameHeight,
                                    PC_CHANNEL_COLOR, &frameBuffer),
               "SortLastMaster::compositeSimpleShader", __LINE__);

    callPcFunc(pcFrameResultChannel(context, id, 0, 0, frameWidth, frameHeight,
                                    PC_CHANNEL_DEPTH, &depthBuffer),
               "SortLastMaster::compositeSimpleShader", __LINE__);

    //LOG_CERR("R)");

    //   FILE * f;
    //   f = fopen("/tmp/frame", "w");
    //   fwrite(frameBuffer.address, sizeof(GLubyte), COMPOSITOR_TEX_SIZE*COMPOSITOR_TEX_SIZE*4, f);
    //   fclose(f);
    //   f = fopen("/tmp/depth", "w");
    //   fwrite(depthBuffer.address, sizeof(GLfloat), COMPOSITOR_TEX_SIZE*COMPOSITOR_TEX_SIZE, f);
    //   fclose(f);

    initTextures(&frameBuffer, &depthBuffer);

    float texCoordX;
    float texCoordY;

    glPushAttrib(GL_VIEWPORT_BIT | GL_ENABLE_BIT);

    glEnable(GL_DEPTH_TEST);
    glViewport(channelLeft, channelBottom, channelWidth, channelHeight);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    GLint currentProgram = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, &currentProgram);

    glUseProgram(program);

    glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textures[0]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frameBuffer.width, frameBuffer.height,
                    GL_BGR, GL_UNSIGNED_BYTE, frameBuffer.address);
    int location = glGetUniformLocation(program, "frame");
    glUniform1i(location, 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, textures[1]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);

#if defined SL_DEPTH_TEXTURE_MODE_I32
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, depthBuffer.width, depthBuffer.height,
                    GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, depthBuffer.address);
#elif defined SL_DEPTH_TEXTURE_MODE_I24
#elif defined SL_DEPTH_TEXTURE_MODE_F32
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, depthBuffer.width, depthBuffer.height,
                    GL_DEPTH_COMPONENT, GL_FLOAT, depthBuffer.address);
#else
#error No depth mode defined
#endif

    location = glGetUniformLocation(program, "depth");
    glUniform1i(location, 1);

    glActiveTexture(GL_TEXTURE0);

    texCoordX = ((float)frameBuffer.width) / (float)COMPOSITOR_TEX_SIZE;
    texCoordY = ((float)frameBuffer.height) / (float)COMPOSITOR_TEX_SIZE;

    glBegin(GL_QUADS);
    glColor3f(0.9f, 0.9f, 0.9f);
    glTexCoord2f(texCoordX, texCoordY);
    glVertex3f(1.0f, 1.0f, 0.0f);
    glTexCoord2f(0.0f, texCoordY);
    glVertex3f(-1.0f, 1.0f, 0.0f);
    glTexCoord2f(0.0f, 0.0f);
    glVertex3f(-1.0f, -1.0f, 0.0f);
    glTexCoord2f(texCoordX, 0.0f);
    glVertex3f(1.0f, -1.0f, 0.0f);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glPopAttrib();
    glUseProgram(currentProgram);
}

void SortLastMaster::compositeSimpleReadback()
{

    PCid id;
    ++frameCtr;
    //LOG_CERR("\rSortLastMaster::compositeSimpleReadback info: compositing frame " << frameCtr);

    callPcFunc(pcFrameBegin(context, &id, 1, 0, 0, 0, 0, 0), "SortLastMaster::compositeSimpleReadback", __LINE__);
    //LOG_CERR(" (B");

    glReadBuffer(GL_BACK);
    callPcFunc(pcFrameAddGLFrameletEXT(context, id, 0, 0), "SortLastMaster::compositeSimpleReadback", __LINE__);
    //LOG_CERR("A");

    callPcFunc(pcFrameEnd(context, id), "SortLastMaster::compositeSimpleReadback", __LINE__);
    //LOG_CERR("E");

    callPcFunc(pcFrameResultChannel(context, id, 0, 0, frameWidth, frameHeight, PC_CHANNEL_COLOR, &frameBuffer),
               "SortLastMaster::compositeSimpleReadback", __LINE__);
    //LOG_CERR("R)");

    //    FILE * f = fopen("/tmp/frame", "w");
    //    fwrite(frame, sizeof(GLubyte), COMPOSITOR_TEX_SIZE*COMPOSITOR_TEX_SIZE*4, f);
    //    fclose(f);

    initTextures(&frameBuffer, 0);

    float texCoordX;
    float texCoordY;

    glPushAttrib(GL_VIEWPORT_BIT | GL_ENABLE_BIT);

    glViewport(channelLeft, channelBottom, channelWidth, channelHeight);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textures[0]);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frameBuffer.width, frameBuffer.height, GL_BGR, GL_UNSIGNED_BYTE, frameBuffer.address);

    texCoordX = ((float)frameBuffer.width) / (float)COMPOSITOR_TEX_SIZE;
    texCoordY = ((float)frameBuffer.height) / (float)COMPOSITOR_TEX_SIZE;

    glBegin(GL_QUADS);
    glColor3f(0.9f, 0.9f, 0.9f);
    glTexCoord2f(texCoordX, texCoordY);
    glVertex3f(1.0f, 1.0f, 0.0f);
    glTexCoord2f(0.0f, texCoordY);
    glVertex3f(-1.0f, 1.0f, 0.0f);
    glTexCoord2f(0.0f, 0.0f);
    glVertex3f(-1.0f, -1.0f, 0.0f);
    glTexCoord2f(texCoordX, 0.0f);
    glVertex3f(1.0f, -1.0f, 0.0f);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glPopAttrib();
}

void SortLastMaster::callPcFunc(PCerr error, const char *location, int line)
{
    if (error)
    {
        if (cerr.bad())
            cerr.clear();
        cerr << location << ":" << line << " err: " << pcGetErrorString(error) << endl;
        exit(1);
    }
}
