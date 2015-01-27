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
#include <cover/coVRConfig.h>
#include <cstdlib>
#include <climits>
#include <algorithm>

#include <osg/Geode>
#include <osg/Group>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Vec4f>
#include <osg/Matrix>

#include <sstream>

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

using namespace std;

static int operator_toInt(const std::string &value)
{
    int v = atoi(value.c_str());
    return v;
}

SortLastMaster::SortLastMaster(const std::string &nodename, int session)
    : SortLastImplementation(nodename, session)
    , frameBuffers(0)
    , depthBuffers(0)
    , textures(0)
    , numTextures(0)
    , program(0)
    , fragmentShader(0)
    , frameCtr(0)
    , initPending(true)
{
}

SortLastMaster::~SortLastMaster()
{
    deleteBuffers();
}

bool SortLastMaster::initialiseAsMaster()
{
    opencover::coVRConfig *config = opencover::coVRConfig::instance();

    this->frame.width = config->windows[0].sx;
    this->frame.height = config->windows[0].sy;
    this->frame.left = config->windows[0].ox;
    this->frame.bottom = config->windows[0].oy;

    this->channel.leftMargin = (int)(config->screens[0].viewportXMin * frame.width);
    this->channel.rightMargin = (int)(config->screens[0].viewportXMax * frame.width);
    this->channel.bottomMargin = (int)(config->screens[0].viewportYMin * frame.height);
    this->channel.topMargin = (int)(config->screens[0].viewportYMax * frame.height);

    LOG_CERR("SortLastMaster::<init> info: setting size to ["
             << frame.left << "," << frame.bottom << " | " << frame.width << "x" << frame.height << "]"
             << ", VP ["
             << channel.leftMargin << ", " << channel.bottomMargin << " | " << channel.rightMargin << ", " << channel.topMargin << "]"
             << std::endl);

    return true;
}

bool SortLastMaster::createContext(const std::list<std::string> &hostlist, int groupIdentifier)
{

    LOG_CERR("SortLastMaster::createContext info: creating master context");

    this->session = groupIdentifier;

    if (this->hostlist.size() != hostlist.size())
    {

        deleteBuffers();
        this->frameBuffers = new FrameBuffer *[hostlist.size()];
        this->depthBuffers = new DepthBuffer *[hostlist.size()];

        for (int ctr = 0; ctr < hostlist.size(); ++ctr)
        {
            this->frameBuffers[ctr] = new FrameBuffer(COMPOSITOR_TEX_SIZE * COMPOSITOR_TEX_SIZE * 3, 3);
            this->depthBuffers[ctr] = new DepthBuffer(COMPOSITOR_TEX_SIZE * COMPOSITOR_TEX_SIZE);

            for (int i = 0; i < this->frameBuffers[ctr]->size; ++i)
            {
                this->frameBuffers[ctr]->data[i] = FrameBuffer::bufferInit;
            }

            for (int i = 0; i < this->depthBuffers[ctr]->size; ++i)
            {
                this->depthBuffers[ctr]->data[i] = DepthBuffer::bufferInit;
            }
        }
    }

    this->hostlist.resize(hostlist.size());
    std::transform(hostlist.begin(), hostlist.end(), this->hostlist.begin(), operator_toInt);

    for (int ctr = 1; ctr < this->hostlist.size(); ++ctr)
    {
        CO_MPI_SEND(&this->frame, sizeof(frame), MPI_BYTE, this->hostlist[ctr],
                    opencover::coVRMSController::AppTag, opencover::coVRMSController::instance()->getAppCommunicator());
    }

    // Create fragment shader
    if (this->hostlist.size() < 2)
    {
        LOG_CERR("SortLastMaster::createContext info: a minimum of 1 slave is needed for compositing");
        exit(-1);
    }

    std::stringstream fSource;

    fSource << "uniform sampler2D textures[" << (this->hostlist.size() - 1) * 2 << "]; \n";
    fSource << "varying vec2 frameCoords; \n";
    fSource << "void main() { \n";
    fSource << "  gl_FragColor = texture2D(textures[0], frameCoords); \n";
    fSource << "  float depth  = texture2D(textures[1], frameCoords).r; \n";
    fSource << "  gl_FragDepth = depth; \n";
    for (int ctr = 1; ctr < this->hostlist.size() - 1; ++ctr)
    {
        fSource << "  depth = texture2D(textures[" << 2 * ctr + 1 << "], frameCoords).r; \n";
        fSource << "  if (gl_FragDepth > depth) { \n";
        fSource << "    gl_FragDepth = depth; \n";
        fSource << "    gl_FragColor = texture2D(textures[" << 2 * ctr << "], frameCoords); \n";
        fSource << "  } \n";
    }
    fSource << "}\n";

    this->fragmentSource = fSource.str();
    //LOG_CERR(std::endl << this->fragmentSource << std::endl);

    this->initPending = true;

#ifdef SL_DEMO_MODE
    {
        static const osg::Vec4f colors[] = {
            osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f), osg::Vec4f(0.0f, 1.0f, 0.0f, 1.0f),
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
    }
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

    //    for (int ctr = 1; ctr < this->hostlist.size(); ++ctr)
    //    {
    //       CO_MPI_SEND(&frame, sizeof(frame), MPI_BYTE, this->hostlist[ctr],
    //                   opencover::coVRMSController::AppTag, opencover::coVRMSController::instance()->getAppCommunicator());
    //    }

    //compositeSimpleReadback();
    compositeSimpleShader();
}

void SortLastMaster::initTextures()
{
    if (this->initPending)
    {

        this->initPending = false;

        if ((this->hostlist.size() - 1) * 2 == this->numTextures)
            return; // Nothing to do

        // Make shader
        const GLchar *fSource = this->fragmentSource.c_str();

        this->fragmentShader = makeShader(program, GL_FRAGMENT_SHADER, fSource, this->fragmentShader);

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
            LOG_CERR("SortLastMaster::initTextures err: failed to link shader "
                     << length << "(" << linked << ") "
                     << " " << logString << std::endl);
            delete[] logString;
        }
        else
        {
            LOG_CERR("SortLastMaster::initTextures info: shader successfully linked" << std::endl);
        }

        // Create textures
        if (textures)
        {
            glDeleteTextures(this->numTextures, this->textures);
            delete[] textures;
        }

        this->numTextures = (this->hostlist.size() - 1) * 2;
        this->textures = new GLuint[this->numTextures];
        glGenTextures(this->numTextures, textures);

        LOG_CERR("SortLastMaster::initTextures info: generating " << this->numTextures << " textures" << std::endl);

        for (int ctr = 0; ctr < this->numTextures; ++ctr)
        {
            // frame sampler
            glActiveTexture(GL_TEXTURE0 + ctr);
            glBindTexture(GL_TEXTURE_2D, textures[ctr]);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

            GLubyte *pixels = new GLubyte[COMPOSITOR_TEX_SIZE * COMPOSITOR_TEX_SIZE * 3];
            memset(pixels, 255, sizeof(GLubyte) * COMPOSITOR_TEX_SIZE * COMPOSITOR_TEX_SIZE * 3);

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, COMPOSITOR_TEX_SIZE, COMPOSITOR_TEX_SIZE, 0,
                         GL_BGR, GL_UNSIGNED_BYTE, pixels);

            // depth sampler
            ++ctr;

            glActiveTexture(GL_TEXTURE0 + ctr);
            glBindTexture(GL_TEXTURE_2D, textures[ctr]);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

#if defined SL_DEPTH_TEXTURE_MODE_I32
#error Depth mode 32I not implemented yet
            GLubyte *depth = new GLubyte[COMPOSITOR_TEX_SIZE * COMPOSITOR_TEX_SIZE * 4];
            memset(depth, 255, sizeof(GLubyte) * COMPOSITOR_TEX_SIZE * COMPOSITOR_TEX_SIZE * 4);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, COMPOSITOR_TEX_SIZE, COMPOSITOR_TEX_SIZE, 0,
                         GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, depth);
#elif defined SL_DEPTH_TEXTURE_MODE_I24
#error Depth mode 24I not working (yet?)
#elif defined SL_DEPTH_TEXTURE_MODE_F32
            GLfloat *depth = new GLfloat[COMPOSITOR_TEX_SIZE * COMPOSITOR_TEX_SIZE];
            for (int ctr = 0; ctr < COMPOSITOR_TEX_SIZE * COMPOSITOR_TEX_SIZE; ++ctr)
            {
                depth[ctr] = 1.0f;
            }
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, COMPOSITOR_TEX_SIZE, COMPOSITOR_TEX_SIZE, 0,
                         GL_DEPTH_COMPONENT, GL_FLOAT, depth);
#else
#error No depth mode defined
#endif
        }
    }
}

GLuint SortLastMaster::makeShader(GLuint program, GLuint type, const char *source, GLuint oldShader)
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
        return oldShader;
    }
    else
    {
        LOG_CERR("SortLastMaster::makeShader info: compiled shader " << type << std::endl);

        if (oldShader != 0)
        {
            glDetachShader(program, oldShader);
            glDeleteShader(oldShader);
        }

        glAttachShader(program, shader);
        return shader;
    }
}

void SortLastMaster::compositeSimpleShader()
{

    static const GLint textureIndex[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    if (this->program == 0)
    {
        glewInit();
        program = glCreateProgram();
        const GLchar *vSource = "varying vec2 frameCoords;"
                                "varying vec2 depthCoords;"
                                "void main() {\n"
                                "  gl_Position = ftransform();\n"
                                "  frameCoords = gl_MultiTexCoord0.st;\n"
                                "}";
        makeShader(program, GL_VERTEX_SHADER, vSource);
    }

    for (int ctr = 0; ctr < this->hostlist.size() - 1; ++ctr)
    {
        MPI_Status status;
        MPI_Recv(this->frameBuffers[ctr]->data, this->frame.width * this->frame.height * this->frameBuffers[ctr]->componentSize,
                 this->frameBuffers[ctr]->mpiType, this->hostlist[ctr + 1],
                 opencover::coVRMSController::AppTag, opencover::coVRMSController::instance()->getAppCommunicator(),
                 &status);
        MPI_Recv(this->depthBuffers[ctr]->data, this->frame.width * this->frame.height * this->frameBuffers[ctr]->componentSize,
                 this->depthBuffers[ctr]->mpiType, this->hostlist[ctr + 1],
                 opencover::coVRMSController::AppTag, opencover::coVRMSController::instance()->getAppCommunicator(),
                 &status);

        //       if (ctr == 3)
        //       {
        //          FILE * f;
        //          f = fopen("/tmp/frame", "w");
        //          fwrite(frameBuffers[ctr]->data, sizeof(GLubyte), COMPOSITOR_TEX_SIZE*COMPOSITOR_TEX_SIZE*3, f);
        //          fclose(f);
        //          f = fopen("/tmp/depth", "w");
        //          fwrite(depthBuffers[ctr]->data, sizeof(GLfloat), COMPOSITOR_TEX_SIZE*COMPOSITOR_TEX_SIZE, f);
        //          fclose(f);
        //       }
    }

    initTextures();

    glPushAttrib(GL_VIEWPORT_BIT | GL_ENABLE_BIT);

    glEnable(GL_DEPTH_TEST);
    glViewport(0, 0, frame.width, frame.height);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    GLint currentProgram = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, &currentProgram);

    glUseProgram(program);

    int location = glGetUniformLocation(program, "numTextures");
    glUniform1i(location, this->numTextures);
    location = glGetUniformLocation(program, "textures");
    glUniform1iv(location, 16, textureIndex);

    glEnable(GL_TEXTURE_2D);

    for (int ctr = 0; ctr < this->numTextures; ++ctr)
    {
        glActiveTexture(GL_TEXTURE0 + ctr);
        glBindTexture(GL_TEXTURE_2D, textures[ctr]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->frame.width, this->frame.height,
                        GL_BGR, GL_UNSIGNED_BYTE, this->frameBuffers[ctr / 2]->data);

        glActiveTexture(GL_TEXTURE0 + ++ctr);
        glBindTexture(GL_TEXTURE_2D, textures[ctr]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);

#if defined SL_DEPTH_TEXTURE_MODE_I32
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, depthBuffer.width, depthBuffer.height,
                        GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, depthBuffers[ctr / 2]->data);
#elif defined SL_DEPTH_TEXTURE_MODE_I24
#elif defined SL_DEPTH_TEXTURE_MODE_F32
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->frame.width, this->frame.height,
                        GL_DEPTH_COMPONENT, GL_FLOAT, depthBuffers[ctr / 2]->data);
#else
#error No depth mode defined
#endif
    }

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textures[0]);

    float texCoordXMax = ((float)this->frame.width) / (float)COMPOSITOR_TEX_SIZE;
    float texCoordYMax = ((float)this->frame.height) / (float)COMPOSITOR_TEX_SIZE;
    ;
    float texCoordXMin = 0.0f;
    float texCoordYMin = 0.0f;

    glBegin(GL_QUADS);
    glColor3f(0.9f, 0.9f, 0.9f);
    glTexCoord2f(texCoordXMax, texCoordYMax);
    glVertex3f(1.0f, 1.0f, 0.0f);
    glTexCoord2f(texCoordXMin, texCoordYMax);
    glVertex3f(-1.0f, 1.0f, 0.0f);
    glTexCoord2f(texCoordXMin, texCoordYMin);
    glVertex3f(-1.0f, -1.0f, 0.0f);
    glTexCoord2f(texCoordXMax, texCoordYMin);
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

void SortLastMaster::deleteBuffers()
{
    if (this->frameBuffers)
    {
        for (int ctr = 0; ctr < this->hostlist.size() - 1; ++ctr)
        {
            delete[] this->frameBuffers[ctr];
        }
    }
    if (this->depthBuffers)
    {
        for (int ctr = 0; ctr < this->hostlist.size() - 1; ++ctr)
        {
            delete[] this->depthBuffers[ctr];
        }
    }

    delete[] this->frameBuffers;
    delete[] this->depthBuffers;

    this->frameBuffers = 0;
    this->depthBuffers = 0;
}
