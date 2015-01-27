/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _WEBGL_RENDERER_H
#define _WEBGL_RENDERER_H

#include <sstream>

#include <stdarg.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netdb.h>

#include <map>
#include <microhttpd.h>

#include <appl/RenderInterface.h>
#include <OpenThreads/Thread>
#include <OpenThreads/Mutex>
#include <OpenThreads/Condition>

#include "md5.h"

class WebGLRenderer *renderer;

class WebSocketThread : public OpenThreads::Thread
{

public:
    WebSocketThread(int s, WebGLRenderer *r);
    ~WebSocketThread();

    virtual void run();

private:
    int socket;
    WebGLRenderer *renderer;
    bool finished;
};

class WebSocketServer : public OpenThreads::Thread
{

public:
    WebSocketServer(WebGLRenderer *r);
    ~WebSocketServer();

    virtual void run();

private:
    int sock;
    WebGLRenderer *renderer;
    bool finished;

    std::vector<WebSocketThread *> threads;
};

using covise::coDistributedObject;

struct file
{
    unsigned char *buf;
    const char *mimeType;
    unsigned int size;
};

struct ltstr
{
    bool operator()(const char *s1, const char *s2) const
    {
        return strcmp(s1, s2) < 0;
    }
};

std::map<string, string> feedbackinfo;

class Object
{

public:
    Object(std::string n, std::ostringstream *s, bool a, int t)
        : name(n)
        , stream(s)
        , added(a)
        , timeStep(t)
    {
    }
    virtual ~Object()
    {
    }

    std::string name;
    std::ostringstream *stream;
    bool added;
    int timeStep;
};

class Label : public Object
{

public:
    Label(std::string name, std::ostringstream *s, float tx, float ty, float tz)
        : Object(name, s, true, -1)
        , x(tx)
        , y(ty)
        , z(tz)
    {
    }

    float x, y, z;
};

class RenderObject : public Object
{

public:
    RenderObject(std::string name,
                 const coDistributedObject *g, const coDistributedObject *n = NULL,
                 const coDistributedObject *c = NULL, const coDistributedObject *t = NULL,
                 int ts = -1)
        : Object(name, NULL, false, ts)
        , geometry(g)
        , normals(n)
        , colors(c)
        , texture(t)
    {
    }

    virtual ~RenderObject()
    {
        delete stream;
    }

    const coDistributedObject *geometry;
    const coDistributedObject *normals;
    const coDistributedObject *colors;
    const coDistributedObject *texture;
};

class WebGLRenderer
{
private:
    // for now, we only support per vertex data
    enum
    {
        CO_PER_VERTEX = 0,
        CO_PER_FACE,
        CO_NONE,
        CO_OVERALL
    };

    // static renderer callbacks
    static void quitCallback(void *userData, void *callbackData);
    static void addObjectCallback(void *userData, void *callbackData);
    static void deleteObjectCallback(void *userData, void *callbackData);
    static void renderCallback(void *userData, void *callbackData);
    static void masterSwitchCallback(void *userData, void *callbackData);
    static void paramCallback(bool inMapLoading, void *userData, void *callbackData);
    static void doCustomCallback(void *userData, void *callbackData);

    // instance renderer callbacks
    void quit(void *callbackData);
    void addObject(void *callbackData);
    void deleteObject(void *callbackData);
    void masterSwitch(void *callbackData);
    void render(void *callbackData);
    void param(bool inMapLoading, const char *paraName, void *callbackData);
    void doCustom(void *callbackData);

    void addObject(const coDistributedObject *geo, const coDistributedObject *col, const coDistributedObject *norm, const coDistributedObject *tex, int timeStep = -1);
    void deleteObject(const coDistributedObject *object);

    std::ostringstream *addGeometry(const coDistributedObject *geometry,
                                    const coDistributedObject *colors,
                                    const coDistributedObject *normals,
                                    const coDistributedObject *texture,
                                    const char *name,
                                    int timeStep);

    struct MHD_Daemon *daemon;

    // objects are represented by the name of their geometry node
    // in the deleteObject callback, objects are identified by the topmost
    // set they are contained in, so for removal we need to have a mapping
    // set -> objectnames
    std::map<const char *, std::vector<std::string> *, ltstr> groups;
    void addToGroup(const char *groupname, const char *objname);

    // name of the file to store the COVISE objects as XML
    std::string fileName;

    // map a file buffer (e.g. a png colormap) to a covise object
    // used for freeing buffers on deletion of an object
    void addFileBufferToObject(const char *name, unsigned char *buf);

    // free buffers associated with a covise object
    void deleteFileBuffers(const char *name);

    // mapping object name -> file buffers
    std::map<const char *, std::vector<unsigned char *> *, ltstr> objectFiles;

    // register a buffer as a file to be served by the webserver
    void registerBuffer(struct file f, const char *mimeType, ...);

    // register a file as a file to be served by the webserver
    void registerFile(const char *fileName, const char *mimeType, ...);

    void createImage(const char *name, const char *text);

public:
    WebGLRenderer(int argc, char *argv[]);
    ~WebGLRenderer();

    void run();

    // mutual exclusion for addObject and http handler
    OpenThreads::Mutex objectMutex;

    // condition if objects are deleted or added
    OpenThreads::Condition *changeCondition;
    OpenThreads::Mutex *changeMutex;

    // mapping object name -> object
    std::map<const char *, Object *, ltstr> objects;

    // objects get the same name when a module up the pipeline is executed
    // again, so we append a revision id to track changes
    int revisionID;
    std::map<const char *, const char *, ltstr> revName;

    // mapping filename -> file
    std::map<const char *, struct file, ltstr> files;

    // number of timesteps in the dataset
    int tMin;
    int tMax;
};

#endif
