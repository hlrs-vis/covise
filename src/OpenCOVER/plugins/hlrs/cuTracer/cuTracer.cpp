/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cover/coVR3DTransInteractor.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>

#include <PluginUtil/ColorBar.h>

#include <OpenVRUI/coMenuItem.h>
#include <cover/OpenCOVER.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/RenderObject.h>

#include <osg/Geode>
#include <osg/ref_ptr>
#include <osg/PointSprite>

#include <osgDB/FileUtils>
#include <osgDB/ReadFile>

#include <GL/glu.h>

#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coRowMenu.h>

#include <cover/coVRMSController.h>
#include <cover/coVRShader.h>

#include <cutil.h>
#include <cutil_math.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "bb.h"
#include "tree.h"
#include "utils.h"

#include "cuTracer.h"

static bool compute = true;
static bool create = true;

/*
static struct bb *flat;
static int *neighbors;
static float *bs;
static int *cells, *outCell, *outPart;

*/
static float3 *outPos;
static float *outVort;
static unsigned char *outVel;

static int *outCell, *outPart;
static float3 *pos;

static GLuint vertexVBO, velocityVBO, vortexVBO;
static struct cudaGraphicsResource *vertexResource, *velocityResource, *vortexResource;

void init_trace(const struct usg &usg, float3 *pos, int *outCell, float3 *outPos, unsigned char *outVel, int num, int ts);

void trace(const struct usg &usg, float3 *pos,
           const float periodic, int *outCell, float3 *outPos, int *outPart,
           unsigned char *outVel, int numParticles, int steps);

void getMinMax(const float *data, int numElem, float *min,
               float *max, float minV = -FLT_MAX, float maxV = FLT_MAX);

void insertNeighbor(struct usg *usg, int e0, int e1)
{

    int start0 = (usg->elementList)[e0];
    int start1 = (usg->elementList)[e1];

    int faces[6][4] = { { 0, 1, 4, 5 },
                        { 1, 2, 5, 6 },
                        { 2, 3, 6, 7 },
                        { 0, 3, 4, 7 },
                        { 0, 1, 2, 3 },
                        { 4, 5, 6, 7 } };

    bool sharedFace = false;
    for (int face = 0; face < 6; face++)
    {
        int sharedCorners = 0;
        for (int vertex = 0; vertex < 4; vertex++)
        {
            for (int corner = 0; corner < 8; corner++)
            {
                if (usg->cornerList[start0 + faces[face][vertex]] == usg->cornerList[start1 + corner])
                    sharedCorners++;
            }
        }
        if (sharedCorners == 4)
            usg->neighbors[e0 * 6 + face] = e1;
    }
}

void createNeighborList(struct usg *usg)
{

    // set of elements per corner
    std::set<int> *corner_elements = new std::set<int>[usg->numCorners];
    usg->neighbors = new int[usg->numElements * 6];
    for (int index = 0; index < usg->numElements; index++)
        for (int i = 0; i < 6; i++)
            usg->neighbors[index * 6 + i] = -1;

    for (int elem = 0; elem < usg->numElements; elem++)
    {

        int start = (usg->elementList)[elem];
        int end;
        if (elem < usg->numElements - 1)
            end = (usg->elementList)[elem + 1] - 1;
        else
            end = usg->numCorners - 1;

        for (int corner = start; corner <= end; corner++)
            corner_elements[usg->cornerList[corner]].insert(elem);
    }

    // if two elements share a corner, test if they share a face
    for (int corner = 0; corner < usg->numCorners; corner++)
    {

        std::set<int>::iterator e0, e1;
        for (e0 = corner_elements[corner].begin(); e0 != corner_elements[corner].end(); e0++)
            for (e1 = corner_elements[corner].begin(); e1 != corner_elements[corner].end(); e1++)
            {
                if (*e0 != *e1)
                    insertNeighbor(usg, *e0, *e1);
            }
    }
    //delete[] corner_elements;
    /*
   for (int elem = 0; elem < 10; elem ++) {
      printf("%05d %05d %05d %05d %05d %05d\n", (*usg->neighbors)[elem * 6],
             (*usg->neighbors)[elem * 6 + 1],
             (*usg->neighbors)[elem * 6 + 2],
             (*usg->neighbors)[elem * 6 + 3],
             (*usg->neighbors)[elem * 6 + 4],
             (*usg->neighbors)[elem * 6 + 5]);
   }
   printf("\n");
   */
}

void removeSpikesAdaptive(const float *data, int numElem,
                          float *min, float *max);

cuTracer::cuTracer()
: coVRPlugin(COVER_PLUGIN_NAME)
, menu(NULL)
{
}

cuTracer::~cuTracer()
{
    std::map<std::string, osg::Geode *>::iterator i;

    for (i = geode.begin(); i != geode.end(); i++)
    {
        geode.erase(i);
        cover->getObjectsRoot()->removeChild((*i).second);
    }
}

bool cuTracer::init()
{
    return true;
}

void cuTracer::removeObject(const char *objName, bool /*replace*/)
{
    minMax.erase(objName);

    std::map<std::string, osg::Group *>::iterator i = groups.find(std::string(objName));
    if (i != groups.end())
    {
        groups.erase(i);
        for (int index = 0; index < i->second->getNumChildren(); index++)
        {
            osg::Node *node = i->second->getChild(index);

            std::string name = node->getName();
            geode.erase(name);
            //delete(dynamic_cast<osg::Geode *>(node));
        }

        // TODO: delete group contents + drawable
        printf("objectsroot removechild [%s]\n", i->second->getName().c_str());
        cover->getObjectsRoot()->removeChild(i->second);
    }

    // erase interactor
    std::map<std::string, coVR3DTransInteractor *>::iterator pi = interactors.find(objName);
    if (pi != interactors.end())
    {
        pi->second->hide();
        interactors.erase(pi);
    }

    if (menu)
    {
        coMenuItem *item = menu->getItemByName(objName);
        if (item)
            menu->remove(item);
    }

    std::map<std::string, coRowMenu *>::iterator mi = menus.find(objName);
    if (mi != menus.end())
        menus.erase(mi);
}

void cuTracer::addObject(const RenderObject *container, osg::Group *, const RenderObject *geometry, const RenderObject *normals, const RenderObject *, const RenderObject *)
{
    /*
   const char * variant = container->getAttribute("VARIANT");
   if (!variant || strncmp(variant, "GPGPU", 5))
      return;
*/
    osg::Group *group = NULL;

    std::map<std::string, osg::Group *>::iterator gi = groups.find(container->getName());
    if (gi == groups.end())
    {
        group = new osg::Group();
        group->setName(container->getName());
        printf("objectsroot addchild [%s]\n", container->getName());
        cover->getObjectsRoot()->addChild(group);
        groups[container->getName()] = group;
    }
    else
        group = gi->second;

    if (geometry && geometry->isType("UNSGRD"))
    {

        if (normals)
        {
            float *vx = NULL, *vy = NULL, *vz = NULL;
            normals->getAddresses(vx, vy, vz);

            osg::ref_ptr<osg::Geode> g = new osg::Geode();

            osg::Texture2D *texture = new osg::Texture2D;
            osg::Image *img = osgDB::readImageFile("arrow.png");
            texture->setImage(img);

            coVRShader *shader = coVRShaderList::instance()->get("particlepoints");

            osg::StateSet *state = g->getOrCreateStateSet();
            state->setGlobalDefaults();

            state->setMode(GL_VERTEX_PROGRAM_POINT_SIZE, osg::StateAttribute::ON);

            state->setTextureAttributeAndModes(0, texture, osg::StateAttribute::ON);

            state->setTextureAttributeAndModes(0, new osg::PointSprite(),
                                               osg::StateAttribute::ON);
            state->setDataVariance(osg::Object::DYNAMIC);
            state->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
            state->setRenderBinDetails(3, "RenderBin");

            char name[256];
            snprintf(name, 256, "%s_%s", container->getName(),
                     geometry->getName());

            coVR3DTransInteractor *inter = new coVR3DTransInteractor(osg::Vec3(0.0, 0.0, 0.0), cover->getSceneSize() / 50.0, coInteraction::ButtonA, "hand", "Plane_S0", coInteraction::Medium);
            inter->show();
            inter->enableIntersection();
            interactors[container->getName()] = inter;

            osg::ref_ptr<TracerDrawable> draw = new TracerDrawable(geometry, normals, inter);

            geode[std::string(name)] = g.get();
            g->setName(strdup(name));
            g->addDrawable(draw.get());
            draw->setUseDisplayList(false);
            shader->apply(g);

            group->addChild(g.get());
            printf(" group [%s] addchild [%p]\n", g->getName().c_str(), name);
        }
        else
            cerr << "no/wrong data received" << endl;
    }
}

void cuTracer::preFrame()
{
    std::map<std::string, osg::Geode *>::iterator i;
    for (i = geode.begin(); i != geode.end(); i++)
    {
        TracerDrawable *drawable = dynamic_cast<TracerDrawable *>(i->second->getDrawable(0));
        if (drawable)
            drawable->preFrame();
    }
}

void cuTracer::postFrame()
{
    std::map<std::string, osg::Geode *>::iterator i;
    for (i = geode.begin(); i != geode.end(); i++)
    {
        TracerDrawable *drawable = dynamic_cast<TracerDrawable *>(i->second->getDrawable(0));
        if (drawable)
            drawable->postFrame();
    }
}

TracerDrawable::TracerDrawable(RenderObject *g, RenderObject *vel,
                               coVR3DTransInteractor *inter)
    : osg::Geometry()
    , geom(g)
    , interactor(inter)
    , step(0)
    , initialized(false)
{
    if (geom && geom->isType("UNSGRD"))
    {

        int *tl;
        struct usg usg;

        geom->getSize(usg.numElements, usg.numCorners, usg.numPoints);
        geom->getAddresses(usg.x, usg.y, usg.z, usg.cornerList, usg.elementList,
                           tl);

        usg.boundingSpheres = new float[usg.numElements * 4];

        for (int index = 0; index < usg.numElements; index++)
        {

            int begin = usg.elementList[index];
            int end;
            if (index < usg.numElements - 1)
                end = usg.elementList[index + 1] - 1;
            else
                end = usg.numCorners - 1;
            osg::Vec3 m;
            float dist = 0.0;

            for (int i = begin; i <= end; i++)
            {

                osg::Vec3 a(usg.x[usg.cornerList[i]],
                            usg.y[usg.cornerList[i]],
                            usg.z[usg.cornerList[i]]);

                for (int j = begin; j <= end; j++)
                {

                    if (i != j)
                    {
                        osg::Vec3 b(usg.x[usg.cornerList[j]],
                                    usg.y[usg.cornerList[j]],
                                    usg.z[usg.cornerList[j]]);
                        float d = (a - b).length();
                        if (d > dist)
                        {
                            dist = d;
                            m = (a + b) / 2;
                        }
                    }
                }
            }
            usg.boundingSpheres[index * 4] = m.x();
            usg.boundingSpheres[index * 4 + 1] = m.y();
            usg.boundingSpheres[index * 4 + 2] = m.z();
            usg.boundingSpheres[index * 4 + 3] = dist;
        }

        const char *treeName = g->getAttribute("KDTREE");
        const char *neighborName = g->getAttribute("NEIGHBOR");

        struct stat fbuf;

        if (treeName && stat(treeName, &fbuf) == 0)
        {

            int fd;
            fprintf(stderr, "cuTracer: reading kdtree [%s]\n", treeName);
            if ((fd = open(treeName, O_RDONLY)) != -1)
            {
                read(fd, &usg.numFlat, sizeof(unsigned int));
                usg.flat = new BB[usg.numFlat];
                read(fd, usg.flat, usg.numFlat * sizeof(BB));
                read(fd, &usg.numCells, sizeof(unsigned int));
                usg.cells = new int[usg.numCells];
                read(fd, usg.cells, usg.numCells * sizeof(int));
                close(fd);
            }
            else
                fprintf(stderr, "cuTracer: reading kdtree [%s] failed\n", treeName);
        }
        else
        {

            Tree tree(&usg);
            tree.build();

            std::vector<BB> flat;
            std::vector<int> cells;
            tree.flatten(flat, cells);

            usg.cells = new int[cells.size()];
            usg.numCells = cells.size();
            memcpy(usg.cells, &(cells[0]), cells.size() * sizeof(int));

            usg.flat = new BB[flat.size()];
            usg.numFlat = flat.size();
            memcpy(usg.flat, &(flat[0]), flat.size() * sizeof(BB));

            if (treeName)
            {
                int fd;
                fprintf(stderr, "cuTracer: writing kdtree [%s]\n", treeName);
                if ((fd = open(treeName, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP)) != -1)
                {
                    write(fd, &usg.numFlat, sizeof(unsigned int));
                    write(fd, usg.flat, usg.numFlat * sizeof(BB));
                    write(fd, &usg.numCells, sizeof(unsigned int));
                    write(fd, usg.cells, usg.numCells * sizeof(int));
                    close(fd);
                }
                else
                    fprintf(stderr, "cuTracer: writing kdtree [%s] failed\n",
                            treeName);
            }
        }

        if (neighborName && stat(neighborName, &fbuf) == 0)
        {

            int fd;
            fprintf(stderr, "cuTracer: reading neighbor list [%s]\n", neighborName);
            if ((fd = open(neighborName, O_RDONLY)) != -1)
            {
                int numNeighbors;
                read(fd, &numNeighbors, sizeof(int));
                usg.neighbors = new int[numNeighbors];
                read(fd, usg.neighbors, numNeighbors * sizeof(int));
                close(fd);
            }
            else
                fprintf(stderr, "cuTracer: reading neighbor list [%s] failed\n",
                        neighborName);
        }
        else
        {

            fprintf(stderr, "cuTracer: creating neighbor list\n");
            createNeighborList(&usg);
            fprintf(stderr, "cuTracer: creating neighbor list done\n");

            if (neighborName)
            {
                int fd;
                if ((fd = open(neighborName, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP)) != -1)
                {
                    int numNeighbors = usg.numElements * 6;
                    write(fd, &numNeighbors, sizeof(int));
                    write(fd, usg.neighbors, numNeighbors * sizeof(int));
                    close(fd);
                }
                else
                    fprintf(stderr, "cuTracer: writing neighbor list [%s] failed\n",
                            treeName);
            }
        }

        if (vel)
            vel->getAddresses(usg.vx, usg.vy, usg.vz);

        const char *velName = NULL;

        if (vel)
            velName = vel->getName();

        numParticles = 1024;
        numSteps = 512;

        if (!glew_init())
        {
            printf("#GLEW initialization failed\n");
        }

        cudaGLSetGLDevice(0);
        cudaError_t err = cudaGetLastError();

        CUDA_SAFE_CALL(cudaMalloc((void **)&cuda_usg.elementList,
                                  usg.numElements * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&cuda_usg.cornerList,
                                  usg.numCorners * sizeof(int)));

        CUDA_SAFE_CALL(cudaMalloc((void **)&cuda_usg.x,
                                  usg.numPoints * sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&cuda_usg.y,
                                  usg.numPoints * sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&cuda_usg.z,
                                  usg.numPoints * sizeof(float)));

        CUDA_SAFE_CALL(cudaMalloc((void **)&cuda_usg.vx,
                                  usg.numPoints * sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&cuda_usg.vy,
                                  usg.numPoints * sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&cuda_usg.vz,
                                  usg.numPoints * sizeof(float)));

        CUDA_SAFE_CALL(cudaMalloc((void **)&cuda_usg.boundingSpheres,
                                  usg.numElements * sizeof(float) * 4));

        CUDA_SAFE_CALL(cudaMalloc((void **)&cuda_usg.flat,
                                  usg.numFlat * sizeof(BB)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&cuda_usg.cells,
                                  usg.numCells * sizeof(int)));

        CUDA_SAFE_CALL(cudaMalloc((void **)&pos,
                                  numParticles * sizeof(float3)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&outCell,
                                  numParticles * sizeof(int)));

        CUDA_SAFE_CALL(cudaMalloc((void **)&outPart, numParticles * sizeof(int)));

        CUDA_SAFE_CALL(cudaMalloc((void **)&cuda_usg.flat,
                                  usg.numFlat * sizeof(BB)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&cuda_usg.cells,
                                  usg.numCells * sizeof(int)));

        if (usg.neighbors)
        {
            CUDA_SAFE_CALL(cudaMalloc((void **)&cuda_usg.neighbors,
                                      usg.numElements * 6 * sizeof(int)));
            CUDA_SAFE_CALL(cudaMemcpy((void *)cuda_usg.neighbors, usg.neighbors,
                                      usg.numElements * 6 * sizeof(int),
                                      cudaMemcpyHostToDevice));
        }

        CUDA_SAFE_CALL(cudaMemcpy((void *)cuda_usg.elementList, usg.elementList,
                                  usg.numElements * sizeof(int),
                                  cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMemcpy((void *)cuda_usg.cornerList, usg.cornerList,
                                  usg.numCorners * sizeof(int),
                                  cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMemcpy((void *)cuda_usg.x, usg.x, usg.numPoints * sizeof(float),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy((void *)cuda_usg.y, usg.y, usg.numPoints * sizeof(float),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy((void *)cuda_usg.z, usg.z, usg.numPoints * sizeof(float),
                                  cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMemcpy((void *)cuda_usg.vx, usg.vx, usg.numPoints * sizeof(float),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy((void *)cuda_usg.vy, usg.vy, usg.numPoints * sizeof(float),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy((void *)cuda_usg.vz, usg.vz, usg.numPoints * sizeof(float),
                                  cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMemcpy((void *)cuda_usg.boundingSpheres,
                                  usg.boundingSpheres,
                                  usg.numElements * sizeof(float) * 4,
                                  cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMemcpy((void *)cuda_usg.flat, usg.flat,
                                  usg.numFlat * sizeof(BB),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy((void *)cuda_usg.cells, usg.cells,
                                  usg.numCells * sizeof(int),
                                  cudaMemcpyHostToDevice));

        BB bbox = usg.flat[0];
        float wx = bbox.maxx - bbox.minx;
        float wy = bbox.maxy - bbox.miny;
        float wz = bbox.maxz - bbox.minz;

        printf("bbox: [%f %f %f %f %f %f]\n", bbox.minx, bbox.miny, bbox.minz, bbox.maxx, bbox.maxy, bbox.maxz);

        startPos = (float3 *)malloc(numParticles * sizeof(float3));
        // select random point inside random cell as a starting poinz
        for (int index = 0; index < numParticles; index++)
        {

            float b[6] = { FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };

            float x, y, z;
            int cell = usg.elementList[rand() % usg.numElements];
            for (int i = 0; i < 8; i++)
            {

                int v = usg.cornerList[cell + i];
                x = usg.x[v];
                y = usg.y[v];
                z = usg.z[v];
                if (x < b[0])
                    b[0] = x;
                if (y < b[1])
                    b[1] = y;
                if (z < b[2])
                    b[2] = z;
                if (x > b[3])
                    b[3] = x;
                if (y > b[4])
                    b[4] = y;
                if (z > b[5])
                    b[5] = z;
            }

            startPos[index].x = ((b[3] - b[0]) * rand() / (RAND_MAX + 1.0)) + b[0];
            startPos[index].y = ((b[4] - b[1]) * rand() / (RAND_MAX + 1.0)) + b[1];
            startPos[index].z = ((b[5] - b[2]) * rand() / (RAND_MAX + 1.0)) + b[2];
        }
        CUDA_SAFE_CALL(cudaMemcpy((void *)pos, startPos, numParticles * sizeof(float3),
                                  cudaMemcpyHostToDevice));

        box = osg::BoundingBox(bbox.minx, bbox.miny, bbox.minz,
                               bbox.maxx, bbox.maxy, bbox.maxz);

        name = geom->getName();
        /*
      delete[] usg.boundingSpheres;
      delete[] usg.flat;
      delete[] usg.cells;
      delete[] usg.neighbors;
      */
    }
}

TracerDrawable::~TracerDrawable()
{
    printf("~TracerDrawable [%s]\n", name.c_str());
    if (tex)
    {
        delete texData;
        glDeleteTextures(1, &texture);
    }
}

void TracerDrawable::preFrame()
{
    if (interactor)
        interactor->preFrame();
}

void TracerDrawable::postFrame()
{
}

TracerDrawable::TracerDrawable(const TracerDrawable &draw,
                               const osg::CopyOp &op)
    : osg::Geometry(draw, op)
{
}

void TracerDrawable::drawImplementation(osg::RenderInfo & /*info*/) const
{
    size_t s;

    if (create)
    {

        create = false;
        createVBO(&vertexVBO, numParticles * numSteps * sizeof(float3));

        CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&vertexResource, vertexVBO,
                                                    cudaGraphicsMapFlagsNone));

        createVBO(&velocityVBO, numParticles * numSteps * 3 * sizeof(unsigned char));

        CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&velocityResource, velocityVBO,
                                                    cudaGraphicsMapFlagsNone));

        createVBO(&vortexVBO, numParticles * numSteps * sizeof(float));

        CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&vortexResource, vortexVBO,
                                                    cudaGraphicsMapFlagsNone));
    }

    if (!initialized || (interactor && interactor->wasStopped()))
    {

        initialized = true;
        osg::Vec3 p(0.0, 0.0, 0.0);
        if (interactor)
            p = interactor->getPos();

        float px = p.x();
        float py = p.y();
        float pz = p.z();
        for (int index = 0; index < numParticles; index++)
        {

            float s = 2 * M_PI * (rand() / (RAND_MAX + 1.0));
            float t = 2 * M_PI * (rand() / (RAND_MAX + 1.0));

            float r = rand() / (RAND_MAX + 1.0) / 2.0;

            startPos[index].x = px + r * sin(s) * cos(t);
            startPos[index].y = py + r * sin(s) * sin(t);
            startPos[index].z = pz + r * cos(t);
        }

        CUDA_SAFE_CALL(cudaMemcpy((void *)pos, startPos, numParticles * sizeof(float3),
                                  cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &vertexResource, 0));
        CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &velocityResource, 0));
        CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &vortexResource, 0));

        CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)&outPos, &s,
                                                            vertexResource));
        CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)&outVel, &s,
                                                            velocityResource));
        CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)&outVort, &s,
                                                            vortexResource));

        CUDA_SAFE_CALL(cudaMemset((void *)outCell, -1, numParticles * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset((void *)outPart, 0, numParticles * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset((void *)outPos, 0, numParticles * numSteps * sizeof(float3)));
        CUDA_SAFE_CALL(cudaMemset((void *)outVort, 0, numParticles * numSteps * sizeof(float)));

        init_trace(cuda_usg, pos, outCell, outPos, outVel, numParticles, numSteps);

        CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &vertexResource, 0));
        CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &velocityResource, 0));
        CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &vortexResource, 0));

        compute = true;
        stepsComputed = 0;
        step = 0;
    }

    if (compute)
    {

        CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &vertexResource, 0));
        CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &velocityResource, 0));
        CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &vortexResource, 0));

        CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)&outPos, &s,
                                                            vertexResource));
        CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)&outVel, &s,
                                                            velocityResource));
        CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)&outVort, &s,
                                                            vortexResource));

        trace(cuda_usg, pos, 0.0, outCell, outPos, outPart, outVel,
              numParticles, numSteps);

        cudaThreadSynchronize();

        CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &vertexResource, 0));
        CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &velocityResource, 0));
        CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &vortexResource, 0));

        compute = false;
    }

    renderVBO(vertexVBO, velocityVBO, vortexVBO, step, numParticles, 16);

    step++;
    if (step >= (numSteps - 16))
        step = 0;
}

osg::BoundingBox TracerDrawable::computeBound()
{
    return box;
}

osg::Object *TracerDrawable::cloneType() const
{
    return new TracerDrawable(NULL, NULL, NULL);
}

osg::Object *TracerDrawable::clone(const osg::CopyOp &op) const
{
    return new TracerDrawable(*this, op);
}

COVERPLUGIN(cuTracer)
