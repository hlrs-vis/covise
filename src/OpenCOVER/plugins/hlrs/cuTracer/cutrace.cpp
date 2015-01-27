/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>

#include <osg/BlendFunc>
#include <osg/Depth>
#include <osg/TexEnv>
#include <osg/PointSprite>
#include <osg/GL2Extensions>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/Image>
#include <osg/PolygonMode>
#include <osg/Shader>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Texture2D>

#include <osgDB/FileUtils>
#include <osgDB/ReadFile>
#include <osgGA/TrackballManipulator>
#include <osgViewer/Viewer>

#include "covReadFiles.h"

#include "kd_tree.h"

#include <cutil.h>
#include <cutil_math.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <vector>
#include <string>

#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "bb.h"
#include "utils.h"
#include "drawable.h"

int *el, *cl;
float *x, *y, *z, *vx, *vy, *vz;

struct bb *flat;
int *cells;
float3 *pos;
int *outCell;
float3 *outPos;
unsigned char *outVel;

int *neighbors;

bool loadShaderSource(osg::Shader *obj, const std::string &fileName)
{

    std::string fqFileName = osgDB::findDataFile(fileName);
    if (fqFileName.length() == 0)
    {
        std::cout << "File \"" << fileName << "\" not found." << std::endl;
        return false;
    }
    bool success = obj->loadShaderSourceFromFile(fqFileName.c_str());
    if (!success)
    {
        std::cout << "Couldn't load file: " << fileName << std::endl;
        return false;
    }
    else
        return true;
}

void cu(const struct bb *flat, const int *cells, const int *neighbors,
        const int *el, const int *cl,
        const float *x, const float *y, const float *z,
        const float *vx, const float *vy, const float *vz,
        float3 *pos, int *outCell, float3 *outPos, unsigned char *outVel,
        int num, int ts);

void checkErr(const char *where)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s [%s]\n", cudaGetErrorString(err), where);
        exit(-1);
    }
}

osg::Group *cuda_init_usg(const struct usg &usg)
{
    init();
    int numTraces = 4096;
    int timeSteps = 256;

    cudaGLSetGLDevice(0);

    checkErr("init");

    printf("numelem: %d\n", usg.numElements);
    CUDA_SAFE_CALL(cudaMalloc((void **)&el, usg.numElements * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&cl, usg.numCorners * sizeof(int)));

    CUDA_SAFE_CALL(cudaMalloc((void **)&x, usg.numPoints * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&y, usg.numPoints * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&z, usg.numPoints * sizeof(float)));

    CUDA_SAFE_CALL(cudaMalloc((void **)&vx, usg.numPoints * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&vy, usg.numPoints * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&vz, usg.numPoints * sizeof(float)));

    CUDA_SAFE_CALL(cudaMalloc((void **)&pos, numTraces * sizeof(float3)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&outCell, numTraces * sizeof(int)));

    GLuint vertexVBO, velocityVBO;
    struct cudaGraphicsResource *vertexResource, *velocityResource;
    size_t s;

    createVBO(&vertexVBO, numTraces * timeSteps * sizeof(float3));
    CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&vertexResource, vertexVBO,
                                                cudaGraphicsMapFlagsNone));
    CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &vertexResource, 0));
    CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)&outPos, &s,
                                                        vertexResource));

    createVBO(&velocityVBO, numTraces * timeSteps * 3 * sizeof(unsigned char));
    CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&velocityResource, velocityVBO,
                                                cudaGraphicsMapFlagsNone));
    CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &velocityResource, 0));
    CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)&outVel, &s,
                                                        velocityResource));

    CUDA_SAFE_CALL(cudaMemset((void *)outCell, -1, numTraces * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset((void *)outPos, 0, numTraces * timeSteps * sizeof(float3)));

    CUDA_SAFE_CALL(cudaMemcpy((void *)el, usg.elementList,
                              usg.numElements * sizeof(int),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy((void *)cl, usg.cornerList,
                              usg.numCorners * sizeof(int),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy((void *)x, usg.x, usg.numPoints * sizeof(float),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy((void *)y, usg.y, usg.numPoints * sizeof(float),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy((void *)z, usg.z, usg.numPoints * sizeof(float),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy((void *)vx, usg.vx, usg.numPoints * sizeof(float),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy((void *)vy, usg.vy, usg.numPoints * sizeof(float),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy((void *)vz, usg.vz, usg.numPoints * sizeof(float),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc((void **)&flat,
                              usg.flat->size() * sizeof(struct bb)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&cells,
                              usg.cells->size() * sizeof(int)));

    CUDA_SAFE_CALL(cudaMalloc((void **)&neighbors,
                              usg.neighbors->size() * sizeof(int)));

    CUDA_SAFE_CALL(cudaMemcpy((void *)flat, &((*usg.flat)[0]),
                              usg.flat->size() * sizeof(struct bb),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy((void *)cells, &((*usg.cells)[0]),
                              usg.cells->size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy((void *)neighbors, &((*usg.neighbors)[0]),
                              usg.neighbors->size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    struct bb bbox = (*(usg.flat))[0];
    float wx = bbox.maxx - bbox.minx;
    float wy = bbox.maxy - bbox.miny;
    float wz = bbox.maxz - bbox.minz;

    float3 *p = (float3 *)malloc(numTraces * sizeof(float3));
    for (int index = 0; index < numTraces; index++)
    {

        p[index].x = (wx * rand() / (RAND_MAX + 1.0)) + bbox.minx;
        p[index].y = (wy * rand() / (RAND_MAX + 1.0)) + bbox.miny;
        p[index].z = (wz * rand() / (RAND_MAX + 1.0)) + bbox.minz;
    }
    /*
   p[0].x = 30.0;
   p[0].y = 6.0;
   p[0].z = 1.0;

   p[1].x = 28.0;
   p[1].y = 6.0;
   p[1].z = 1.0;
   */

    CUDA_SAFE_CALL(cudaMemcpy((void *)pos, p, numTraces * sizeof(float3),
                              cudaMemcpyHostToDevice));
    timeval t0, t1;
    gettimeofday(&t0, NULL);

    cu(flat, cells, neighbors, el, cl, x, y, z, vx, vy, vz, pos,
       outCell, outPos, outVel, numTraces, timeSteps);
    cudaThreadSynchronize();

    gettimeofday(&t1, NULL);

    long long usec = ((long long)(t1.tv_sec - t0.tv_sec)) * 1000000 + (t1.tv_usec - t0.tv_usec);

    printf("   time: %lld usec\n", usec);

    CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &vertexResource, 0));
    CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &velocityResource, 0));

    /*
   float *outV = (float *) malloc(numTraces * timeSteps * sizeof(float));
   CUDA_SAFE_CALL(cudaMemcpy((void *) outV, outVel,
                             numTraces * timeSteps * sizeof(float),
                             cudaMemcpyDeviceToHost));
   
   for (int index = 0; index < 10; index ++)
      printf(" vel %f\n", outV[index]);
*/
    osg::Program *shader = new osg::Program;

    osg::Shader *vs = new osg::Shader(osg::Shader::VERTEX);
    osg::Shader *fs = new osg::Shader(osg::Shader::FRAGMENT);

    loadShaderSource(vs, "particles.vert");
    loadShaderSource(fs, "particles.frag");

    shader->addShader(vs);
    shader->addShader(fs);

    osg::Texture2D *texture = new osg::Texture2D;
    osg::Image *img = osgDB::readImageFile("particle.png");
    texture->setImage(img);

    osg::Group *group = new osg::Group();
    osg::ref_ptr<Drawable> draw = new Drawable(&bbox, vertexVBO, velocityVBO, numTraces * timeSteps);
    osg::ref_ptr<osg::Geode> g = new osg::Geode();

    g->addDrawable(draw.get());
    draw->setUseDisplayList(false);
    osg::StateSet *state = g->getOrCreateStateSet();
    state->setAttributeAndModes(shader, osg::StateAttribute::ON);
    state->setMode(GL_VERTEX_PROGRAM_POINT_SIZE, osg::StateAttribute::ON);
    state->setTextureAttributeAndModes(0, texture, osg::StateAttribute::ON);
    state->setTextureAttributeAndModes(0, new osg::PointSprite(),
                                       osg::StateAttribute::ON);
    group->addChild(g.get());

    osg::Box *box = new osg::Box(osg::Vec3((bbox.maxx + bbox.minx) / 2.0,
                                           (bbox.maxy + bbox.miny) / 2.0,
                                           (bbox.maxz + bbox.minz) / 2.0),
                                 bbox.maxx - bbox.minx, bbox.maxy - bbox.miny, bbox.maxz - bbox.minz);

    osg::ShapeDrawable *boxDrawable = new osg::ShapeDrawable(box);
    g = new osg::Geode();
    g->addDrawable(boxDrawable);

    state = g->getOrCreateStateSet();
    osg::PolygonMode *polyModeObj = dynamic_cast<osg::PolygonMode *>(state->getAttribute(osg::StateAttribute::POLYGONMODE));
    if (!polyModeObj)
    {
        polyModeObj = new osg::PolygonMode;
        state->setAttribute(polyModeObj);
    }
    polyModeObj->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
    group->addChild(g);

    return group;
}
/*
float lerp(const float a, const float b, const float t) {
   
   return a * (1 - t) + b * t;
}
*/
osg::Vec3 lerp(const osg::Vec3 &a, const osg::Vec3 &b, const float t)
{

    return osg::Vec3(lerp(a.x(), b.x(), t),
                     lerp(a.y(), b.y(), t),
                     lerp(a.z(), b.z(), t));
}

osg::Vec3 *inside(const struct usg *usg, const int element, const float x, const float y, const float z)
{

    /*
   int begin = usg.elementList[element];
   int end;
   if (index < usg.numElements - 1)
      end = usg.elementList[element + 1] - 1;
   else
      end = usg.numCorners - 1;
*/
    osg::Vec3 v[8];

    int e = usg->elementList[element];

    v[0] = osg::Vec3(usg->x[usg->cornerList[e + 0]],
                     usg->y[usg->cornerList[e + 0]],
                     usg->z[usg->cornerList[e + 0]]);
    v[1] = osg::Vec3(usg->x[usg->cornerList[e + 1]],
                     usg->y[usg->cornerList[e + 1]],
                     usg->z[usg->cornerList[e + 1]]);
    v[2] = osg::Vec3(usg->x[usg->cornerList[e + 2]],
                     usg->y[usg->cornerList[e + 2]],
                     usg->z[usg->cornerList[e + 2]]);
    v[3] = osg::Vec3(usg->x[usg->cornerList[e + 3]],
                     usg->y[usg->cornerList[e + 3]],
                     usg->z[usg->cornerList[e + 3]]);
    v[4] = osg::Vec3(usg->x[usg->cornerList[e + 4]],
                     usg->y[usg->cornerList[e + 4]],
                     usg->z[usg->cornerList[e + 4]]);
    v[5] = osg::Vec3(usg->x[usg->cornerList[e + 5]],
                     usg->y[usg->cornerList[e + 5]],
                     usg->z[usg->cornerList[e + 5]]);
    v[6] = osg::Vec3(usg->x[usg->cornerList[e + 6]],
                     usg->y[usg->cornerList[e + 6]],
                     usg->z[usg->cornerList[e + 6]]);
    v[7] = osg::Vec3(usg->x[usg->cornerList[e + 7]],
                     usg->y[usg->cornerList[e + 7]],
                     usg->z[usg->cornerList[e + 7]]);
    /*
   printf("  points\n");
   for (int index = 0; index < 8; index ++)
      printf(" (%f %f %f)", v[index].x(), v[index].y(), v[index].z());
   printf("\n");
*/
    osg::Vec3 n[6];
    n[0] = (v[1] - v[0]) ^ (v[4] - v[0]);
    n[1] = (v[2] - v[1]) ^ (v[5] - v[1]);
    n[2] = (v[3] - v[2]) ^ (v[6] - v[2]);
    n[3] = (v[0] - v[3]) ^ (v[7] - v[3]);
    n[4] = (v[3] - v[0]) ^ (v[1] - v[0]);
    n[5] = (v[5] - v[4]) ^ (v[7] - v[4]);

    /*
   printf("  normals:\n");
   for (int index = 0; index < 6; index ++)
      printf(" (%f %f %f)", n[index].x(), n[index].y(), n[index].z());
   printf("\n");
*/
    osg::Vec3 p(x, y, z);

    float s0 = n[0] * (p - v[0]);
    float s1 = n[1] * (p - v[1]);
    float s2 = n[2] * (p - v[2]);
    float s3 = n[3] * (p - v[7]);
    float s4 = n[4] * (p - v[0]);
    float s5 = n[5] * (p - v[5]);

    // printf("%d (%f %f %f): %f %f %f %f %f %f\n", element, x, y, z, s0, s1, s2, s3, s4, s5);
    if (s0 <= 0 && s1 <= 0 && s2 <= 0 && s3 <= 0 && s4 <= 0 && s5 <= 0)
    {

        // trilinear interpolation
        osg::Vec3 *result;
        osg::Vec3 d1(usg->vx[usg->cornerList[e + 0]],
                     usg->vy[usg->cornerList[e + 0]],
                     usg->vz[usg->cornerList[e + 0]]);

        float t0 = (x - v[0].x()) / (v[1].x() - v[0].x());
        float t1 = (y - v[0].y()) / (v[2].y() - v[0].y());
        float t2 = (z - v[0].z()) / (v[4].z() - v[0].x());

        osg::Vec3 d[8];
        d[0] = osg::Vec3(usg->vx[usg->cornerList[e + 0]],
                         usg->vy[usg->cornerList[e + 0]],
                         usg->vz[usg->cornerList[e + 0]]);
        d[1] = osg::Vec3(usg->vx[usg->cornerList[e + 1]],
                         usg->vy[usg->cornerList[e + 1]],
                         usg->vz[usg->cornerList[e + 1]]);
        d[2] = osg::Vec3(usg->vx[usg->cornerList[e + 2]],
                         usg->vy[usg->cornerList[e + 2]],
                         usg->vz[usg->cornerList[e + 2]]);
        d[3] = osg::Vec3(usg->vx[usg->cornerList[e + 3]],
                         usg->vy[usg->cornerList[e + 3]],
                         usg->vz[usg->cornerList[e + 3]]);
        d[4] = osg::Vec3(usg->vx[usg->cornerList[e + 4]],
                         usg->vy[usg->cornerList[e + 4]],
                         usg->vz[usg->cornerList[e + 4]]);
        d[5] = osg::Vec3(usg->vx[usg->cornerList[e + 5]],
                         usg->vy[usg->cornerList[e + 5]],
                         usg->vz[usg->cornerList[e + 5]]);
        d[6] = osg::Vec3(usg->vx[usg->cornerList[e + 6]],
                         usg->vy[usg->cornerList[e + 6]],
                         usg->vz[usg->cornerList[e + 6]]);
        d[7] = osg::Vec3(usg->vx[usg->cornerList[e + 7]],
                         usg->vy[usg->cornerList[e + 7]],
                         usg->vz[usg->cornerList[e + 7]]);

        osg::Vec3 d01 = lerp(d[0], d[1], t0);
        osg::Vec3 d23 = lerp(d[3], d[2], t0);
        osg::Vec3 a = lerp(d01, d23, t1);
        osg::Vec3 d45 = lerp(d[4], d[5], t0);
        osg::Vec3 d76 = lerp(d[7], d[6], t0);
        osg::Vec3 b = lerp(d45, d76, t1);

        return new osg::Vec3(lerp(a, b, t2));
    }

    return NULL;
}

bool inside(const struct usg *usg, const int element, const float x, const float y, const float z, osg::Vec3 &vel)
{

    osg::Vec3 v[8];

    int e = usg->elementList[element];

    v[0] = osg::Vec3(usg->x[usg->cornerList[e + 0]],
                     usg->y[usg->cornerList[e + 0]],
                     usg->z[usg->cornerList[e + 0]]);
    v[1] = osg::Vec3(usg->x[usg->cornerList[e + 1]],
                     usg->y[usg->cornerList[e + 1]],
                     usg->z[usg->cornerList[e + 1]]);
    v[2] = osg::Vec3(usg->x[usg->cornerList[e + 2]],
                     usg->y[usg->cornerList[e + 2]],
                     usg->z[usg->cornerList[e + 2]]);
    v[3] = osg::Vec3(usg->x[usg->cornerList[e + 3]],
                     usg->y[usg->cornerList[e + 3]],
                     usg->z[usg->cornerList[e + 3]]);
    v[4] = osg::Vec3(usg->x[usg->cornerList[e + 4]],
                     usg->y[usg->cornerList[e + 4]],
                     usg->z[usg->cornerList[e + 4]]);
    v[5] = osg::Vec3(usg->x[usg->cornerList[e + 5]],
                     usg->y[usg->cornerList[e + 5]],
                     usg->z[usg->cornerList[e + 5]]);
    v[6] = osg::Vec3(usg->x[usg->cornerList[e + 6]],
                     usg->y[usg->cornerList[e + 6]],
                     usg->z[usg->cornerList[e + 6]]);
    v[7] = osg::Vec3(usg->x[usg->cornerList[e + 7]],
                     usg->y[usg->cornerList[e + 7]],
                     usg->z[usg->cornerList[e + 7]]);
    /*
   printf("  points\n");
   for (int index = 0; index < 8; index ++)
      printf(" (%f %f %f)", v[index].x(), v[index].y(), v[index].z());
   printf("\n");
*/
    osg::Vec3 n[6];
    n[0] = (v[1] - v[0]) ^ (v[4] - v[0]);
    n[1] = (v[2] - v[1]) ^ (v[5] - v[1]);
    n[2] = (v[3] - v[2]) ^ (v[6] - v[2]);
    n[3] = (v[0] - v[3]) ^ (v[7] - v[3]);
    n[4] = (v[3] - v[0]) ^ (v[1] - v[0]);
    n[5] = (v[5] - v[4]) ^ (v[7] - v[4]);

    /*
   printf("  normals:\n");
   for (int index = 0; index < 6; index ++)
      printf(" (%f %f %f)", n[index].x(), n[index].y(), n[index].z());
   printf("\n");
*/
    osg::Vec3 p(x, y, z);

    float s0 = n[0] * (p - v[0]);
    float s1 = n[1] * (p - v[1]);
    float s2 = n[2] * (p - v[2]);
    float s3 = n[3] * (p - v[7]);
    float s4 = n[4] * (p - v[0]);
    float s5 = n[5] * (p - v[5]);

    // printf("%d (%f %f %f): %f %f %f %f %f %f\n", element, x, y, z, s0, s1, s2, s3, s4, s5);
    if (s0 <= 0 && s1 <= 0 && s2 <= 0 && s3 <= 0 && s4 <= 0 && s5 <= 0)
    {

        // trilinear interpolation
        osg::Vec3 d1(usg->vx[usg->cornerList[e + 0]],
                     usg->vy[usg->cornerList[e + 0]],
                     usg->vz[usg->cornerList[e + 0]]);

        float t0 = (x - v[0].x()) / (v[1].x() - v[0].x());
        float t1 = (y - v[0].y()) / (v[2].y() - v[0].y());
        float t2 = (z - v[0].z()) / (v[4].z() - v[0].x());

        osg::Vec3 d[8];
        d[0] = osg::Vec3(usg->vx[usg->cornerList[e + 0]],
                         usg->vy[usg->cornerList[e + 0]],
                         usg->vz[usg->cornerList[e + 0]]);
        d[1] = osg::Vec3(usg->vx[usg->cornerList[e + 1]],
                         usg->vy[usg->cornerList[e + 1]],
                         usg->vz[usg->cornerList[e + 1]]);
        d[2] = osg::Vec3(usg->vx[usg->cornerList[e + 2]],
                         usg->vy[usg->cornerList[e + 2]],
                         usg->vz[usg->cornerList[e + 2]]);
        d[3] = osg::Vec3(usg->vx[usg->cornerList[e + 3]],
                         usg->vy[usg->cornerList[e + 3]],
                         usg->vz[usg->cornerList[e + 3]]);
        d[4] = osg::Vec3(usg->vx[usg->cornerList[e + 4]],
                         usg->vy[usg->cornerList[e + 4]],
                         usg->vz[usg->cornerList[e + 4]]);
        d[5] = osg::Vec3(usg->vx[usg->cornerList[e + 5]],
                         usg->vy[usg->cornerList[e + 5]],
                         usg->vz[usg->cornerList[e + 5]]);
        d[6] = osg::Vec3(usg->vx[usg->cornerList[e + 6]],
                         usg->vy[usg->cornerList[e + 6]],
                         usg->vz[usg->cornerList[e + 6]]);
        d[7] = osg::Vec3(usg->vx[usg->cornerList[e + 7]],
                         usg->vy[usg->cornerList[e + 7]],
                         usg->vz[usg->cornerList[e + 7]]);

        osg::Vec3 d01 = lerp(d[0], d[1], t0);
        osg::Vec3 d23 = lerp(d[3], d[2], t0);
        osg::Vec3 a = lerp(d01, d23, t1);
        osg::Vec3 d45 = lerp(d[4], d[5], t0);
        osg::Vec3 d76 = lerp(d[7], d[6], t0);
        osg::Vec3 b = lerp(d45, d76, t1);

        vel = osg::Vec3(lerp(a, b, t2));
        return true;
    }

    return false;
}

/*
osg::Vec3 * find(const struct usg *usg, const KD *tree, int &element, const float x, const float y, const float z) {

   std::vector<int> elems;
   tree->search(elems, x, y, z);
   for (int index = 0; index < elems.size(); index ++) {
      osg::Vec3 *result = inside(usg, elems[index], x, y, z);
      if (result) {
         element = elems[index];
         return result;
      }
   }
   element = -1;
   return NULL;
}
*/
bool find(const struct usg *usg, int &element,
          const float x, const float y, const float z, osg::Vec3 &v)
{

    if (element != -1)
    {

        if (inside(usg, element, x, y, z, v))
            return true;

        int idx = (*usg->neighbors)[element];
        int num = (*usg->neighbors)[idx];
        for (int index = 1; index <= num; index++)
        {
            if (inside(usg, (*usg->neighbors)[idx + index], x, y, z, v))
            {
                element = (*usg->neighbors)[idx + index];
                return true;
            }
        }
    }

    std::vector<int> elems;
    search(usg->flat, usg->cells, elems, x, y, z);
    for (int index = 0; index < elems.size(); index++)
    {
        if (inside(usg, elems[index], x, y, z, v))
        {
            element = elems[index];
            return true;
        }
    }
    element = -1;
    return false;
}
/*
osg::Vec3 * find(struct usg *usg, KD *tree, int &element, osg::Vec3 p) {
   
   return find(usg, tree, element, p.x(), p.y(), p.z());
}
*/
void createNeighborList(struct usg *usg)
{

    std::set<int> *corners = new std::set<int>[usg->numCorners];
    std::set<int> *elements = new std::set<int>[usg->numElements];

    for (int elem = 0; elem < usg->numElements; elem++)
    {

        int start = (usg->elementList)[elem];
        int end;
        if (elem < usg->numElements - 1)
            end = (usg->elementList)[elem + 1] - 1;
        else
            end = usg->numCorners - 1;

        for (int corner = start; corner <= end; corner++)
            corners[usg->cornerList[corner]].insert(elem);
    }

    for (int corner = 0; corner < usg->numCorners; corner++)
    {

        std::set<int>::iterator a, b;
        for (a = corners[corner].begin(); a != corners[corner].end(); a++)
            for (b = corners[corner].begin(); b != corners[corner].end(); b++)
                if (*a != *b)
                {
                    elements[*a].insert(*b);
                    elements[*b].insert(*a);
                }
    }

    usg->neighbors = new std::vector<int>(usg->numElements);

    for (int elem = 0; elem < usg->numElements; elem++)
    {

        // index to neighborlist
        (*usg->neighbors)[elem] = usg->neighbors->size();
        std::set<int>::iterator a;
        usg->neighbors->push_back(elements[elem].size());
        for (a = elements[elem].begin(); a != elements[elem].end(); a++)
            usg->neighbors->push_back(*a);
    }

    delete[] corners;
    delete[] elements;

    printf("neighbors: %ld\n", usg->neighbors->size());
}

struct usg *loadUSG(const char *gridName, const char *dataName)
{

    int fd = covOpenInFile(gridName);
    char buf[7];
    buf[6] = 0;
    read(fd, buf, 6);

    while (!strncmp(buf, "SETELE", 6))
    {
        int numSet;
        covReadSetBegin(fd, &numSet);
        read(fd, buf, 6);
    }

    struct usg *usg = new struct usg;

    covReadSizeUNSGRD(fd, &usg->numElements, &usg->numCorners, &usg->numPoints);

    usg->elementList = new int[usg->numElements];
    usg->typeList = new int[usg->numElements];
    usg->cornerList = new int[usg->numCorners];
    usg->x = new float[usg->numPoints];
    usg->y = new float[usg->numPoints];
    usg->z = new float[usg->numPoints];

    covReadUNSGRD(fd, usg->numElements, usg->numCorners, usg->numPoints,
                  usg->elementList, usg->cornerList, usg->typeList,
                  usg->x, usg->y, usg->z);
    printf("usg: %d %d %d\n", usg->numElements, usg->numCorners, usg->numPoints);

    covCloseInFile(fd);

    fd = covOpenInFile(dataName);
    read(fd, buf, 6);

    while (!strncmp(buf, "SETELE", 6))
    {
        int numSet;
        covReadSetBegin(fd, &numSet);
        read(fd, buf, 6);
    }

    int numElems;
    covReadSizeUSTVDT(fd, &numElems);

    usg->vx = new float[numElems];
    usg->vy = new float[numElems];
    usg->vz = new float[numElems];
    covReadUSTVDT(fd, numElems, usg->vx, usg->vy, usg->vz);
    covCloseInFile(fd);

    float bbox[6];
    KD tree;

    for (int index = 0; index < usg->numElements; index++)
    {

        int begin = usg->elementList[index];
        int end;
        if (index < usg->numElements - 1)
            end = usg->elementList[index + 1] - 1;
        else
            end = usg->numCorners - 1;

        bbox[0] = FLT_MAX;
        bbox[1] = FLT_MAX;
        bbox[2] = FLT_MAX;
        bbox[3] = -FLT_MAX;
        bbox[4] = -FLT_MAX;
        bbox[5] = -FLT_MAX;

        for (int i = begin; i <= end; i++)
        {
            if (usg->x[usg->cornerList[i]] < bbox[0])
                bbox[0] = usg->x[usg->cornerList[i]];
            if (usg->y[usg->cornerList[i]] < bbox[1])
                bbox[1] = usg->y[usg->cornerList[i]];
            if (usg->z[usg->cornerList[i]] < bbox[2])
                bbox[2] = usg->z[usg->cornerList[i]];

            if (usg->x[usg->cornerList[i]] > bbox[3])
                bbox[3] = usg->x[usg->cornerList[i]];
            if (usg->y[usg->cornerList[i]] > bbox[4])
                bbox[4] = usg->y[usg->cornerList[i]];
            if (usg->z[usg->cornerList[i]] > bbox[5])
                bbox[5] = usg->z[usg->cornerList[i]];
        }

        tree.insert(new Element(index, bbox));
    }

    struct stat fbuf;
    char name[256];
    snprintf(name, 256, "%s.kdtree", gridName);

    if (stat(name, &fbuf) != 0)
    {

        printf("building kd tree\n");
        int l = tree.build();
        printf("kdtree depth: %d\n", l);
        usg->cells = new std::vector<int>;
        usg->flat = new std::vector<struct bb>;
        printf("flattening kd tree\n");
        tree.flatten(usg->flat, usg->cells);
        int kd = open(name, O_CREAT | O_WRONLY | S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
        int numFlat = usg->flat->size();
        write(kd, &numFlat, sizeof(int));
        write(kd, &((*usg->flat)[0]), numFlat * sizeof(struct bb));
        int numCells = usg->cells->size();
        write(kd, &numCells, sizeof(int));
        write(kd, &((*usg->cells)[0]), numCells * sizeof(int));
        close(kd);
    }
    else
    {
        int kd = open(name, O_RDONLY);
        int numFlat, numCells;
        read(kd, &numFlat, sizeof(int));
        usg->flat = new std::vector<struct bb>(numFlat);
        read(kd, &((*usg->flat)[0]), numFlat * sizeof(struct bb));

        read(kd, &numCells, sizeof(int));
        usg->cells = new std::vector<int>(numCells);
        read(kd, &((*usg->cells)[0]), numCells * sizeof(int));
        close(kd);
    }

    snprintf(name, 256, "%s.neighbor", gridName);
    if (stat(name, &fbuf) != 0)
    {

        printf("create neighbor list\n");
        createNeighborList(usg);
        printf("neighbor list done\n");
        int kd = open(name, O_CREAT | O_WRONLY | S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
        int numNeighbors = usg->neighbors->size();
        write(kd, &numNeighbors, sizeof(int));
        write(kd, &((*usg->neighbors)[0]), numNeighbors * sizeof(int));
        close(kd);
    }
    else
    {
        int kd = open(name, O_RDONLY);
        int numNeighbors;
        read(kd, &numNeighbors, sizeof(int));
        usg->neighbors = new std::vector<int>(numNeighbors);
        read(kd, &((*usg->neighbors)[0]), numNeighbors * sizeof(int));
        close(kd);
    }

    return usg;
}

osg::Group *trace(const struct usg *usg)
{

    return cuda_init_usg(*usg);

    std::vector<int> result;
    osg::Vec3Array *lines = new osg::Vec3Array();

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    int e;
    osg::Vec3 v;
    find(usg, e, 20.0, 10.0, 1.0, v);
    printf("cpu: %d\n", e);

    osg::Vec3 cur(30.0, 6.0, 1.0);

    float dt = 1.0 / 100.0;
    float x = 30.0, y = 6.0, z = 1.0;
    int index;
    int element;

    osg::Vec3 v0, v1, v2, v3;
    bool b0, b1, b2, b3;
    osg::Vec3 p0, p1, p2, p3;

    element = -1;
    for (index = 0; index < 100000; index++)
    {

        b0 = find(usg, element, cur.x(), cur.y(), cur.z(), v0);
        p0 = cur;
        if (b0)
        {
            p1 = p0 + ((v0)*dt) / 2;
            b1 = find(usg, element, p1.x(), p1.y(), p1.z(), v1);
            if (b1)
            {
                p2 = p0 + ((v1)*dt) / 2;
                b2 = find(usg, element, p2.x(), p2.y(), p2.z(), v2);
                if (b2)
                {
                    p3 = p0 + ((v2)*dt);
                    b3 = find(usg, element, p3.x(), p3.y(), p3.z(), v3);
                    if (b3)
                    {
                        cur = p0 + (v0 + v1 * 2 + v2 * 2 + v3) / 6.0 * dt;
                        lines->push_back(cur);
                    }
                }
            }
        }
        if (!(b0 && b1 && b2 && b3))
        {
            gettimeofday(&t1, NULL);
            printf("\n stopping at particle %d\n", index);
            break;
        }
    }
    long long usec = ((long long)(t1.tv_sec - t0.tv_sec)) * 1000000 + (t1.tv_usec - t0.tv_usec);

    printf("   time: %lld usec (%f particles/s)\n", usec, ((double)index) * 1000000.0 / usec);

    osg::Group *group = new osg::Group();
    osg::Geode *geode = new osg::Geode();
    osg::Geometry *geom = new osg::Geometry();

    osg::DrawArrays *da = new osg::DrawArrays(osg::PrimitiveSet::LINE_STRIP, 0,
                                              lines->size());
    geom->setVertexArray(lines);
    geom->addPrimitiveSet(da);

    geode->addDrawable(geom);

    group->addChild(geode);

    struct bb bbox = (*usg->flat)[0];

    osg::Box *box = new osg::Box(osg::Vec3((bbox.maxx + bbox.minx) / 2.0,
                                           (bbox.maxy + bbox.miny) / 2.0,
                                           (bbox.maxz + bbox.minz) / 2.0),
                                 bbox.maxx - bbox.minx, bbox.maxy - bbox.miny, bbox.maxz - bbox.minz);

    osg::ShapeDrawable *boxDrawable = new osg::ShapeDrawable(box);
    osg::Geode *g = new osg::Geode();
    g->addDrawable(boxDrawable);

    osg::StateSet *state = g->getOrCreateStateSet();
    osg::PolygonMode *polyModeObj = dynamic_cast<osg::PolygonMode *>(state->getAttribute(osg::StateAttribute::POLYGONMODE));
    if (!polyModeObj)
    {
        polyModeObj = new osg::PolygonMode;
        state->setAttribute(polyModeObj);
    }
    polyModeObj->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);

    group->addChild(g);

    return group;
}

osg::Group *init_cuda(const char *grid, const char *data)
{

    struct usg *usg = loadUSG(grid, data);
    osg::Group *g = trace(usg);
}
