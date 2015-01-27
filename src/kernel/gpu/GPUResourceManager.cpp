/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**
 * Memory manager for COVISE objects on GPUs
 *
 * Handles copying from host to device and simple reference counting of
 * objects for reusal of data in different kernels.
 */
#include <stdlib.h>
#include <string.h>

#include "GPUResourceManager.h"

GPUObject::GPUObject(const char *n)
{
    name = n;
}

const char *GPUObject::getName()
{
    return name.c_str();
}

GPUObject::~GPUObject()
{
}

GPUUsg::GPUUsg(const char *name, int elem, int conn,
               int coord, int *elemL, int *typeL,
               int *connL, float *v, int elemM, int connM, int coordM)
    : GPUObject(name)
    , numElem(elem)
    , numConn(conn)
    , numCoord(coord)
    , typeList(typeL)
    , elemList(elemL)
    , connList(connL)
    , vertices(v)
    , numElemM(elemM)
    , numConnM(connM)
    , numCoordM(coordM)
{
}

int GPUUsg::getNumElem()
{
    return numElem;
}

int GPUUsg::getNumConn()
{
    return numConn;
}

int GPUUsg::getNumCoord()
{
    return numCoord;
}

int *GPUUsg::getTypeList()
{
    return typeList;
}

int *GPUUsg::getElemList()
{
    return elemList;
}

int *GPUUsg::getConnList()
{
    return connList;
}

float *GPUUsg::getVertices()
{
    return vertices;
}

GPUScalar::GPUScalar(const char *name, const int elem, float *d, const int elemM)
    : GPUObject(name)
    , numElem(elem)
    , data(d)
    , numElemM(elemM)
{
}

float *GPUScalar::getData()
{
    return data;
}

int GPUScalar::getNumElem()
{
    return numElem;
}

GPUVector::GPUVector(const char *name, const int elem, float *d, const int elemM)
    : GPUObject(name)
    , numElem(elem)
    , data(d)
    , numElemM(elemM)
{
}

float *GPUVector::getData()
{
    return data;
}

int GPUVector::getNumElem()
{
    return numElem;
}

GPUResourceManager::GPUResourceManager()
{
}

GPUResourceManager::GPUResourceManager(class GPUResourceManager const &oldrm)
{
    (void)oldrm;
}
GPUResourceManager::~GPUResourceManager()
{
}
/*
GPUUsg * GPUResourceManager::replaceUSG(const char *name,
                                        const int numElem, const int numConn,
                                        const int numCoord, const int *typeList,
                                        const int *elemList, const int *connList,
                                        const float *x, const float *y, const float *z,
                                        cudaStream_t *stream)
{
   if (name == "")
      return NULL;
   
   GPUUsg *usg = NULL;

   std::map<std::string, GPUObject *>::iterator i = objects.find(name);
   if (i != objects.end()) {
      usg = dynamic_cast<GPUUsg *>(i->second);
      
      if (usg->getNumElem() >= numElem &&
          usg->getNumConn() >= numConn &&
          usg->getNumCoord() >= numCoord) {

         int *tl = usg->getTypeList();
         int *el = usg->getElemList();
         int *cl = usg->getConnList();
         float *vl = usg->getVertices();

         cudaMemcpyAsync(el, elemList, sizeof(uint) * numElem,
                         cudaMemcpyHostToDevice, stream);
         
         (void) tl;
         (void) el;
         (void) cl;
         (void) vl;
         (void) elemList;
         (void) connList;
         (void) typeList;
         (void) x;
         (void) y;
         (void) z;
      }
   }
   return usg;
}
*/

GPUUsg *GPUResourceManager::addUSG(const char *name,
                                   const int numElem, const int numConn,
                                   const int numCoord, const int *typeList,
                                   const int *elemList, const int *connList,
                                   const float *x, const float *y, const float *z,
                                   int numElemM, int numConnM, int numCoordM)
{
    if (!name || name[0] == '\0')
        return NULL;

    std::map<std::string, GPUObject *>::iterator i = objects.find(name);
    if (i != objects.end())
    {
        GPUUsg *usg = dynamic_cast<GPUUsg *>(i->second);
        /*      
      if (replace) {
         if (usg->getNumElem() >= numElem &&
             usg->getNumConn() >= numConn &&
             usg->getNumCoord() >= numCoord) {
            replaceUSG(usg, numElem, numConn, numCoord,
                       typeList, elemList, connList, x, y, z);
         } else {
            printf("usg replace error\n");
         }
      }
*/
        if (usg)
        {
            refCount[name]++;
            //printf("GPUResourceManager::addUSG reusing [%s]\n", name);
            return usg;
        }
        //printf("GPUResourceManager::addUSG collision [%s]\n", name);
        return NULL;
    }
    //printf("GPUResourceManager::addUSG allocUSG [%s]\n", name.c_str);
    GPUUsg *usg = allocUSG(name, numElem, numConn, numCoord, typeList, elemList,
                           connList, x, y, z, numElemM, numConnM, numCoordM);

    objects[name] = usg;
    refCount[name] = 1;
    return usg;
}

GPUScalar *GPUResourceManager::addScalar(const char *name,
                                         const int numElem, const float *data,
                                         const int numElemM)
{
    if (!name || name[0] == '\0')
        return NULL;

    std::map<std::string, GPUObject *>::iterator i = objects.find(name);
    if (i != objects.end())
    {
        GPUScalar *scalar = dynamic_cast<GPUScalar *>(i->second);
        if (scalar)
        {
            refCount[name]++;
            //printf("GPUResourceManager::addScalar reusing [%s]\n", name);
            return scalar;
        }

        //printf("GPUResourceManager::addScalar collision [%s]\n", name);
        return NULL;
    }
    //printf("GPUResourceManager::addScalar allocScalar [%s]\n", name.c_str());
    GPUScalar *scalar = allocScalar(name, numElem, data, numElemM);

    objects[name] = scalar;
    refCount[name] = 1;
    return scalar;
}

GPUVector *GPUResourceManager::addVector(const char *name,
                                         const int numElem, const float *u,
                                         const float *v, const float *w,
                                         const int numElemM)
{
    if (!name || name[0] == '\0')
        return NULL;

    std::map<std::string, GPUObject *>::iterator i = objects.find(name);
    if (i != objects.end())
    {
        GPUVector *vector = dynamic_cast<GPUVector *>(i->second);
        if (vector)
        {
            refCount[name]++;
            //printf("GPUResourceManager::addVector reusing [%s]\n", name);
            return vector;
        }

        //printf("GPUResourceManager::addVector collision [%s]\n", name);
        return NULL;
    }
    //printf("GPUResourceManager::addScalar allocScalar [%s]\n", name.c_str());
    GPUVector *vector = allocVector(name, numElem, u, v, w, numElemM);

    objects[name] = vector;
    refCount[name] = 1;
    return vector;
}

void GPUResourceManager::deleteObject(const char *name)
{
    //printf("GPUResourceManager::deleteObject [%s]\n", name.c_str());
    if (!name || name[0] == '\0')
        return;

    std::map<std::string, GPUObject *>::iterator i = objects.find(name);
    if (i == objects.end())
    {
        //printf("GPUResourceManager::deleteObject object [%s] not found\n", name.c_str());
        return;
    }

    if (--refCount[name] == 0)
    {

        //printf("GPUResourceManager::deleteObject deleting[%s]\n", name.c_str());
        GPUUsg *usg = dynamic_cast<GPUUsg *>(i->second);
        GPUScalar *scalar = dynamic_cast<GPUScalar *>(i->second);
        if (usg)
        {
            deallocUSG(usg);
            delete i->second;
            objects.erase(i);
        }
        else
        {
            deallocScalar(scalar);
            delete i->second;
            objects.erase(i);
        }
    }
    else
    {
        //printf("GPUResourceManager::deleteObject [%s] refCount %d\n", name.c_str(), refCount[name]);
    }
}
