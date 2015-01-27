/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef GPU_RESOURCE_MANAGER_H
#define GPU_RESOURCE_MANAGER_H

#include <string.h>
#include <util/coTypes.h>

#include <string>
#include <map>

struct ltstr
{
    bool operator()(const char *s1, const char *s2) const
    {
        return strcmp(s1, s2) < 0;
    }
};

class GPUEXPORT GPUObject
{
public:
    GPUObject(const char *name);
    virtual ~GPUObject();
    const char *getName();

private:
    std::string name;
};

class GPUEXPORT GPUUsg : public GPUObject
{
public:
    GPUUsg(const char *name, int numElem, int numConn,
           int numCoord, int *elemList, int *typeList,
           int *connList, float *vertices, int numElemM, int numConnM, int numCoordM);

    virtual ~GPUUsg()
    {
    }

    int getNumElem();
    int getNumConn();
    int getNumCoord();

    int *getTypeList();
    int *getElemList();
    int *getConnList();

    float *getVertices();

private:
    int numElem;
    int numConn;
    int numCoord;

    int *typeList;
    int *elemList;
    int *connList;
    float *vertices;

    int numElemM;
    int numConnM;
    int numCoordM;
};

class GPUEXPORT GPUScalar : public GPUObject
{
public:
    GPUScalar(const char *name, const int numElem, float *data,
              const int numElemM);
    virtual ~GPUScalar()
    {
    }

    float *getData();
    int getNumElem();

private:
    int numElem;
    float *data;
    int numElemM;
};

class GPUEXPORT GPUVector : public GPUObject
{
public:
    GPUVector(const char *name, const int numElem, float *data,
              const int numElemM);
    virtual ~GPUVector()
    {
    }

    float *getData();
    int getNumElem();

private:
    int numElem;
    float *data;
    int numElemM;
};

class GPUEXPORT GPUResourceManager
{
public:
    GPUUsg *addUSG(const char *name,
                   const int numElem, const int numConn, const int numCoord,
                   const int *typeList, const int *elemList, const int *connList,
                   const float *x, const float *y, const float *z,
                   const int numElemM = 0, const int numConnM = 0, const int numCoordM = 0);

    GPUScalar *addScalar(const char *name, const int numElem, const float *data, const int numElemM);
    GPUVector *addVector(const char *name, const int numElem, const float *u, const float *v, const float *w, const int numElemM);
    void deleteObject(const char *name);

protected:
    GPUResourceManager();
    GPUResourceManager(const GPUResourceManager &);
    virtual ~GPUResourceManager() = 0;

private:
    /*
   virtual GPUUsg *replaceUSG(GPUUsg *usg,
                              const int numElem, const int numConn,
                              const int numCoord, const int *typeList,
                              const int *elemList, const int *connList,
                              const float *x, const float *y, const float *z) = 0;
*/
    virtual GPUUsg *allocUSG(const char *name,
                             const int numElem, const int numConn,
                             const int numCoord, const int *typeList,
                             const int *elemList, const int *connList,
                             const float *x, const float *y, const float *z,
                             const int numElemM = 0, const int numConnM = 0,
                             const int numCoordM = 0) = 0;

    virtual GPUScalar *allocScalar(const char *name, const int numElem,
                                   const float *data, const int numElemM = 0) = 0;
    virtual GPUVector *allocVector(const char *name, const int numElem,
                                   const float *u, const float *v, const float *w, const int numElemM = 0) = 0;

    virtual void deallocUSG(GPUUsg *usg) = 0;
    virtual void deallocScalar(GPUScalar *scalar) = 0;
    virtual void deallocVector(GPUVector *vector) = 0;

    std::map<std::string, GPUObject *> objects;
    std::map<std::string, int> refCount;

    std::map<void *, void *> pointers;
};

#endif
