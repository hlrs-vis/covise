/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __BODYSYSTEM_H__
#define __BODYSYSTEM_H__

template <class T>
class CudaParticles
{
public:
    CudaParticles(int n);
    unsigned int getPosVBO()
    {
        return posVbo;
    };

    void setInitialData(T *pos, T *velo);
    void copyInitialData(T *pos, T *velo);
    void setInitialPlanetData(T *pos, T *velo);
    void copyPlanetData(T *pos, T *velo);
    void integrate(double deltaT, int iterations, int numActivePlanets);
    int getNumActiveParticles()
    {
        return numActiveParticles;
    };
    void setNumActiveParticles(int a)
    {
        numActiveParticles = a;
    };

private:
    int numParticles;
    int numActiveParticles;
    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
    void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);

    unsigned int createVBO(unsigned int size);

    unsigned int posVbo; // vertex buffer object for particle positions
    struct cudaGraphicsResource *cuda_posvbo_resource; // handles OpenGL-CUDA exchange

    T *cudaPosVBO; // these are the CUDA deviceMem Pos
    T *cudaVelo;
    T *cudaPlanetPos;
    T *cudaPlanetVelo;
};

#endif // __BODYSYSTEM_H__
