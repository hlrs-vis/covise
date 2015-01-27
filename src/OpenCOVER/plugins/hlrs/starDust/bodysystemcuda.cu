#include <gpu/helper_cuda.h>
#include <math.h>

#include <GL/glew.h>

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


#include "bodysystem.h"


//////////////////////////////////////////////////////////////////////
//
// vec3 - template class for 3-tuple vector
//
//////////////////////////////////////////////////////////////////////
 template <class T>
 class vec3
{
public:

	typedef T value_type;
	__device__ int size() const
	{
		return 3;
	}

	////////////////////////////////////////////////////////
	//
	//  Constructors
	//
	////////////////////////////////////////////////////////

	// Default/scalar constructor
	__device__ vec3(const T &t = T())
	{
		for (int i = 0; i < size(); i++)
		{
			_array[i] = t;
		}
	}

	// Construct from array
	__device__ vec3(const T *tp)
	{
		for (int i = 0; i < size(); i++)
		{
			_array[i] = tp[i];
		}
	}

	// Construct from explicit values
	__device__ vec3(const T v0, const T v1, const T v2)
	{
		x = v0;
		y = v1;
		z = v2;
	}

	__device__ const T *get_value() const
	{
		return _array;
	}

	__device__ vec3<T> &set_value(const T *rhs)
	{
		for (int i = 0; i < size(); i++)
		{
			_array[i] = rhs[i];
		}

		return *this;
	}

	// indexing operators
	__device__ T &operator [](int i)
	{
		return _array[i];
	}

	__device__ const T &operator [](int i) const
	{
		return _array[i];
	}

	// type-cast operators
	__device__ operator T *()
	{
		return _array;
	}

	__device__ operator const T *() const
	{
		return _array;
	}

	////////////////////////////////////////////////////////
	//
	//  Math operators
	//
	////////////////////////////////////////////////////////

	// dot product
	__device__ T dot(vec3<T> &v)
	{
		return x*v.x+ y*v.y + z*v.z;
	}

	__device__ T sqrLen()
	{
		return x*x+y*y+z*z;
	}

	__device__ T length()
	{
		return sqrt(x*x+y*y+z*z);
	}

	__device__ vec3<T> getA(T m)
	{
		const T G = 6.67384E-11;
		const T G_km = G*-0.000000001;

		T distSqr = sqrLen();
		T  len = sqrt(distSqr);
		T factor = G_km*(m)/len/distSqr;
		return((*this)*factor);




	}

	// scalar multiply assign
	__device__ friend vec3<T> &operator *= (vec3<T> &lhs, T d)
	{
		for (int i = 0; i < lhs.size(); i++)
		{
			lhs._array[i] *= d;
		}

		return lhs;
	}

	// component-wise vector multiply assign
	__device__ friend vec3<T> &operator *= (vec3<T> &lhs, const vec3<T> &rhs)
	{
		for (int i = 0; i < lhs.size(); i++)
		{
			lhs._array[i] *= rhs[i];
		}

		return lhs;
	}

	// scalar divide assign
	__device__ friend vec3<T> &operator /= (vec3<T> &lhs, T d)
	{
		if (d == 0)
		{
			return lhs;
		}

		for (int i = 0; i < lhs.size(); i++)
		{
			lhs._array[i] /= d;
		}

		return lhs;
	}

	// component-wise vector divide assign
	__device__ friend vec3<T> &operator /= (vec3<T> &lhs, const vec3<T> &rhs)
	{
		for (int i = 0; i < lhs.size(); i++)
		{
			lhs._array[i] /= rhs._array[i];
		}

		return lhs;
	}

	// component-wise vector add assign
	__device__ friend vec3<T> &operator += (vec3<T> &lhs, const vec3<T> &rhs)
	{
		for (int i = 0; i < lhs.size(); i++)
		{
			lhs._array[i] += rhs._array[i];
		}

		return lhs;
	}

	// component-wise vector subtract assign
	__device__ friend vec3<T> &operator -= (vec3<T> &lhs, const vec3<T> &rhs)
	{
		for (int i = 0; i < lhs.size(); i++)
		{
			lhs._array[i] -= rhs._array[i];
		}

		return lhs;
	}

	// unary negate
	__device__ friend vec3<T> operator - (const vec3<T> &rhs)
	{
		vec3<T> rv;

		for (int i = 0; i < rhs.size(); i++)
		{
			rv._array[i] = -rhs._array[i];
		}

		return rv;
	}

	// vector add
	__device__ friend vec3<T> operator + (const vec3<T> &lhs, const vec3<T> &rhs)
	{
		vec3<T> rt(lhs);
		return rt += rhs;
	}

	// vector subtract
	__device__ friend vec3<T> operator - (const vec3<T> &lhs, const vec3<T> &rhs)
	{
		vec3<T> rt(lhs);
		return rt -= rhs;
	}

	// scalar multiply
	__device__ friend vec3<T> operator * (const vec3<T> &lhs, T rhs)
	{
		vec3<T> rt(lhs);
		return rt *= rhs;
	}

	// scalar multiply
	__device__ friend vec3<T> operator * (T lhs, const vec3<T> &rhs)
	{
		vec3<T> rt(lhs);
		return rt *= rhs;
	}

	// vector component-wise multiply
	__device__ friend vec3<T> operator * (const vec3<T> &lhs, const vec3<T> &rhs)
	{
		vec3<T> rt(lhs);
		return rt *= rhs;
	}

	// scalar multiply
	__device__ friend vec3<T> operator / (const vec3<T> &lhs, T rhs)
	{
		vec3<T> rt(lhs);
		return rt /= rhs;
	}

	// vector component-wise multiply
	__device__ friend vec3<T> operator / (const vec3<T> &lhs, const vec3<T> &rhs)
	{
		vec3<T> rt(lhs);
		return rt /= rhs;
	}

	////////////////////////////////////////////////////////
	//
	//  Comparison operators
	//
	////////////////////////////////////////////////////////

	// equality
	__device__ friend bool operator == (const vec3<T> &lhs, const vec3<T> &rhs)
	{
		bool r = true;

		for (int i = 0; i < lhs.size(); i++)
		{
			r &= lhs._array[i] == rhs._array[i];
		}

		return r;
	}

	// inequality
	__device__ friend bool operator != (const vec3<T> &lhs, const vec3<T> &rhs)
	{
		bool r = true;

		for (int i = 0; i < lhs.size(); i++)
		{
			r &= lhs._array[i] != rhs._array[i];
		}

		return r;
	}

	////////////////////////////////////////////////////////////////////////////////
	//
	// dimension specific operations
	//
	////////////////////////////////////////////////////////////////////////////////

	// cross product
	__device__ friend vec3<T> cross(const vec3<T> &lhs, const vec3<T> &rhs)
	{
		vec3<T> r;

		r.x = lhs.y * rhs.z - lhs.z * rhs.y;
		r.y = lhs.z * rhs.x - lhs.x * rhs.z;
		r.z = lhs.x * rhs.y - lhs.y * rhs.x;

		return r;
	}

	//data intentionally left public to allow vec2.x
	union
	{
		struct
		{
			T x, y, z;          // standard names for components
		};
		struct
		{
			T s, t, r;          // standard names for components
		};
		T _array[3];     // array access
	};
};

/**
* CUDA Kernel Device code
*
* integrates particles (Euler)
*/

 template <typename T>
__global__ void
	integrateParticlesEuler(T *pos, T *velo, int numElements, double deltaTime,int iterations,T *pPos, T *pVelo ,int numActivePlanets)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int ip = i*4;
	int iv = i*3;
	T ms = 1.98892E30;
	T G = 6.67384E-11;
	T G_km = G*-0.000000001;
	if (i < numElements)
	{
		for(int n=0;n<iterations;n++)
		{
			for(int p=0;p<numActivePlanets;p++)
			{
				T dx=pos[ip  ] - pPos[p*4  ];
				T dy=pos[ip+1] - pPos[p*4+1];
				T dz=pos[ip+2] - pPos[p*4+2];
				//float pm =pos[ip+3];
				T distSqr = dx * dx + dy * dy + dz * dz;
				T  len = sqrt(distSqr);
				T factor = G_km*(pPos[p*4+3])/len/distSqr;
				//float ax = (px/len)*(G*ms/distSqr)*-0.000000001;
				T ax = dx*factor;
				T ay = dy*factor;
				T az = dz*factor;
				velo[iv] = velo[iv] + ax *deltaTime;
				velo[iv+1] = velo[iv+1] + ay *deltaTime;
				velo[iv+2] = velo[iv+2] + az *deltaTime;
			}
			T px= pos[ip];
			T py=pos[ip+1];
			T pz=pos[ip+2];
			//float pm =pos[ip+3];
			T distSqr = px * px + py * py + pz * pz;
			T  len = sqrt(distSqr);
			T factor = G_km*ms*(1-9.188354137893872E-4)/len/distSqr;
			//float ax = (px/len)*(G*ms/distSqr)*-0.000000001;
			T ax = px*factor;
			T ay = py*factor;
			T az = pz*factor;
			velo[iv] = velo[iv] + ax *deltaTime;
			velo[iv+1] = velo[iv+1] + ay *deltaTime;
			velo[iv+2] = velo[iv+2] + az *deltaTime;
			pos[ip] = px + velo[iv]*deltaTime;
			pos[ip+1] = py + velo[iv+1]*deltaTime;
			pos[ip+2] = pz + velo[iv+2]*deltaTime;
		}
	}
}
//template void integrateParticlesEuler<float>(float *pos, float *velo, int numElements, double deltaTime,int iterations,float *pPos, float *pVelo ,int numActivePlanets)
//template void integrateParticlesEuler<double>(double *pos, double *velo, int numElements, double deltaTime,int iterations,double *pPos, double *pVelo ,int numActivePlanets)



 template <typename T>
__global__ void
	integrateParticlesTest(T *pos, T *velo, int numElements, double deltaTime,int iterations,T *pPos, T *pVelo ,int numActivePlanets)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int ip = i*4;
	int iv = i*3;
	T ms = 1.98892E30;
	T G = 6.67384E-11;
	T G_km = G*-0.000000001;
	if (i < numElements)
	{
		vec3<T> velocity(velo[iv],velo[iv+1],velo[iv+2]);
		vec3<T> position(pos[ip],pos[ip+1],pos[ip+2]);
		for(int n=0;n<iterations;n++)
		{
			for(int p=0;p<numActivePlanets;p++)
			{
				int iPlanet = p*4;
				T PlanetMass = pPos[iPlanet+3];
				vec3<T> PlanetPosition(pPos[iPlanet],pPos[iPlanet+1],pPos[iPlanet+2]);
				vec3<T> d=position - PlanetPosition;
				vec3<T> a = d.getA(PlanetMass);
				velocity +=a*deltaTime;
			}

			T MS_RadiationPressure=ms*(1.0-9.188354137893872E-4);
			vec3<T> as=position.getA(MS_RadiationPressure);
			velocity +=as*deltaTime;
			position +=velocity*deltaTime;

		}
		velo[iv] = velocity.x;
		velo[iv+1] = velocity.y;
		velo[iv+2] = velocity.z;
		pos[ip] = position.x;
		pos[ip+1] = position.y;
		pos[ip+2] = position.z;
	}
}

/**
* CUDA Kernel Device code
*
* integrates particles rk4
*/
 template <typename T>
__global__ void
	integrateParticles(T *pos, T *velo, int numElements, double deltaTime,int iterations,T *pPos, T *pVelo ,int numActivePlanets)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int ip = i*4;
	int iv = i*3;
	T ms = 1.98892E30;
	T totalTime = iterations*deltaTime;
	T currentTime = 0;
	T currentDeltaTime=deltaTime;
	if (i < numElements)
	{
		
		vec3<T> velocity(velo[iv],velo[iv+1],velo[iv+2]);
		vec3<T> position(pos[ip],pos[ip+1],pos[ip+2]);
		while(currentTime < totalTime)
		{
			vec3<T> startVelocity = velocity;
			if(currentTime + currentDeltaTime > totalTime)
				currentDeltaTime = totalTime - currentTime;
		    T h2 = currentDeltaTime/2.0;
			for(int p=0;p<numActivePlanets;p++)
			{
				int iPlanet = p*4;
				T PlanetMass = pPos[iPlanet+3];
		        vec3<T> PlanetPosition(pPos[iPlanet],pPos[iPlanet+1],pPos[iPlanet+2]);
				vec3<T> d=position - PlanetPosition;
				vec3<T> k1=d.getA(PlanetMass);
				vec3<T> dn = d+(k1*h2)*h2;
				vec3<T> k2=dn.getA(PlanetMass);
				      dn = d+(k2*h2)*h2;
				vec3<T> k3=dn.getA(PlanetMass);
				dn = d+((k3*deltaTime)*currentDeltaTime);
				vec3<T> k4=dn.getA(PlanetMass);
				vec3<T> a =(k1*0.16666666667+k2*0.3333333333333+k3*0.3333333333333+k4*0.16666666667);
				velocity += a*currentDeltaTime;
			}
			T MS_RadiationPressure=ms*(1.0-9.188354137893872E-4);
			vec3<T> k1=position.getA(MS_RadiationPressure);

			vec3<T> pn = position+((k1*h2)*h2);
			vec3<T> k2=pn.getA(MS_RadiationPressure);
			pn = position+((k2*h2)*h2);
			vec3<T> k3=pn.getA(MS_RadiationPressure);
			pn = position+((k3*deltaTime)*currentDeltaTime);
			vec3<T> k4=pn.getA(MS_RadiationPressure);
			vec3<T> a =(k1*0.16666666667+k2*0.3333333333333+k3*0.3333333333333+k4*0.16666666667);
			velocity += a*currentDeltaTime;

			vec3<T> ka;

			/* compute kappa-s */
			ka.x = fabsf( (2.0f*(k3.x-k2.x))/(k2.x-k1.x) );
			ka.y = fabsf( (2.0f*(k3.y-k2.y))/(k2.y-k1.y) );
			ka.z = fabsf( (2.0f*(k3.z-k2.z))/(k2.z-k1.z) );

			// and find minimum kappa
			T kappa = ka[0];
			if( ka[1]<kappa )
				kappa = ka[1];
			if( ka[2]<kappa )
				kappa = ka[2];

			// stepsize adjustment
			if( kappa<0.05)
			{
				// double stepsize for next step
				currentDeltaTime *= 2.0;
				position +=velocity*currentDeltaTime;
				currentTime += currentDeltaTime;
			}
			else
			{
				if( kappa>0.2 )
				{
					// halve stepsize and redo current step

					currentDeltaTime = currentDeltaTime * 0.5f;
					velocity = startVelocity;

				}
				else
				{
					position +=velocity*currentDeltaTime;
					currentTime += currentDeltaTime;
				}
			}

		}
		velo[iv] = velocity.x;
		velo[iv+1] = velocity.y;
		velo[iv+2] = velocity.z;
		pos[ip] = position.x;
		pos[ip+1] = position.y;
		pos[ip+2] = position.z;
	}
}

template <class T> CudaParticles<T>::CudaParticles(int n)
{
	numParticles = n;
	numActiveParticles = 0;
}

template <class T> void *CudaParticles<T>::mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
{
	void *ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,
		*cuda_vbo_resource));
	return ptr;

}

template <class T> void CudaParticles<T>::unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
}

template <class T> unsigned int CudaParticles<T>::createVBO(unsigned int size)
{
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return vbo;
}


template <class T> void CudaParticles<T>::setInitialData(T *pos,T*velo)
{

	cudaError_t err = cudaSuccess;

	posVbo = createVBO(numParticles*sizeof(T) *4);


	glBindBuffer(GL_ARRAY_BUFFER, posVbo);
	glBufferSubData(GL_ARRAY_BUFFER, 0, numParticles*4*sizeof(T), pos);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	err = cudaGraphicsGLRegisterBuffer(&cuda_posvbo_resource, posVbo, cudaGraphicsMapFlagsNone);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "cudaGraphicsGLRegisterBuffer failed (error code: %s)!\n", cudaGetErrorString(err));
	}

	err = cudaMalloc((void **)&cudaVelo, numParticles*3*sizeof(T));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Cuda Malloc of %ld bytes failed (error code: %s)!\n",
            (long)numParticles*3*sizeof(T), cudaGetErrorString(err));
	}
	cudaMemcpy((char *) cudaVelo, velo, numParticles*3*sizeof(T), cudaMemcpyHostToDevice);

}

template <class T> void CudaParticles<T>::copyInitialData(T *pos,T*velo)
{
	glBindBuffer(GL_ARRAY_BUFFER, posVbo);
	glBufferSubData(GL_ARRAY_BUFFER, 0, numParticles*4*sizeof(T), pos);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	cudaMemcpy((char *) cudaVelo, velo, numParticles*3*sizeof(T), cudaMemcpyHostToDevice);
}


template <class T> void CudaParticles<T>::setInitialPlanetData(T *pos,T*velo)
{

	cudaError_t err = cudaSuccess;
	err = cudaMalloc((void **)&cudaPlanetPos, numParticles*4*sizeof(T));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Cuda Malloc of %ld bytes failed (error code: %s)!\n",
            (long)numParticles*4*sizeof(T), cudaGetErrorString(err));
	}
	cudaMemcpy((char *) cudaPlanetPos, pos, numParticles*4*sizeof(T), cudaMemcpyHostToDevice);

	err = cudaMalloc((void **)&cudaPlanetVelo, numParticles*3*sizeof(T));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Cuda Malloc of %ld bytes failed (error code: %s)!\n",
            (long)numParticles*3*sizeof(T), cudaGetErrorString(err));
	}
	cudaMemcpy((char *) cudaPlanetVelo, velo, numParticles*3*sizeof(T), cudaMemcpyHostToDevice);

}

template <class T> void CudaParticles<T>::copyPlanetData(T *pos,T*velo)
{	
	cudaMemcpy((char *) cudaPlanetPos, pos, numParticles*4*sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy((char *) cudaPlanetVelo, velo, numParticles*3*sizeof(T), cudaMemcpyHostToDevice);
}

template <class T> void CudaParticles<T>::integrate(double deltaT,int iterations,int numActivePlanets)
{

	T *dPos = (T *) mapGLBufferObject(&cuda_posvbo_resource);
	// Launch the integrateParticle CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid =(numParticles + threadsPerBlock - 1) / threadsPerBlock;
	//printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	integrateParticlesEuler<T><<<blocksPerGrid, threadsPerBlock>>>(dPos, cudaVelo, numActiveParticles,deltaT,iterations,cudaPlanetPos,cudaPlanetVelo,numActivePlanets);
	//integrateParticlesTest<T><<<blocksPerGrid, threadsPerBlock>>>(dPos, cudaVelo, numActiveParticles,deltaT,iterations,cudaPlanetPos,cudaPlanetVelo,numActivePlanets);

	unmapGLBufferObject(cuda_posvbo_resource);
}

template class CudaParticles<float>;
template class CudaParticles<double>;
