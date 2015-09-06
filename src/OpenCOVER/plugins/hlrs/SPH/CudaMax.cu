#include "CudaMax.cuh"

#include "cutil.h"

#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

namespace SimLib
{

	CudaMax::CudaMax(size_t elements)
	{
		mElements = elements;
		mMemSize = elements*sizeof(float);

#ifdef USE_CUDPP
		// Scan configuration
		CUDPPConfiguration config;
		config.algorithm = CUDPP_SCAN;
		config.op = CUDPP_MAX;
		config.datatype = CUDPP_FLOAT;  
		config.options = CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE;
		cudppPlan(&scanPlan, config, elements, 1, 0);
#endif

		/*d_odata; */CUDA_SAFE_CALL(cudaMalloc( (void**) &d_odata, mMemSize));
// 		h_idata = (float*)malloc(mMemSize); memset(h_idata,0, mMemSize);
// 		h_odata = (float*)malloc(mMemSize); memset(h_odata,0, mMemSize);
	}


	CudaMax::~CudaMax()
	{
#ifdef USE_CUDPP
		cudppDestroyPlan(scanPlan);
#endif
		CUDA_SAFE_CALL(cudaFree(d_odata));
	}

	float CudaMax::FindMax(float* d_idata)
	{
		thrust::device_ptr<float> dev_ptr(d_idata);

		float maxval = thrust::reduce(dev_ptr, dev_ptr + mElements, -1.0f,  thrust::maximum<float>());;

		return maxval;
	}

} // namespace SimLib