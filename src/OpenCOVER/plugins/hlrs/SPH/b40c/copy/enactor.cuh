/******************************************************************************
 * 
 * Copyright 2010-2011 Duane Merrill
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 ******************************************************************************/

/******************************************************************************
 * Copy enactor
 ******************************************************************************/

#pragma once

#include <b40c/util/enactor_base.cuh>
#include <b40c/util/error_utils.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/arch_dispatch.cuh>

#include <b40c/copy/autotuned_policy.cuh>
#include <b40c/copy/kernel.cuh>

namespace b40c {
namespace copy {


/**
 * Copy enactor class.
 */
class Enactor : public util::EnactorBase
{
protected:

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Temporary device storage needed for managing work-stealing progress
	// within a kernel invocation.
	util::CtaWorkProgressLifetime work_progress;


public:

	/**
	 * Constructor
	 */
	Enactor() {}


	/**
	 * Enacts a copy operation on the specified device data using
	 * a heuristic for selecting granularity configuration based upon
	 * problem size.
	 *
	 * @param d_dest
	 * 		Pointer to result location
	 * @param d_src
	 * 		Pointer to array of bytes to be copied
	 * @param num_bytes
	 * 		Number of bytes to copy
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename SizeT>
	cudaError_t Copy(
		void *d_dest,
		void *d_src,
		SizeT num_bytes,
		int max_grid_size = 0);


	/**
	 * Enacts a copy operation on the specified device data using the
	 * enumerated tuned granularity configuration
	 *
	 * @param d_dest
	 * 		Pointer to result location
	 * @param d_src
	 * 		Pointer to array of bytes to be copied
	 * @param num_bytes
	 * 		Number of bytes to copy
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <ProbSizeGenre PROB_SIZE_GENRE, typename SizeT>
	cudaError_t Copy(
		void *d_dest,
		void *d_src,
		SizeT num_bytes,
		int max_grid_size = 0);


	/**
	 * Enacts a copy on the specified device data using the specified
	 * granularity configuration
	 *
	 * For generating copy kernels having computational granularities in accordance
	 * with user-supplied granularity-specialization types.  (Useful for auto-tuning.)
	 *
	 * @param d_dest
	 * 		Pointer to array of elements to be copyd
	 * @param d_src
	 * 		Pointer to result location
	 * @param num_elements
	 * 		Number of elements to copy
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename Policy>
	cudaError_t Copy(
		typename Policy::T *d_dest,
		typename Policy::T *d_src,
		typename Policy::SizeT num_elements,
		int max_grid_size = 0);


    /**
	 * Performs a copy pass
	 */
	template <typename Policy>
	cudaError_t Copy(
		typename Policy::T *d_dest,
		typename Policy::T *d_src,
		typename Policy::SizeT num_elements,
		int extra_bytes,
		int max_grid_size);
};



/******************************************************************************
 * Helper structures
 ******************************************************************************/

/**
 * Type for encapsulating operational details regarding an invocation
 */
template <typename _SizeT>
struct Detail
{
	typedef _SizeT SizeT;

	Enactor 	*enactor;

	void		*d_dest;
	void		*d_src;
	SizeT 		num_bytes;
	int 		max_grid_size;

	// Constructor
	Detail(
		Enactor *enactor,
		void *d_dest,
		void *d_src,
		SizeT num_bytes,
		int max_grid_size = 0) :
			enactor(enactor),
			d_dest(d_dest),
			d_src(d_src),
			num_bytes(num_bytes),
			max_grid_size(max_grid_size)
	{}

	template <typename Policy>
	cudaError_t Enact()
	{
		typedef typename Policy::T T;

		SizeT num_elements = num_bytes / sizeof(T);
		int extra_bytes = num_bytes - (num_elements * sizeof(T));

		// Invoke enactor with type
		return enactor->template Copy<Policy>(
			(T*) d_dest, (T*) d_src, num_elements, extra_bytes, max_grid_size);
	}
};


/**
 * Helper structure for resolving and enacting autotuned policy
 */
template <ProbSizeGenre PROB_SIZE_GENRE>
struct PolicyResolver
{
	/**
	 * ArchDispatch call-back with static CUDA_ARCH
	 */
	template <int CUDA_ARCH, typename Detail>
	static cudaError_t Enact(Detail &detail)
	{
		// Obtain tuned granularity type
		typedef AutotunedPolicy<
			typename Detail::SizeT,
			CUDA_ARCH,
			PROB_SIZE_GENRE> Policy;

		return detail.template Enact<Policy>();
	}
};


/**
 * Helper structure for resolving and enacting autotuned policy
 *
 * Specialization for UNKNOWN problem size genre to select other problem size
 * genres based upon problem size, machine width, etc.
 */
template <>
struct PolicyResolver <UNKNOWN_SIZE>
{
	/**
	 * ArchDispatch call-back with static CUDA_ARCH
	 */
	template <int CUDA_ARCH, typename Detail>
	static cudaError_t Enact(Detail &detail)
	{
		typedef typename Detail::SizeT SizeT;

		// Obtain large tuned granularity type
		typedef AutotunedPolicy<SizeT, CUDA_ARCH, LARGE_SIZE> LargePolicy;

		// Identify the maximum problem size for which we can saturate loads
		int saturating_load = LargePolicy::TILE_ELEMENTS *
			LargePolicy::CTA_OCCUPANCY *
			detail.enactor->SmCount();

		SizeT num_elements = detail.num_bytes / sizeof(typename LargePolicy::T);
		if (num_elements < saturating_load) {

			// Invoke base class enact with small-problem config type
			typedef AutotunedPolicy<SizeT, CUDA_ARCH, SMALL_SIZE> SmallPolicy;

			return detail.template Enact<SmallPolicy>();
		}

		return detail.template Enact<LargePolicy>();
	}
};


/******************************************************************************
 * Enactor Implementation
 ******************************************************************************/

/**
 * Performs a copy pass
 */
template <typename Policy>
cudaError_t Enactor::Copy(
	typename Policy::T 				*d_dest,
	typename Policy::T 				*d_src,
	typename Policy::SizeT			num_elements,
	int 							extra_bytes,
	int 							max_grid_size)
{
	typedef typename Policy::T 				T;
	typedef typename Policy::SizeT 			SizeT;

	cudaError_t retval = cudaSuccess;
	do {

		// Make sure we have a valid policy
		if (!Policy::VALID) {
			retval = util::B40CPerror(cudaErrorInvalidConfiguration, "Enactor invalid policy", __FILE__, __LINE__);
			break;
		}

		// Copy kernel
		typename Policy::KernelPtr Kernel = Policy::Kernel();

		// Get kernel attributes
		cudaFuncAttributes kernel_attrs;
		if (retval = util::B40CPerror(cudaFuncGetAttributes(&kernel_attrs, Kernel),
			"Enactor cudaFuncGetAttributes kernel_attrs failed", __FILE__, __LINE__)) break;

		// Max CTA occupancy for the actual target device
		int max_cta_occupancy;
		if (retval = MaxCtaOccupancy(max_cta_occupancy, Kernel, Policy::THREADS)) break;

		if (ENACTOR_DEBUG) printf("Occupancy:\t[sweep occupancy: %d]\n",
			max_cta_occupancy);

		// Compute sweep grid size
		int grid_size = GridSize(
			Policy::OVERSUBSCRIBED_GRID_SIZE,
			Policy::SCHEDULE_GRANULARITY,
			max_cta_occupancy,
			num_elements,
			max_grid_size);

		// Obtain a CTA work distribution
		util::CtaWorkDistribution<SizeT> work;
		work.template Init<Policy::LOG_SCHEDULE_GRANULARITY>(num_elements, grid_size);

		if (ENACTOR_DEBUG) {
			PrintPassInfo<Policy>(work);
		}

		// If we're work-stealing, make sure our work progress is set up
		// for the next pass
		if (Policy::WORK_STEALING) {
			if (retval = work_progress.Setup()) break;
		}

		int dynamic_smem = 0;

		Kernel<<<work.grid_size, Policy::THREADS, dynamic_smem>>>(
			d_src, d_dest, work, work_progress, extra_bytes);

		if (ENACTOR_DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor Kernel failed ", __FILE__, __LINE__, ENACTOR_DEBUG))) break;

	} while (0);

	// Cleanup
	if (retval) {
		// We had an error, which means that the device counters may not be
		// properly initialized for the next pass: reset them.
		work_progress.HostReset();
	}

	return retval;
}


/**
 * Enacts a copy operation on the specified device data.
 */
template <typename Policy>
cudaError_t Enactor::Copy(
	typename Policy::T *d_dest,
	typename Policy::T *d_src,
	typename Policy::SizeT num_elements,
	int max_grid_size)
{
	return Copy<Policy>(d_dest, d_src, num_elements, 0, max_grid_size);
}


/**
 * Enacts a copy operation on the specified device data.
 */
template <
	ProbSizeGenre PROB_SIZE_GENRE,
	typename SizeT>
cudaError_t Enactor::Copy(
	void *d_dest,
	void *d_src,
	SizeT num_bytes,
	int max_grid_size)
{
	typedef Detail<SizeT> Detail;
	typedef PolicyResolver<PROB_SIZE_GENRE> Resolver;

	Detail detail(this, d_dest, d_src, num_bytes, max_grid_size);

	return util::ArchDispatch<__B40C_CUDA_ARCH__, Resolver>::Enact(
		detail, PtxVersion());
}


/**
 * Enacts a copy operation on the specified device data.
 */
template <typename SizeT>
cudaError_t Enactor::Copy(
	void *d_dest,
	void *d_src,
	SizeT num_bytes,
	int max_grid_size)
{
	return Copy<UNKNOWN_SIZE>(d_dest, d_src, num_bytes, max_grid_size);
}



}// namespace copy
}// namespace b40c

