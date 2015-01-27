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
 * Consecutive reduction enactor
 ******************************************************************************/

#pragma once

#include <b40c/util/enactor_base.cuh>
#include <b40c/util/error_utils.cuh>
#include <b40c/util/spine.cuh>
#include <b40c/util/arch_dispatch.cuh>
#include <b40c/util/ping_pong_storage.cuh>

#include <b40c/consecutive_reduction/problem_type.cuh>
#include <b40c/consecutive_reduction/policy.cuh>
#include <b40c/consecutive_reduction/autotuned_policy.cuh>

namespace b40c {
namespace consecutive_reduction {


/**
 * Consecutive reduction enactor class.
 */
class Enactor : public util::EnactorBase
{
protected:

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Temporary device storage needed for scanning value partials produced
	// by separate CTAs
	util::Spine partial_spine;

	// Temporary device storage needed for scanning flag partials produced
	// by separate CTAs
	util::Spine flag_spine;


	//-----------------------------------------------------------------------------
	// Helper structures
	//-----------------------------------------------------------------------------

	template <typename ProblemType>
	friend class Detail;


	//-----------------------------------------------------------------------------
	// Utility Routines
	//-----------------------------------------------------------------------------

    /**
	 * Performs a consecutive reduction pass
	 */
	template <typename Policy, typename Detail>
	cudaError_t EnactPass(Detail &detail);


public:

	/**
	 * Constructor
	 */
	Enactor() {}


	/**
	 * Enacts a consecutive reduction operation on the specified device data.  Uses
	 * a heuristic for selecting an autotuning policy based upon problem size.
	 *
	 * @param problem_storage
	 * 		Instance of b40c::util::PingPongStorage type describing the details of the
	 * 		problem to reduce.
	 * @param num_elements
	 * 		The number of elements in problem_storage to reduce (starting at offset 0)
	 * @param h_num_compacted
	 * 		Host pointer to write the number of elements in the reduced
	 * 		output storage.  May be NULL.
	 * @param d_num_compacted
	 * 		Device pointer to write the number of elements in the reduced
	 * 		output storage.  May be NULL.
	 * @param reduction_op
	 * 		The function or functor type for binary scan, i.e., a type instance
	 * 		that implements "T (const T&, const T&)"
	 * @param equality_op
	 * 		The function or functor type for determining equality amongst
	 * 		PingPongStorage::KeyType instances, a type instance that
	 * 		implements "bool (const &KeyType, const &KeyType)"
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <
		typename PingPongStorage,
		typename SizeT,
		typename ReductionOp,
		typename EqualityOp>
	cudaError_t Reduce(
		PingPongStorage 	&problem_storage,
		SizeT 				num_elements,
		SizeT				*h_num_compacted,
		SizeT				*d_num_compacted,
		ReductionOp 		reduction_op,
		EqualityOp			equality_op,
		int 				max_grid_size = 0);


	/**
	 * Enacts a consecutive reduction operation on the specified device data.  Uses the
	 * specified problem size genre enumeration to select autotuning policy.
	 *
	 * (Using this entrypoint can save compile time by not compiling tuned
	 * kernels for each problem size genre.)
	 *
	 * @param problem_storage
	 * 		Instance of b40c::util::PingPongStorage type describing the details of the
	 * 		problem to reduce.
	 * @param num_elements
	 * 		The number of elements in problem_storage to reduce (starting at offset 0)
	 * @param h_num_compacted
	 * 		Host pointer to write the number of elements in the reduced
	 * 		output storage.  May be NULL.
	 * @param d_num_compacted
	 * 		Device pointer to write the number of elements in the reduced
	 * 		output storage.  May be NULL.
	 * @param reduction_op
	 * 		The function or functor type for binary scan, i.e., a type instance
	 * 		that implements "T (const T&, const T&)"
	 * @param equality_op
	 * 		The function or functor type for determining equality amongst
	 * 		PingPongStorage::KeyType instances, a type instance that
	 * 		implements "bool (const &KeyType, const &KeyType)"
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <
		ProbSizeGenre PROB_SIZE_GENRE,
		typename PingPongStorage,
		typename SizeT,
		typename ReductionOp,
		typename EqualityOp>
	cudaError_t Reduce(
		PingPongStorage 	&problem_storage,
		SizeT 				num_elements,
		SizeT				*h_num_compacted,
		SizeT				*d_num_compacted,
		ReductionOp 		reduction_op,
		EqualityOp			equality_op,
		int 				max_grid_size = 0);


	/**
	 * Enacts a consecutive reduction on the specified device data.  Uses the specified
	 * kernel configuration policy.  (Useful for auto-tuning.)
	 *
	 * @param problem_storage
	 * 		Instance of b40c::util::PingPongStorage type describing the details of the
	 * 		problem to reduce.
	 * @param num_elements
	 * 		The number of elements in problem_storage to reduce (starting at offset 0)
	 * @param h_num_compacted
	 * 		Host pointer to write the number of elements in the reduced
	 * 		output storage.  May be NULL.
	 * @param d_num_compacted
	 * 		Device pointer to write the number of elements in the reduced
	 * 		output storage.  May be NULL.
	 * @param reduction_op
	 * 		The function or functor type for binary scan, i.e., a type instance
	 * 		that implements "T (const T&, const T&)"
	 * @param equality_op
	 * 		The function or functor type for determining equality amongst
	 * 		PingPongStorage::KeyType instances, a type instance that
	 * 		implements "bool (const &KeyType, const &KeyType)"
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename Policy>
	cudaError_t Reduce(
		util::PingPongStorage<
			typename Policy::KeyType,
			typename Policy::ValueType> 	&problem_storage,
		typename Policy::SizeT 				num_elements,
		typename Policy::SizeT				*h_num_compacted,
		typename Policy::SizeT				*d_num_compacted,
		typename Policy::ReductionOp 		reduction_op,
		typename Policy::EqualityOp			equality_op,
		int 								max_grid_size = 0);

};



/******************************************************************************
 * Helper structures
 ******************************************************************************/

/**
 * Type for encapsulating operational details regarding an invocation
 */
template <typename ProblemType>
struct Detail : ProblemType
{
	typedef typename ProblemType::SizeT 		SizeT;
	typedef typename ProblemType::ReductionOp 	ReductionOp;
	typedef typename ProblemType::EqualityOp 	EqualityOp;

	typedef util::PingPongStorage<
		typename ProblemType::KeyType,
		typename ProblemType::ValueType> 		PingPongStorage;

	// Problem data
	Enactor 			*enactor;
	PingPongStorage 	&problem_storage;
	SizeT				num_elements;
	SizeT				*h_num_compacted;
	SizeT				*d_num_compacted;
	ReductionOp			reduction_op;
	EqualityOp			equality_op;
	int			 		max_grid_size;

	// Constructor
	Detail(
		Enactor 			*enactor,
		PingPongStorage 	&problem_storage,
		SizeT 				num_elements,
		SizeT 				*h_num_compacted,
		SizeT 				*d_num_compacted,
		ReductionOp			reduction_op,
		EqualityOp			equality_op,
		int 				max_grid_size = 0) :
			enactor(enactor),
			num_elements(num_elements),
			h_num_compacted(h_num_compacted),
			d_num_compacted(d_num_compacted),
			problem_storage(problem_storage),
			reduction_op(reduction_op),
			equality_op(equality_op),
			max_grid_size(max_grid_size)
	{}

	template <typename Policy>
	cudaError_t EnactPass()
	{
		return enactor->template EnactPass<Policy>(*this);
	}
};


/**
 * Helper structure for resolving and enacting tuning configurations
 *
 * Default specialization for problem type genres
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
			Detail,
			CUDA_ARCH,
			PROB_SIZE_GENRE> AutotunedPolicy;

		// Invoke enactor with type
		return detail.template EnactPass<AutotunedPolicy>();
	}
};


/**
 * Helper structure for resolving and enacting tuning configurations
 *
 * Specialization for UNKNOWN problem type to select other problem type genres
 * based upon problem size, etc.
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
		// Obtain large tuned granularity type
		typedef AutotunedPolicy<
			Detail,
			CUDA_ARCH,
			LARGE_SIZE> LargePolicy;

		// Identify the maximum problem size for which we can saturate loads
		int saturating_load = LargePolicy::Upsweep::TILE_ELEMENTS *
			B40C_SM_CTAS(CUDA_ARCH) *
			detail.enactor->SmCount();

		if (detail.num_elements < saturating_load) {

			// Invoke enactor with small-problem config type
			typedef AutotunedPolicy<
				Detail,
				CUDA_ARCH,
				SMALL_SIZE> SmallPolicy;

			return detail.template EnactPass<SmallPolicy>();
		}

		// Invoke enactor with type
		return detail.template EnactPass<LargePolicy>();
	}
};


/******************************************************************************
 * Enactor Implementation
 ******************************************************************************/

/**
 * Performs a consecutive reduction pass
 */
template <
	typename Policy,
	typename DetailType>
cudaError_t Enactor::EnactPass(DetailType &detail)
{
	typedef typename Policy::KeyType 			KeyType;
	typedef typename Policy::ValueType			ValueType;
	typedef typename Policy::SizeT 				SizeT;
	typedef typename Policy::SpineSizeT			SpineSizeT;

	typedef typename Policy::Upsweep 			Upsweep;
	typedef typename Policy::Spine 				Spine;
	typedef typename Policy::Downsweep 			Downsweep;
	typedef typename Policy::Single 			Single;

	cudaError_t retval = cudaSuccess;
	do {

		// Make sure we have a valid policy
		if (!Policy::VALID) {
			retval = util::B40CPerror(cudaErrorInvalidConfiguration, "Enactor invalid policy", __FILE__, __LINE__);
			break;
		}

		// Kernels
		typename Policy::UpsweepKernelPtr UpsweepKernel = Policy::UpsweepKernel();
		typename Policy::DownsweepKernelPtr DownsweepKernel = Policy::DownsweepKernel();

		// Max CTA occupancy for the actual target device
		int max_cta_occupancy;
		if (retval = MaxCtaOccupancy(
			max_cta_occupancy,
			UpsweepKernel,
			Upsweep::THREADS,
			DownsweepKernel,
			Downsweep::THREADS)) break;

		// Compute sweep grid size
		int sweep_grid_size = GridSize(
			Policy::OVERSUBSCRIBED_GRID_SIZE,
			Upsweep::SCHEDULE_GRANULARITY,
			max_cta_occupancy,
			detail.num_elements,
			detail.max_grid_size);

		// Use single-CTA kernel instead of multi-pass if problem is small enough
		if (detail.num_elements <= Single::TILE_ELEMENTS * 3) {
			sweep_grid_size = 1;
		}

		// Compute spine elements: one element per CTA plus 1 extra for total, rounded
		// up to nearest spine tile size
		SpineSizeT spine_elements = ((sweep_grid_size + 1 + Spine::TILE_ELEMENTS - 1) / Spine::TILE_ELEMENTS) *
			Spine::TILE_ELEMENTS;

		// Obtain a CTA work distribution
		util::CtaWorkDistribution<SizeT> work;
		work.template Init<Downsweep::LOG_SCHEDULE_GRANULARITY>(detail.num_elements, sweep_grid_size);

		if (ENACTOR_DEBUG) {
			if (sweep_grid_size > 1) {
				PrintPassInfo<Upsweep, Spine, Downsweep>(work, spine_elements);
			} else {
				PrintPassInfo<Single>(work);
			}
		}

		if (detail.d_num_compacted == NULL) {
			// If we're to output the compacted sizes to device memory, write out
			// compacted size to the last element of our flag spine instead
			detail.d_num_compacted = ((SizeT*) flag_spine()) + spine_elements - 1;
		}

		if (work.grid_size == 1) {

			// Single-CTA, single-grid operation
			typename Policy::SingleKernelPtr SingleKernel = Policy::SingleKernel();

			SingleKernel<<<1, Single::THREADS, 0>>>(
				detail.problem_storage.d_keys[detail.problem_storage.selector],
				detail.problem_storage.d_keys[detail.problem_storage.selector ^ 1],
				detail.problem_storage.d_values[detail.problem_storage.selector],
				detail.problem_storage.d_values[detail.problem_storage.selector ^ 1],
				detail.d_num_compacted,
				detail.num_elements,
				detail.reduction_op,
				detail.equality_op);

			if (ENACTOR_DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor SingleKernel failed ", __FILE__, __LINE__, ENACTOR_DEBUG))) break;

		} else {

			// Upsweep-downsweep operation
			typename Policy::SpineKernelPtr SpineKernel = Policy::SpineKernel();

			// Make sure our spine is big enough
			if (retval = partial_spine.Setup<ValueType>(spine_elements)) break;
			if (retval = flag_spine.Setup<SizeT>(spine_elements)) break;

			int dynamic_smem[3] = 	{0, 0, 0};
			int grid_size[3] = 		{work.grid_size, 1, work.grid_size};

			// Tuning option: make sure all kernels have the same overall smem allocation
			if (Policy::UNIFORM_SMEM_ALLOCATION) if (retval = PadUniformSmem(dynamic_smem, UpsweepKernel, SpineKernel, DownsweepKernel)) break;

			// Tuning option: make sure that all kernels launch the same number of CTAs)
			if (Policy::UNIFORM_GRID_SIZE) grid_size[1] = grid_size[0];

			// Upsweep scan into spine
			UpsweepKernel<<<grid_size[0], Upsweep::THREADS, dynamic_smem[0]>>>(
				detail.problem_storage.d_keys[detail.problem_storage.selector],
				detail.problem_storage.d_values[detail.problem_storage.selector],
				(ValueType*) partial_spine(),
				(SizeT*) flag_spine(),
				detail.reduction_op,
				detail.equality_op,
				work);

			if (ENACTOR_DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor UpsweepKernel failed ", __FILE__, __LINE__, ENACTOR_DEBUG))) break;

			// Spine scan
			SpineKernel<<<grid_size[1], Spine::THREADS, dynamic_smem[1]>>>(
				(ValueType*) partial_spine(),
				(ValueType*) partial_spine(),
				(SizeT*) flag_spine(),
				(SizeT*) flag_spine(),
				spine_elements,
				detail.reduction_op);

			if (ENACTOR_DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor SpineKernel failed ", __FILE__, __LINE__, ENACTOR_DEBUG))) break;

			// Downsweep scan from spine
			DownsweepKernel<<<grid_size[2], Downsweep::THREADS, dynamic_smem[2]>>>(
				detail.problem_storage.d_keys[detail.problem_storage.selector],
				detail.problem_storage.d_keys[detail.problem_storage.selector ^ 1],
				detail.problem_storage.d_values[detail.problem_storage.selector],
				detail.problem_storage.d_values[detail.problem_storage.selector ^ 1],
				(ValueType*) partial_spine(),
				(SizeT*) flag_spine(),
				detail.d_num_compacted,
				detail.reduction_op,
				detail.equality_op,
				work);

			if (ENACTOR_DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor DownsweepKernel failed ", __FILE__, __LINE__, ENACTOR_DEBUG))) break;
		}

		// Copy out compacted size if necessary
		if (detail.h_num_compacted != NULL) {
			if (util::B40CPerror(cudaMemcpy(
					detail.h_num_compacted,
					detail.d_num_compacted,
					sizeof(SizeT) * 1,
					cudaMemcpyDeviceToHost),
				"Enactor cudaMemcpy d_num_compacted failed: ", __FILE__, __LINE__, ENACTOR_DEBUG)) break;
		}

	} while (0);

	return retval;
}


/**
 * Enacts a consecutive reduction on the specified device data.
 */
template <typename Policy>
cudaError_t Enactor::Reduce(
	util::PingPongStorage<
		typename Policy::KeyType,
		typename Policy::ValueType> 	&problem_storage,
	typename Policy::SizeT 				num_elements,
	typename Policy::SizeT				*h_num_compacted,
	typename Policy::SizeT				*d_num_compacted,
	typename Policy::ReductionOp 		reduction_op,
	typename Policy::EqualityOp			equality_op,
	int 								max_grid_size)
{
	Detail<Policy> detail(
		this,
		problem_storage,
		num_elements,
		h_num_compacted,
		d_num_compacted,
		reduction_op,
		equality_op,
		max_grid_size);

	return EnactPass<Policy>(detail);
}


/**
 * Enacts a consecutive reduction operation on the specified device.
 */
template <
	ProbSizeGenre PROB_SIZE_GENRE,
	typename PingPongStorage,
	typename SizeT,
	typename ReductionOp,
	typename EqualityOp>
cudaError_t Enactor::Reduce(
	PingPongStorage 	&problem_storage,
	SizeT 				num_elements,
	SizeT				*h_num_compacted,
	SizeT				*d_num_compacted,
	ReductionOp 		reduction_op,
	EqualityOp			equality_op,
	int 				max_grid_size)
{
	typedef ProblemType<
		typename PingPongStorage::KeyType,
		typename PingPongStorage::ValueType,
		SizeT,
		ReductionOp,
		EqualityOp> ProblemType;

	Detail<ProblemType> detail(
		this,
		problem_storage,
		num_elements,
		h_num_compacted,
		d_num_compacted,
		reduction_op,
		equality_op,
		max_grid_size);

	return util::ArchDispatch<
		__B40C_CUDA_ARCH__,
		PolicyResolver<PROB_SIZE_GENRE> >::Enact (detail, PtxVersion());
}


/**
 * Enacts a consecutive reduction operation on the specified device data.
 */
template <
	typename PingPongStorage,
	typename SizeT,
	typename ReductionOp,
	typename EqualityOp>
cudaError_t Enactor::Reduce(
	PingPongStorage 	&problem_storage,
	SizeT 				num_elements,
	SizeT				*h_num_compacted,
	SizeT				*d_num_compacted,
	ReductionOp 		reduction_op,
	EqualityOp			equality_op,
	int 				max_grid_size)
{
	return Reduce<UNKNOWN_SIZE>(
		problem_storage,
		num_elements,
		h_num_compacted,
		d_num_compacted,
		reduction_op,
		equality_op,
		max_grid_size);
}




} // namespace consecutive_reduction
} // namespace b40c

