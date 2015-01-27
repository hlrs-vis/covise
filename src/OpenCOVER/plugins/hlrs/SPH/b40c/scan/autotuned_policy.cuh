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
 * Autotuned scan policy
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

#include <b40c/scan/upsweep/kernel.cuh>
#include <b40c/scan/spine/kernel.cuh>
#include <b40c/scan/downsweep/kernel.cuh>
#include <b40c/scan/policy.cuh>

namespace b40c {
namespace scan {


/******************************************************************************
 * Genre enumerations to classify problems by
 ******************************************************************************/

/**
 * Enumeration of problem-size genres that we may have tuned for
 */
enum ProbSizeGenre
{
	UNKNOWN_SIZE = -1,			// Not actually specialized on: the enactor should use heuristics to select another size genre
	SMALL_SIZE,					// Tuned @ 128KB input
	LARGE_SIZE					// Tuned @ 128MB input
};


/**
 * Enumeration of architecture-family genres that we have tuned for below
 */
enum ArchGenre
{
	SM20 	= 200,
	SM13	= 130,
	SM10	= 100
};


/**
 * Enumeration of type size genres
 */
enum TypeSizeGenre
{
	TINY_TYPE,
	SMALL_TYPE,
	MEDIUM_TYPE,
	LARGE_TYPE
};


/**
 * Autotuning policy genre, to be specialized
 */
template <
	// Problem and machine types
	typename ProblemType,
	int CUDA_ARCH,

	// Genres to specialize upon
	ProbSizeGenre PROB_SIZE_GENRE,
	ArchGenre ARCH_GENRE,
	TypeSizeGenre TYPE_SIZE_GENRE,
	TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre;


/******************************************************************************
 * Classifiers for identifying classification genres
 ******************************************************************************/

/**
 * Classifies a given CUDA_ARCH into an architecture-family genre
 */
template <int CUDA_ARCH>
struct ArchClassifier
{
	static const ArchGenre GENRE 			=	// (CUDA_ARCH < SM13) ? SM10 :			// Haven't tuned for SM1.0 yet
												(CUDA_ARCH < SM20) ? SM13 : SM20;
};


/**
 * Classifies the problem type(s) into a type-size genre
 */
template <typename ProblemType>
struct TypeSizeClassifier
{
	static const int ROUNDED_SIZE			= 1 << util::Log2<sizeof(typename ProblemType::T)>::VALUE;	// Round up to the nearest arch subword

	static const TypeSizeGenre GENRE =		(ROUNDED_SIZE < 2) ? TINY_TYPE :
											(ROUNDED_SIZE < 4) ? SMALL_TYPE :
											(ROUNDED_SIZE < 8) ? MEDIUM_TYPE : LARGE_TYPE;
};


/**
 * Classifies the pointer type into a type-size genre
 */
template <typename ProblemType>
struct PointerSizeClassifier
{
	static const TypeSizeGenre GENRE 		= (sizeof(typename ProblemType::SizeT) < 8) ? MEDIUM_TYPE : LARGE_TYPE;
};


/**
 * Autotuning policy classifier
 */
template <
	typename ProblemType,
	int CUDA_ARCH,
	ProbSizeGenre PROB_SIZE_GENRE>
struct AutotunedClassifier :
	AutotunedGenre<
		ProblemType,
		CUDA_ARCH,
		PROB_SIZE_GENRE,
		ArchClassifier<CUDA_ARCH>::GENRE,
		TypeSizeClassifier<ProblemType>::GENRE,
		PointerSizeClassifier<ProblemType>::GENRE>
{};


/******************************************************************************
 * Autotuned genre specializations
 ******************************************************************************/

//-----------------------------------------------------------------------------
// SM2.0 specializations(s)
//-----------------------------------------------------------------------------

// Large problems, 1B data
template <typename ProblemType, int CUDA_ARCH, TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, LARGE_SIZE, SM20, TINY_TYPE, POINTER_SIZE_GENRE>
	: Policy<ProblemType, SM20, util::io::ld::NONE, util::io::st::NONE, false, true, false, 11,
	  1, 7, 2, 2, 5,
	  5, 2, 0, 5,
	  1, 7, 2, 2, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE_SIZE;
};

// Large problems, 2B data
template <typename ProblemType, int CUDA_ARCH, TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, LARGE_SIZE, SM20, SMALL_TYPE, POINTER_SIZE_GENRE>
	: Policy<ProblemType, SM20, util::io::ld::NONE, util::io::st::NONE, false, true, false, 10,
	  1, 7, 1, 2, 5,
	  5, 2, 0, 5,
	  1, 6, 2, 2, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE_SIZE;
};

// Large problems, 4B data
template <typename ProblemType, int CUDA_ARCH, TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, LARGE_SIZE, SM20, MEDIUM_TYPE, POINTER_SIZE_GENRE>
	: Policy<ProblemType, SM20, util::io::ld::NONE, util::io::st::NONE, false, false, false, 11,
	  3, 9, 2, 0, 5,
	  5, 2, 0, 5,
	  3, 9, 1, 1, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE_SIZE;
};

// All other large problems (tuned at 8B)
template <typename ProblemType, int CUDA_ARCH, TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, LARGE_SIZE, SM20, LARGE_TYPE, POINTER_SIZE_GENRE>
	: Policy<ProblemType, SM20, util::io::ld::NONE, util::io::st::NONE, false, false, false, 8,
	  1, 7, 0, 1, 5,
	  5, 2, 0, 5,
	  1, 5, 1, 2, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE_SIZE;
};


// Small problems, 1B data
template <typename ProblemType, int CUDA_ARCH, TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, SMALL_SIZE, SM20, TINY_TYPE, POINTER_SIZE_GENRE>
	: Policy<ProblemType, SM20, util::io::ld::NONE, util::io::st::NONE, false, false, false, 9,
	  1, 5, 2, 2, 5,
	  5, 2, 0, 5,
	  1, 6, 2, 1, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = SMALL_SIZE;
};

// Small problems, 2B data
template <typename ProblemType, int CUDA_ARCH, TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, SMALL_SIZE, SM20, SMALL_TYPE, POINTER_SIZE_GENRE>
	: Policy<ProblemType, SM20, util::io::ld::NONE, util::io::st::NONE, false, false, false, 9,
	  1, 5, 2, 2, 5,
	  5, 2, 0, 5,
	  1, 5, 2, 2, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = SMALL_SIZE;
};

// Small problems, 4B data
template <typename ProblemType, int CUDA_ARCH, TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, SMALL_SIZE, SM20, MEDIUM_TYPE, POINTER_SIZE_GENRE>
	: Policy<ProblemType, SM20, util::io::ld::NONE, util::io::st::NONE, false, false, false, 8,
	  1, 5, 2, 1, 5,
	  5, 2, 0, 5,
	  1, 5, 2, 1, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = SMALL_SIZE;
};

// All other small problems (tuned at 8B)
template <typename ProblemType, int CUDA_ARCH, TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, SMALL_SIZE, SM20, LARGE_TYPE, POINTER_SIZE_GENRE>
	: Policy<ProblemType, SM20, util::io::ld::NONE, util::io::st::NONE, false, false, false, 7,
	  1, 5, 1, 1, 5,
	  5, 2, 0, 5,
	  1, 5, 0, 2, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = SMALL_SIZE;
};


//-----------------------------------------------------------------------------
// SM1.3 specializations(s)
//-----------------------------------------------------------------------------

// Large problems, 1B data
template <typename ProblemType, int CUDA_ARCH, TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, LARGE_SIZE, SM13, TINY_TYPE, POINTER_SIZE_GENRE>
	: Policy<ProblemType, SM13, util::io::ld::NONE, util::io::st::NONE, false, false, false, 12,
	  1, 8, 2, 2, 5,
	  6, 2, 0, 5,
	  1, 8, 2, 1, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE_SIZE;
};

// Large problems, 2B data
template <typename ProblemType, int CUDA_ARCH, TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, LARGE_SIZE, SM13, SMALL_TYPE, POINTER_SIZE_GENRE>
	: Policy<ProblemType, SM13, util::io::ld::NONE, util::io::st::NONE, false, false, false, 10,
	  1, 7, 1, 1, 5,
	  6, 2, 0, 5,
	  1, 6, 2, 2, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE_SIZE;
};

// Large problems, 4B data
template <typename ProblemType, int CUDA_ARCH, TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, LARGE_SIZE, SM13, MEDIUM_TYPE, POINTER_SIZE_GENRE>
	: Policy<ProblemType, SM13, util::io::ld::NONE, util::io::st::NONE, false, false, false, 8,
	  1, 7, 0, 1, 5,
	  6, 2, 0, 5,
	  1, 7, 1, 0, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE_SIZE;
};

// All other Large problems
template <typename ProblemType, int CUDA_ARCH, TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, LARGE_SIZE, SM13, LARGE_TYPE, POINTER_SIZE_GENRE>
	: Policy<ProblemType, SM13, util::io::ld::NONE, util::io::st::NONE, false, false, false, 8,
	  1, 7, 1, 0, 5,
	  6, 2, 0, 5,
	  1, 7, 1, 0, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE_SIZE;
};



// Small problems, 1B data
template <typename ProblemType, int CUDA_ARCH, TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, SMALL_SIZE, SM13, TINY_TYPE, POINTER_SIZE_GENRE>
	: Policy<ProblemType, SM13, util::io::ld::NONE, util::io::st::NONE, false, false, false, 10,
	  1, 6, 2, 1, 5,
	  6, 2, 0, 5,
	  1, 6, 2, 2, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = SMALL_SIZE;
};

// Small problems, 2B data
template <typename ProblemType, int CUDA_ARCH, TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, SMALL_SIZE, SM13, SMALL_TYPE, POINTER_SIZE_GENRE>
	: Policy<ProblemType, SM13, util::io::ld::NONE, util::io::st::NONE, false, true, false, 9,
	  1, 6, 1, 2, 5,
	  6, 2, 0, 5,
	  1, 5, 2, 2, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = SMALL_SIZE;
};

// Small problems, 4B data
template <typename ProblemType, int CUDA_ARCH, TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, SMALL_SIZE, SM13, MEDIUM_TYPE, POINTER_SIZE_GENRE>
	: Policy<ProblemType, SM13, util::io::ld::NONE, util::io::st::NONE, false, false, false, 8,
	  1, 5, 2, 0, 5,
	  6, 2, 0, 5,
	  1, 6, 1, 1, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = SMALL_SIZE;
};

// All other Small problems
template <typename ProblemType, int CUDA_ARCH, TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<ProblemType, CUDA_ARCH, SMALL_SIZE, SM13, LARGE_TYPE, POINTER_SIZE_GENRE>
	: Policy<ProblemType, SM13, util::io::ld::NONE, util::io::st::NONE, false, false, false, 6,
	  1, 5, 0, 1, 5,
	  6, 2, 0, 5,
	  1, 5, 1, 0, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = SMALL_SIZE;
};



//-----------------------------------------------------------------------------
// SM1.0 specializations(s)
//-----------------------------------------------------------------------------






/******************************************************************************
 * Scan kernel entry points that can derive a tuned granularity type
 * implicitly from the PROB_SIZE_GENRE template parameter.  (As opposed to having
 * the granularity type passed explicitly.)
 *
 * TODO: Section can be removed if CUDA Runtime is fixed to
 * properly support template specialization around kernel call sites.
 ******************************************************************************/

/**
 * Tuned upsweep reduction kernel entry point
 */
template <typename ProblemType, int PROB_SIZE_GENRE>
__launch_bounds__ (
	(AutotunedClassifier<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Upsweep::THREADS),
	(AutotunedClassifier<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Upsweep::MIN_CTA_OCCUPANCY))
__global__ void TunedUpsweepKernel(
	typename ProblemType::T 									*d_in,
	typename ProblemType::T 									*d_spine,
	typename ProblemType::ReductionOp 							scan_op,
	typename ProblemType::IdentityOp 							identity_op,
	util::CtaWorkDistribution<typename ProblemType::SizeT> 		work_decomposition)
{
	// Load the kernel policy type identified by the enum for this architecture
	typedef typename AutotunedClassifier<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Upsweep KernelPolicy;

	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	upsweep::UpsweepPass<KernelPolicy>::Invoke(
		d_in,
		d_spine,
		scan_op,
		identity_op,
		work_decomposition,
		smem_storage);
}

/**
 * Tuned spine scan kernel entry point
 */
template <typename ProblemType, int PROB_SIZE_GENRE>
__launch_bounds__ (
	(AutotunedClassifier<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Spine::THREADS),
	(AutotunedClassifier<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Spine::MIN_CTA_OCCUPANCY))
__global__ void TunedSpineKernel(
	typename ProblemType::T				*d_in,
	typename ProblemType::T				*d_out,
	typename ProblemType::SizeT 		spine_elements,
	typename ProblemType::ReductionOp 	scan_op,
	typename ProblemType::IdentityOp 	identity_op)
{
	// Load the kernel policy type identified by the enum for this architecture
	typedef typename AutotunedClassifier<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Spine KernelPolicy;

	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	spine::SpinePass<KernelPolicy>(
		d_in,
		d_out,
		spine_elements,
		scan_op,
		identity_op,
		smem_storage);
}


/**
 * Tuned downsweep scan kernel entry point
 */
template <typename ProblemType, int PROB_SIZE_GENRE>
__launch_bounds__ (
	(AutotunedClassifier<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Downsweep::THREADS),
	(AutotunedClassifier<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Downsweep::MIN_CTA_OCCUPANCY))
__global__ void TunedDownsweepKernel(
	typename ProblemType::T 				*d_in,
	typename ProblemType::T 				*d_out,
	typename ProblemType::T 				*d_spine,
	typename ProblemType::ReductionOp 		scan_op,
	typename ProblemType::IdentityOp 		identity_op,
	util::CtaWorkDistribution<typename ProblemType::SizeT> work_decomposition)
{
	// Load the kernel policy type identified by the enum for this architecture
	typedef typename AutotunedClassifier<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Downsweep KernelPolicy;

	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	downsweep::DownsweepPass<KernelPolicy>(
		d_in,
		d_out,
		d_spine,
		scan_op,
		identity_op,
		work_decomposition,
		smem_storage);
}


/**
 * Tuned single scan kernel entry point
 */
template <typename ProblemType, int PROB_SIZE_GENRE>
__launch_bounds__ (
	(AutotunedClassifier<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Single::THREADS),
	(AutotunedClassifier<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Single::MIN_CTA_OCCUPANCY))
__global__ void TunedSingleKernel(
	typename ProblemType::T				*d_in,
	typename ProblemType::T				*d_out,
	typename ProblemType::SizeT 		spine_elements,
	typename ProblemType::ReductionOp 	scan_op,
	typename ProblemType::IdentityOp 	identity_op)
{
	// Load the kernel policy type identified by the enum for this architecture
	typedef typename AutotunedClassifier<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Single KernelPolicy;

	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	spine::SpinePass<KernelPolicy>(
		d_in,
		d_out,
		spine_elements,
		scan_op,
		identity_op,
		smem_storage);
}


/******************************************************************************
 * Autotuned policy
 *******************************************************************************/

/**
 * Autotuned policy type
 */
template <
	typename ProblemType,
	int CUDA_ARCH,
	ProbSizeGenre PROB_SIZE_GENRE>
struct AutotunedPolicy :
	AutotunedClassifier<
		ProblemType,
		CUDA_ARCH,
		PROB_SIZE_GENRE>
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename ProblemType::T 			T;
	typedef typename ProblemType::SizeT 		SizeT;
	typedef typename ProblemType::ReductionOp 	ReductionOp;
	typedef typename ProblemType::IdentityOp 	IdentityOp;

	typedef void (*UpsweepKernelPtr)(T*, T*, ReductionOp, IdentityOp, util::CtaWorkDistribution<SizeT>);
	typedef void (*SpineKernelPtr)(T*, T*, SizeT, ReductionOp, IdentityOp);
	typedef void (*DownsweepKernelPtr)(T*, T*, T*, ReductionOp, IdentityOp, util::CtaWorkDistribution<SizeT>);
	typedef void (*SingleKernelPtr)(T*, T*, SizeT, ReductionOp, IdentityOp);

	//---------------------------------------------------------------------
	// Kernel function pointer retrieval
	//---------------------------------------------------------------------

	static UpsweepKernelPtr UpsweepKernel() {
		return TunedUpsweepKernel<ProblemType, PROB_SIZE_GENRE>;
	}

	static SpineKernelPtr SpineKernel() {
		return TunedSpineKernel<ProblemType, PROB_SIZE_GENRE>;
	}

	static DownsweepKernelPtr DownsweepKernel() {
		return TunedDownsweepKernel<ProblemType, PROB_SIZE_GENRE>;
	}

	static SingleKernelPtr SingleKernel() {
		return TunedSingleKernel<ProblemType, PROB_SIZE_GENRE>;
	}
};




}// namespace scan
}// namespace b40c

