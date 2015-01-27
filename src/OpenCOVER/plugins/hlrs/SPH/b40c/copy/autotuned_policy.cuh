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
 * Autotuned copy policy
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

#include <b40c/copy/kernel.cuh>
#include <b40c/copy/policy.cuh>

namespace b40c {
namespace copy {


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
typename SizeT,
	int CUDA_ARCH,

	// Genres to specialize upon
	ProbSizeGenre PROB_SIZE_GENRE,
	ArchGenre ARCH_GENRE,
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
	static const ArchGenre GENRE =			//(CUDA_ARCH < SM13) ? 	SM10 :			// Have not yet tuned configs for SM10-11
											(CUDA_ARCH < SM20) ? 	SM13 :
																	SM20;
};


/**
 * Classifies the pointer type into a type-size genre
 */
template <typename SizeT>
struct PointerSizeClassifier
{
	static const TypeSizeGenre GENRE 		= (sizeof(SizeT) < 8) ? MEDIUM_TYPE : LARGE_TYPE;
};


/**
 * Autotuning policy classifier
 */
template <
	typename SizeT,
	int CUDA_ARCH,
	ProbSizeGenre PROB_SIZE_GENRE>
struct AutotunedClassifier :
	AutotunedGenre<
		SizeT,
		CUDA_ARCH,
		PROB_SIZE_GENRE,
		ArchClassifier<CUDA_ARCH>::GENRE,
		PointerSizeClassifier<SizeT>::GENRE>
{};


/******************************************************************************
 * Autotuned genre specializations
 ******************************************************************************/

//-----------------------------------------------------------------------------
// SM2.0 specializations(s)
//-----------------------------------------------------------------------------

// Large problems
template <typename SizeT, int CUDA_ARCH, TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<SizeT, CUDA_ARCH, LARGE_SIZE, SM20, POINTER_SIZE_GENRE>
	: Policy<unsigned long long, SizeT,
	  SM20, 8, 1, 7, 1, 0,
	  util::io::ld::cg, util::io::st::cg, true, false>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE_SIZE;
};

// Small problems
template <typename SizeT, int CUDA_ARCH, TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<SizeT, CUDA_ARCH, SMALL_SIZE, SM20, POINTER_SIZE_GENRE>
	: Policy<unsigned long long, SizeT,
	  SM20, 6, 1, 6, 0, 0,
	  util::io::ld::cg, util::io::st::cs, false, false>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = SMALL_SIZE;
};



//-----------------------------------------------------------------------------
// SM1.3 specializations(s)
//-----------------------------------------------------------------------------

// Large problems
template <typename SizeT, int CUDA_ARCH, TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<SizeT, CUDA_ARCH, LARGE_SIZE, SM13, POINTER_SIZE_GENRE>
	: Policy<unsigned short, SizeT,
	  SM13, 8, 1, 7, 2, 0,
	  util::io::ld::NONE, util::io::st::NONE, false, false>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE_SIZE;
};

// Small problems
template <typename SizeT, int CUDA_ARCH, TypeSizeGenre POINTER_SIZE_GENRE>
struct AutotunedGenre<SizeT, CUDA_ARCH, SMALL_SIZE, SM13, POINTER_SIZE_GENRE>
	: Policy<unsigned long long, SizeT,
	  SM13, 6, 1, 5, 0, 1,
	  util::io::ld::NONE, util::io::st::NONE, false, false>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = SMALL_SIZE;
};


//-----------------------------------------------------------------------------
// SM1.0 specializations(s)
//-----------------------------------------------------------------------------







/******************************************************************************
 * Copy kernel entry points that can derive a tuned granularity type
 * implicitly from the PROB_SIZE_GENRE template parameter.  (As opposed to having
 * the granularity type passed explicitly.)
 *
 * TODO: Section can be removed if CUDA Runtime is fixed to
 * properly support template specialization around kernel call sites.
 ******************************************************************************/

/**
 * Tuned byte-copy kernel entry point
 */
template <typename SizeT, int PROB_SIZE_GENRE>
__launch_bounds__ (
	(AutotunedClassifier<SizeT, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::THREADS),
	(AutotunedClassifier<SizeT, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::MIN_CTA_OCCUPANCY))
__global__ void TunedKernel(
	void 								*d_in,
	void 								*d_out,
	util::CtaWorkDistribution<SizeT> 	work_decomposition,
	util::CtaWorkProgress				work_progress,
	int 								extra_bytes)
{
	// Load the tuned granularity type identified by the enum for this architecture
	typedef AutotunedClassifier<SizeT, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE> Policy;
	typedef typename Policy::T T;

	T* out = (T*)(d_out);
	T* in = (T*)(d_in);

	SweepPass<Policy, Policy::WORK_STEALING>::Invoke(
		in,
		out,
		work_decomposition,
		work_progress,
		extra_bytes);
}


/******************************************************************************
 * Autotuned policy
 *******************************************************************************/

/**
 * Autotuned policy type
 */
template <
	typename SizeT,
	int CUDA_ARCH,
	ProbSizeGenre PROB_SIZE_GENRE>
struct AutotunedPolicy :
	AutotunedClassifier<
		SizeT,
		CUDA_ARCH,
		PROB_SIZE_GENRE>
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef void (*KernelPtr)(void*, void*, util::CtaWorkDistribution<SizeT>, util::CtaWorkProgress, int);

	//---------------------------------------------------------------------
	// Kernel function pointer retrieval
	//---------------------------------------------------------------------

	static KernelPtr Kernel() {
		return TunedKernel<SizeT, PROB_SIZE_GENRE>;
	}
};



}// namespace copy
}// namespace b40c

