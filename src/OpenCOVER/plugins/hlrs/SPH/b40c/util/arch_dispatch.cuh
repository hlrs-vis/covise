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
 * Thanks!
 * 
 ******************************************************************************/

/******************************************************************************
 * Base class for dynamic architecture dispatch
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>

namespace b40c {
namespace util {


/**
 * Specialization for the device compilation-path.
 *
 * Dispatches to the static method Dispatch::Enact templated by the static CUDA_ARCH.
 * This path drives the actual compilation of kernels, allowing invocation sites to be
 * specialized in type and number by CUDA_ARCH.
 */
template <int CUDA_ARCH, typename Dispatch>
struct ArchDispatch
{
	template<typename Detail>
	static cudaError_t Enact(Detail &detail, int dummy)
	{
		return Dispatch::template Enact<CUDA_ARCH, Detail>(detail);
	}
};


/**
 * Specialization specialization for the host compilation-path.
 *
 * Dispatches to the static method Dispatch::Enact templated by the dynamic
 * ptx_version.  This path does not drive the compilation of kernels.
 */
template <typename Dispatch>
struct ArchDispatch<0, Dispatch>
{
	template<typename Detail>
	static cudaError_t Enact(Detail &detail, int ptx_version)
	{
		// Dispatch
		switch (ptx_version) {
		case 100:
			return Dispatch::template Enact<100, Detail>(detail);
		case 110:
			return Dispatch::template Enact<110, Detail>(detail);
		case 120:
			return Dispatch::template Enact<120, Detail>(detail);
		case 130:
			return Dispatch::template Enact<130, Detail>(detail);
		case 200:
			return Dispatch::template Enact<200, Detail>(detail);
		case 210:
			return Dispatch::template Enact<210, Detail>(detail);
		default:
			// We were compiled for something new: treat it as we would SM2.0
			return Dispatch::template Enact<200, Detail>(detail);
		};
	}
};



} // namespace util
} // namespace b40c

