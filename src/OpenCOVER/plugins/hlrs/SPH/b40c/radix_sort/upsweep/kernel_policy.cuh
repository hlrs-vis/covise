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
 * Configuration policy for partitioning upsweep reduction kernels
 ******************************************************************************/

#pragma once

#include <b40c/partition/upsweep/kernel_policy.cuh>

namespace b40c {
namespace radix_sort {
namespace upsweep {


/**
 * A detailed radix sort upsweep kernel configuration policy type that specializes kernel
 * code for a specific pass. It encapsulates tuning configuration
 * policy details derived from TuningPolicy and PassPolicy.
 */
template <
	typename 		TuningPolicy,
	typename 		PassPolicy>
struct KernelPolicy :
	partition::upsweep::KernelPolicy<TuningPolicy>,
	PassPolicy
{
};
	


} // namespace upsweep
} // namespace radix_sort
} // namespace b40c

