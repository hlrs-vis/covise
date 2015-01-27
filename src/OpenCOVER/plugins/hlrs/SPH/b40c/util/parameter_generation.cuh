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
 * Functionality for generating parameter lists based upon ranges, suitable
 * for auto-tuning
 ******************************************************************************/

#pragma once

namespace b40c {
namespace util {


/**
 * A recursive tuple type that wraps a constant integer and another tuple type.
 *
 * Can be used to construct types that describe static lists of integer constants.
 */
template <typename NextTuple, int V, int P, int PREV_SIZE = NextTuple::SIZE>
struct ParamTuple
{
	typedef typename NextTuple::next next;

	enum {
		P0 = (PREV_SIZE > 0) ? NextTuple::P0 : P, 	V0 = (PREV_SIZE > 0) ? NextTuple::V0 : V,
		P1 = (PREV_SIZE > 1) ? NextTuple::P1 : P, 	V1 = (PREV_SIZE > 1) ? NextTuple::V1 : V,
		P2 = (PREV_SIZE > 2) ? NextTuple::P2 : P, 	V2 = (PREV_SIZE > 2) ? NextTuple::V2 : V,
		P3 = (PREV_SIZE > 3) ? NextTuple::P3 : P, 	V3 = (PREV_SIZE > 3) ? NextTuple::V3 : V,

		P4 = (PREV_SIZE > 4) ? NextTuple::P4 : P, 	V4 = (PREV_SIZE > 4) ? NextTuple::V4 : V,
		P5 = (PREV_SIZE > 5) ? NextTuple::P5 : P, 	V5 = (PREV_SIZE > 5) ? NextTuple::V5 : V,
		P6 = (PREV_SIZE > 6) ? NextTuple::P6 : P, 	V6 = (PREV_SIZE > 6) ? NextTuple::V6 : V,
		P7 = P, 									V7 = V,

		SIZE = PREV_SIZE + 1
	};
};

// Prev was full: start a new tuple
template <typename NextTuple, int V, int P>
struct ParamTuple<NextTuple, V, P, 8>
{
	typedef NextTuple next;

	enum {
		P0 = P, 	V0 = V,
		P1 = P, 	V1 = V,
		P2 = P, 	V2 = V,
		P3 = P, 	V3 = V,

		P4 = P, 	V4 = V,
		P5 = P, 	V5 = V,
		P6 = P, 	V6 = V,
		P7 = P, 	V7 = V,

		SIZE = 1
	};
};

// Empty tuple (forces new tuple)
struct EmptyTuple
{
	enum {
		SIZE = 8
	};
};


template <typename ParamList, int SEARCH_PARAM>
struct Access
{
	enum {
		VALUE = 	(SEARCH_PARAM == ParamList::P0) ?		(int) ParamList::V0 :
					(SEARCH_PARAM == ParamList::P1) ?		(int) ParamList::V1 :
					(SEARCH_PARAM == ParamList::P2) ?		(int) ParamList::V2 :
					(SEARCH_PARAM == ParamList::P3) ?		(int) ParamList::V3 :
					(SEARCH_PARAM == ParamList::P4) ?		(int) ParamList::V4 :
					(SEARCH_PARAM == ParamList::P5) ?		(int) ParamList::V5 :
					(SEARCH_PARAM == ParamList::P6) ?		(int) ParamList::V6 :
					(SEARCH_PARAM == ParamList::P7) ?		(int) ParamList::V7 :
															Access<typename ParamList::next, SEARCH_PARAM>::VALUE
	};
};

template <int SEARCH_PARAM>
struct Access<EmptyTuple, SEARCH_PARAM>
{
	enum {
		VALUE = 0
	};
};



/**
 * A type generator that sweeps an enumerated sequence of tuning parameters,
 * each of which has an associated (integer) range.  A static list of integer
 * constants is generated for every possible permutation of parameter values.
 * A static callback function on the problem-description type TuneProblemDetail
 * is invoked for each permutation.
 *
 * The range structure for a given parameter may be dependent upon the
 * values selected for tuning parameters occurring prior in the
 * enumeration. (E.g., the range structure for a "raking threads" parameter
 * may incorporate a "cta threads" parameter that is swept earlier in
 * the enumeration to establish an upper bound on raking threads.)
 */
template <
	typename TuneProblemDetail,
	int PARAM,
	int MAX_PARAM,
	template <typename, int> class Ranges>
struct ParamListSweep
{
	// Next parameter increment
	template <int COUNT, int MAX>
	struct Sweep
	{
		template <typename ParamList>
		static void Invoke(TuneProblemDetail &detail)
		{
			// Sweep subsequent parameter
			ParamListSweep<
				TuneProblemDetail,
				PARAM + 1,
				MAX_PARAM,
				Ranges>::template Invoke<ParamTuple<ParamList, COUNT, PARAM> >(detail);

			// Continue sweep with increment of this parameter
			Sweep<COUNT + 1, MAX>::template Invoke<ParamList>(detail);
		}
	};

	// Terminate
	template <int MAX>
	struct Sweep<MAX, MAX>
	{
		template <typename ParamList>
		static void Invoke(TuneProblemDetail &detail) {}
	};

	// Interface
	template <typename ParamList>
	static void Invoke(TuneProblemDetail &detail)
	{
		// Sweep current parameter
		Sweep<
			Ranges<ParamList, PARAM>::MIN,
			Ranges<ParamList, PARAM>::MAX + 1>::template Invoke<ParamList>(detail);

	}
};

// End of currently-generated list
template <
	typename TuneProblemDetail,
	int MAX_PARAM,
	template <typename, int> class Ranges>
struct ParamListSweep <TuneProblemDetail, MAX_PARAM, MAX_PARAM, Ranges>
{
	template <typename ParamList>
	static void Invoke(TuneProblemDetail &detail)
	{
		// Invoke callback
		detail.template Invoke<ParamList>();
	}

};


} // namespace util
} // namespace b40c

