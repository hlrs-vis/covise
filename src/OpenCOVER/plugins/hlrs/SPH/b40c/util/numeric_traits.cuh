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
 * Type traits for numeric types
 ******************************************************************************/

#pragma once


namespace b40c {
namespace util {


enum Representation
{
	NOT_A_NUMBER,
	SIGNED_INTEGER,
	UNSIGNED_INTEGER,
	FLOATING_POINT
};


template <Representation R>
struct BaseTraits
{
	static const Representation REPRESENTATION = R;
};


// Default, non-numeric types
template <typename T> struct NumericTraits : 				BaseTraits<NOT_A_NUMBER> {};

template <> struct NumericTraits<char> : 					BaseTraits<SIGNED_INTEGER> {};
template <> struct NumericTraits<signed char> : 			BaseTraits<SIGNED_INTEGER> {};
template <> struct NumericTraits<short> : 					BaseTraits<SIGNED_INTEGER> {};
template <> struct NumericTraits<int> : 					BaseTraits<SIGNED_INTEGER> {};
template <> struct NumericTraits<long> : 					BaseTraits<SIGNED_INTEGER> {};
template <> struct NumericTraits<long long> : 				BaseTraits<SIGNED_INTEGER> {};

template <> struct NumericTraits<unsigned char> : 			BaseTraits<UNSIGNED_INTEGER> {};
template <> struct NumericTraits<unsigned short> : 			BaseTraits<UNSIGNED_INTEGER> {};
template <> struct NumericTraits<unsigned int> : 			BaseTraits<UNSIGNED_INTEGER> {};
template <> struct NumericTraits<unsigned long> : 			BaseTraits<UNSIGNED_INTEGER> {};
template <> struct NumericTraits<unsigned long long> : 		BaseTraits<UNSIGNED_INTEGER> {};

template <> struct NumericTraits<float> : 					BaseTraits<FLOATING_POINT> {};
template <> struct NumericTraits<double> : 					BaseTraits<FLOATING_POINT> {};


} // namespace util
} // namespace b40c

