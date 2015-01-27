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
 * Kernel utilities for storing types through global memory with cache modifiers
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/vector_types.cuh>

namespace b40c {
namespace util {
namespace io {


/**
 * Enumeration of data movement cache modifiers.
 */
namespace st {

	enum CacheModifier {
		NONE,			// default (currently wb)
		cg,				// cache global
		wb,				// write back all levels
		cs, 			// cache streaming

		LIMIT
	};

} // namespace st



/**
 * Basic utility for performing modified stores through cache.
 */
template <st::CacheModifier CACHE_MODIFIER>
struct ModifiedStore
{
	/**
	 * Store operation we will provide specializations for
	 */
	template <typename T>
	__device__ __forceinline__ static void St(T val, T *ptr);

	/**
	 * Vec-4 stores for 64-bit types are implemented as two vec-2 stores
	 */
	__device__ __forceinline__ static void St(double4 val, double4* ptr)
	{
		ModifiedStore<CACHE_MODIFIER>::St(*reinterpret_cast<double2*>(&val.x), reinterpret_cast<double2*>(ptr));
		ModifiedStore<CACHE_MODIFIER>::St(*reinterpret_cast<double2*>(&val.z), reinterpret_cast<double2*>(ptr) + 1);
	}

	__device__ __forceinline__ static void St(ulonglong4 val, ulonglong4* ptr)
	{
		ModifiedStore<CACHE_MODIFIER>::St(*reinterpret_cast<ulonglong2*>(&val.x), reinterpret_cast<ulonglong2*>(ptr));
		ModifiedStore<CACHE_MODIFIER>::St(*reinterpret_cast<ulonglong2*>(&val.z), reinterpret_cast<ulonglong2*>(ptr) + 1);
	}

	__device__ __forceinline__ static void St(longlong4 val, longlong4* ptr)
	{
		ModifiedStore<CACHE_MODIFIER>::St(*reinterpret_cast<longlong2*>(&val.x), reinterpret_cast<longlong2*>(ptr));
		ModifiedStore<CACHE_MODIFIER>::St(*reinterpret_cast<longlong2*>(&val.z), reinterpret_cast<longlong2*>(ptr) + 1);
	}
};



#if __CUDA_ARCH__ >= 200

	/**
	 * Specialization for NONE modifier
	 */
	template <>
	template <typename T>
	__device__ __forceinline__ void ModifiedStore<st::NONE>::St(T val, T *ptr)
	{
		*ptr = val;
	}

	/**
	 * Singleton store op
	 */
	#define B40C_STORE(base_type, ptx_type, reg_mod, cast_type, modifier)																	\
		template<> template<> void ModifiedStore<st::modifier>::St(base_type val, base_type* ptr) {											\
			asm("st.global."#modifier"."#ptx_type" [%0], %1;" : : _B40C_ASM_PTR_(ptr), #reg_mod(reinterpret_cast<cast_type&>(val)));			\
		}

	/**
	 * Vector store ops
	 */
	#define B40C_STORE_VEC1(component_type, base_type, ptx_type, reg_mod, cast_type, modifier)																	\
		template<> template<> void ModifiedStore<st::modifier>::St(base_type val, base_type* ptr) {											\
			component_type c = val.x;																											\
			asm("st.global."#modifier"."#ptx_type" [%0], %1;" : : _B40C_ASM_PTR_(ptr), #reg_mod(reinterpret_cast<cast_type&>(c)));			\
		}

	#define B40C_STORE_VEC2(component_type, base_type, ptx_type, reg_mod, cast_type, modifier)																	\
		template<> template<> void ModifiedStore<st::modifier>::St(base_type val, base_type* ptr) {											\
			component_type cx = val.x;																											\
			component_type cy = val.y;																											\
			asm("st.global."#modifier".v2."#ptx_type" [%0], {%1, %2};" : : _B40C_ASM_PTR_(ptr), #reg_mod(reinterpret_cast<cast_type&>(cx)), #reg_mod(reinterpret_cast<cast_type&>(cy)));		\
		}

	#define B40C_STORE_VEC4(component_type, base_type, ptx_type, reg_mod, cast_type, modifier)																	\
		template<> template<> void ModifiedStore<st::modifier>::St(base_type val, base_type* ptr) {											\
			component_type cx = val.x;																											\
			component_type cy = val.y;																											\
			component_type cz = val.z;																											\
			component_type cw = val.w;																											\
			asm("st.global."#modifier".v4."#ptx_type" [%0], {%1, %2, %3, %4};" : : _B40C_ASM_PTR_(ptr), #reg_mod(reinterpret_cast<cast_type&>(cx)), #reg_mod(reinterpret_cast<cast_type&>(cy)), #reg_mod(reinterpret_cast<cast_type&>(cz)), #reg_mod(reinterpret_cast<cast_type&>(cw)));		\
		}


	/**
	 * Defines specialized store ops for only the base type
	 */
	#define B40C_STORE_BASE(base_type, ptx_type, reg_mod, cast_type)		\
		B40C_STORE(base_type, ptx_type, reg_mod, cast_type, cg)		\
		B40C_STORE(base_type, ptx_type, reg_mod, cast_type, wb)		\
		B40C_STORE(base_type, ptx_type, reg_mod, cast_type, cs)


	/**
	 * Defines specialized store ops for the base type and for its derivative vec1 and vec2 types
	 */
	#define B40C_STORE_BASE_ONE_TWO(base_type, dest_type, short_type, ptx_type, reg_mod, cast_type)		\
		B40C_STORE_BASE(base_type, ptx_type, reg_mod, cast_type)										\
																										\
		B40C_STORE_VEC1(base_type, short_type##1, ptx_type, reg_mod, cast_type, cg)						\
		B40C_STORE_VEC1(base_type, short_type##1, ptx_type, reg_mod, cast_type, wb)						\
		B40C_STORE_VEC1(base_type, short_type##1, ptx_type, reg_mod, cast_type, cs)						\
																										\
		B40C_STORE_VEC2(base_type, short_type##2, ptx_type, reg_mod, cast_type, cg)								\
		B40C_STORE_VEC2(base_type, short_type##2, ptx_type, reg_mod, cast_type, wb)								\
		B40C_STORE_VEC2(base_type, short_type##2, ptx_type, reg_mod, cast_type, cs)


	/**
	 * Defines specialized store ops for the base type and for its derivative vec1, vec2, and vec4 types
	 */
	#define B40C_STORE_BASE_ONE_TWO_FOUR(base_type, dest_type, short_type, ptx_type, reg_mod, cast_type)	\
		B40C_STORE_BASE_ONE_TWO(base_type, dest_type, short_type, ptx_type, reg_mod, cast_type)				\
																											\
		B40C_STORE_VEC4(base_type, short_type##4, ptx_type, reg_mod, cast_type, cg)									\
		B40C_STORE_VEC4(base_type, short_type##4, ptx_type, reg_mod, cast_type, wb)									\
		B40C_STORE_VEC4(base_type, short_type##4, ptx_type, reg_mod, cast_type, cs)


#if CUDA_VERSION >= 4000
	#define B40C_REG8		h
	#define B40C_REG16 		h
	#define B40C_CAST8 		short
#else
	#define B40C_REG8		r
	#define B40C_REG16 		r
	#define B40C_CAST8 		char
#endif


	/**
	 * Define cache-modified stores for all 4-byte (and smaller) structures
	 */
	B40C_STORE_BASE_ONE_TWO_FOUR(char, 				char, 			char, 	s8, 	B40C_REG8, 		B40C_CAST8)
	B40C_STORE_BASE_ONE_TWO_FOUR(short, 			short, 			short, 	s16, 	B40C_REG16, 	short)
	B40C_STORE_BASE_ONE_TWO_FOUR(int, 				int, 			int, 	s32, 	r, 				int)
	B40C_STORE_BASE_ONE_TWO_FOUR(unsigned char, 	unsigned char, 	uchar,	u8, 	B40C_REG8, 		unsigned B40C_CAST8)
	B40C_STORE_BASE_ONE_TWO_FOUR(unsigned short,	unsigned short,	ushort,	u16, 	B40C_REG16, 	unsigned short)
	B40C_STORE_BASE_ONE_TWO_FOUR(unsigned int, 		unsigned int, 	uint,	u32, 	r, 				unsigned int)
	B40C_STORE_BASE_ONE_TWO_FOUR(float, 			float, 			float, 	f32, 	f, 				float)

	#if !defined(__LP64__) || (__LP64__ == 0)
	// longs are 64-bit on non-Windows 64-bit compilers
	B40C_STORE_BASE_ONE_TWO_FOUR(long, 				long, 			long, 	s32, 	r, long)
	B40C_STORE_BASE_ONE_TWO_FOUR(unsigned long, 	unsigned long, 	ulong, 	u32, 	r, unsigned long)
	#endif

	B40C_STORE_BASE(signed char, s8, r, unsigned int)		// Only need to define base: char2,char4, etc already defined from char


	/**
	 * Define cache-modified stores for all 8-byte structures
	 */
	B40C_STORE_BASE_ONE_TWO(unsigned long long, 	unsigned long long, 	ulonglong, 	u64, l, unsigned long long)
	B40C_STORE_BASE_ONE_TWO(long long, 				long long, 				longlong, 	s64, l, long long)
	B40C_STORE_BASE_ONE_TWO(double, 				double, 				double, 	s64, l, long long)				// Cast to 64-bit long long a workaround for the fact that the 3.x assembler has no register constraint for doubles

	#if defined(__LP64__)
	// longs are 64-bit on non-Windows 64-bit compilers
	B40C_STORE_BASE_ONE_TWO(long, 					long, 					long, 		s64, l, long)
	B40C_STORE_BASE_ONE_TWO(unsigned long, 			unsigned long, 			ulong, 		u64, l, unsigned long)
	#endif

	/**
	 * Undefine macros
	 */
	#undef B40C_STORE_VEC1
	#undef B40C_STORE_VEC2
	#undef B40C_STORE_VEC4
	#undef B40C_STORE_BASE
	#undef B40C_STORE_BASE_ONE_TWO
	#undef B40C_STORE_BASE_ONE_TWO_FOUR
	#undef B40C_CAST8
	#undef B40C_REG8
	#undef B40C_REG16

#else  //__CUDA_ARCH__

	template <st::CacheModifier WRITE_MODIFIER>
	template <typename T>
	__device__ __forceinline__ void ModifiedStore<WRITE_MODIFIER>::St(T val, T *ptr)
	{
		*ptr = val;
	}

#endif //__CUDA_ARCH__




} // namespace io
} // namespace util
} // namespace b40c

