// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

//--------------------------------------------------------------------------------------------------
// Detect architecture
//

// Base arch (same for 32-bit and 64-bit)
#define VV_BASE_ARCH_UNKNOWN  9999
#define VV_BASE_ARCH_X86         0
#define VV_BASE_ARCH_ARM      1000

#define VV_ARCH_UNKNOWN          0
#define VV_ARCH_X86             10
#define VV_ARCH_X86_64          11
#define VV_ARCH_ARM             20
#define VV_ARCH_ARM64           21

#if defined(_M_X64) || defined(_M_AMD64) || defined(__amd64__) || defined(__amd64) || defined(__x86_64__) || defined(__x86_64)
#define VV_BASE_ARCH VV_BASE_ARCH_X86
#define VV_ARCH VV_ARCH_X86_64
#elif defined(__arm__) || defined(__arm) || defined(_ARM) || defined(_M_ARM)
#define VV_BASE_ARCH VV_BASE_ARCH_ARM
#define VV_ARCH VV_ARCH_ARM
#elif defined(__aarch64__)
#define VV_BASE_ARCH VV_BASE_ARCH_ARM
#define VV_ARCH VV_ARCH_ARM64
#else
#define VV_BASE_ARCH VV_BASE_ARCH_UNKNOWN
#define VV_ARCH VV_ARCH_UNKNOWN
#endif

//--------------------------------------------------------------------------------------------------
// Detect instruction set
//

// x86 [0-1000)
#define VV_SIMD_ISA_SSE       10
#define VV_SIMD_ISA_SSE2      20
#define VV_SIMD_ISA_SSE3      30
#define VV_SIMD_ISA_SSSE3     31
#define VV_SIMD_ISA_SSE4_1    41
#define VV_SIMD_ISA_SSE4_2    42
#define VV_SIMD_ISA_AVX       50
#define VV_SIMD_ISA_AVX2      60
#define VV_SIMD_ISA_AVX512F   70

// ARM [1000-2000)
#define VV_SIMD_ISA_NEON    1010
#define VV_SIMD_ISA_NEON_FP 1020

#ifndef VV_SIMD_ISA__
#if defined(__AVX512F__)                            && !defined(__CUDACC__) // nvcc does not support AVX intrinsics
#define VV_SIMD_ISA__ VV_SIMD_ISA_AVX512F
#elif defined(__AVX2__)                             && !defined(__CUDACC__)
#define VV_SIMD_ISA__ VV_SIMD_ISA_AVX2
#elif defined(__AVX__)                              && !defined(__CUDACC__)
#define VV_SIMD_ISA__ VV_SIMD_ISA_AVX
#elif defined(__SSE4_2__)
#define VV_SIMD_ISA__ VV_SIMD_ISA_SSE4_2
#elif defined(__SSE4_1__)
#define VV_SIMD_ISA__ VV_SIMD_ISA_SSE4_1
#elif defined(__SSSE3__)
#define VV_SIMD_ISA__ VV_SIMD_ISA_SSSE3
#elif defined(__SSE3__)
#define VV_SIMD_ISA__ VV_SIMD_ISA_SSE3
#elif defined(__SSE2__) || VV_ARCH == VV_ARCH_X86_64 // SSE2 is always available on 64-bit Intel compatible platforms
#define VV_SIMD_ISA__ VV_SIMD_ISA_SSE2
#elif defined(__ARM_NEON) && defined(__ARM_NEON_FP)
#define VV_SIMD_ISA__ VV_SIMD_ISA_NEON_FP
#elif defined(__ARM_NEON)
#define VV_SIMD_ISA__ VV_SIMD_ISA_NEON
#else
#define VV_SIMD_ISA__ 0
#endif
#endif

// Intel Short Vector Math Library available?
#ifndef VV_SIMD_HAS_SVML
#if defined(__INTEL_COMPILER)
#define VV_SIMD_HAS_SVML 1
#endif
#endif

//-------------------------------------------------------------------------------------------------
// Macros to identify SIMD isa availability
//

#define VV_SIMD_ISA_GE(ISA)                                                     \
    ISA - VV_BASE_ARCH >= 0 &&                                                  \
    ISA - VV_BASE_ARCH < 1000 &&                                                \
    VV_SIMD_ISA__ >= ISA

//--------------------------------------------------------------------------------------------------
// SIMD intrinsic #include's
//

#if VV_SIMD_ISA_GE(VV_SIMD_ISA_SSE2)
#include <emmintrin.h>
#endif
#if VV_SIMD_ISA_GE(VV_SIMD_ISA_SSE3)
#include <pmmintrin.h>
#endif
#if VV_SIMD_ISA_GE(VV_SIMD_ISA_SSSE3)
#include <tmmintrin.h>
#endif
#if VV_SIMD_ISA_GE(VV_SIMD_ISA_SSE4_1)
#include <smmintrin.h>
#endif
#if VV_SIMD_ISA_GE(VV_SIMD_ISA_SSE4_2)
#include <nmmintrin.h>
#endif
#if VV_SIMD_ISA_GE(VV_SIMD_ISA_AVX)
#include <immintrin.h>
#endif
#if VV_SIMD_ISA_GE(VV_SIMD_ISA_NEON)
#include <arm_neon.h>
#endif
