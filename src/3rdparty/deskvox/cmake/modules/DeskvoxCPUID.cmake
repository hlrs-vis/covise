function(deskvox_cpuid name result)
  message(STATUS "Performing CPUID test for " ${name} "...")

  try_run(run_result compile_result ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/vvcpuid.cpp ARGS "${name}")

  if(NOT ${run_result} OR NOT ${compile_result})
    set(${result} 0 PARENT_SCOPE)
  else()
    set(${result} 1 PARENT_SCOPE)
  endif()

  if(${result})
    message(STATUS "  CPU supports ${name} instruction set")
  else()
    message(STATUS "  CPU does NOT support ${name} instruction set")
  endif()
endfunction()

deskvox_cpuid("mmx" HAVE_MMX)
deskvox_cpuid("sse" HAVE_SSE)
deskvox_cpuid("sse2" HAVE_SSE2)
deskvox_cpuid("sse3" HAVE_SSE3)
deskvox_cpuid("ssse3" HAVE_SSSE3)
deskvox_cpuid("sse4.1" HAVE_SSE4_1)
deskvox_cpuid("sse4.2" HAVE_SSE4_2)
deskvox_cpuid("avx" HAVE_AVX)

