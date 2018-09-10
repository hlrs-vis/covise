find_package(Pthreads REQUIRED)

deskvox_use_package(Pthreads)

deskvox_add_test(vvmulticast
  vvmulticasttest.cpp
)

deskvox_add_test(vvmulticast2
  vvmulticasttest2.cpp
)
