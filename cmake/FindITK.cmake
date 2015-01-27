find_file(ITK_CONFIG_FILE
   NAMES
   "lib/cmake/ITK-4.10/ITKConfig.cmake"
   "lib/cmake/ITK-4.9/ITKConfig.cmake"
   "lib/cmake/ITK-4.8/ITKConfig.cmake"
   "lib/cmake/ITK-4.7/ITKConfig.cmake"
   "lib/cmake/ITK-4.6/ITKConfig.cmake"
   "lib/cmake/ITK-4.5/ITKConfig.cmake"
   "lib/cmake/ITK-4.4/ITKConfig.cmake"
   "lib/cmake/ITK-4.3/ITKConfig.cmake"
   "lib/cmake/ITK-4.2/ITKConfig.cmake"
   "lib/cmake/ITK-4.1/ITKConfig.cmake"
   "lib/cmake/ITK-4.0/ITKConfig.cmake"
)

if (ITK_CONFIG_FILE)
   include(${ITK_CONFIG_FILE})
endif()
