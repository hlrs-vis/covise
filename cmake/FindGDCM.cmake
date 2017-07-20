find_file(GDCM_CONFIG_FILE
   NAMES
   "lib/gdcm-2.2/GDCMConfig.cmake"
   "lib/gdcm-2.4/GDCMConfig.cmake"
   "lib/gdcm-2.5/GDCMConfig.cmake"
   "lib/gdcm-2.6/GDCMConfig.cmake"
   "lib/gdcm-2.7/GDCMConfig.cmake"
   "lib/gdcm-2.8/GDCMConfig.cmake"
   "lib/gdcm-3.0/GDCMConfig.cmake"
   "lib/gdcm-3.1/GDCMConfig.cmake"
   "lib/gdcm-3.2/GDCMConfig.cmake"
)

if (GDCM_CONFIG_FILE)
   include(${GDCM_CONFIG_FILE})
endif()
