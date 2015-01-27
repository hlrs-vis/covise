# @file
# 
# Simply descend to subdirectories
#
# @author Blasius Czink

if (NOT OPENSSL_FOUND)
   return()
else()
   add_definitions(-DHAVE_OPENSSL)
endif()

ADD_SUBDIRECTORY(CoviseDaemon)
ADD_SUBDIRECTORY(CoviseClient)
