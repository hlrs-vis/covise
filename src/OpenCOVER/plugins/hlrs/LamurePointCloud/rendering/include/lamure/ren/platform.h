// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef RENDERING_PLATFORM_H_
#define RENDERING_PLATFORM_H_

#ifdef _MSC_VER
  #pragma warning (disable: 4251) // needs to have dll-interface to be used by clients of class
#endif

#if WIN32
  #if defined(LAMURE_RENDERING_LIBRARY)
     #define RENDERING_DLL __declspec( dllexport )
  #else
    #define RENDERING_DLL __declspec( dllimport )
  #endif
#else
   #define RENDERING_DLL
#endif

#endif // RENDERING_PLATFORM_H_