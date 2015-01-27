/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// api specification of the dll
// author: Andre Buchau
// 19.11.2010: file created

#ifdef WIN32
#ifdef INTERFACECOMSOL_EXPORT
#define API_INTERFACECOMSOL __declspec(dllexport)
#else
#define API_INTERFACECOMSOL __declspec(dllimport)
#endif
#else
#define API_INTERFACECOMSOL
#endif
