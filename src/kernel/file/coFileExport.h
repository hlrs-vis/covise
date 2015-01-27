/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_FILE_EXPORT_H
#define CO_FILE_EXPORT_H

/* ---------------------------------------------------------------------- //
//                                                                        //
//                                                                        //
// Description: DLL EXPORT/IMPORT specification and type definitions      //
//                                                                        //
//                                                                        //
//                                                                        //
//                                                                        //
//                                                                        //
//                                                                        //
//                             (C)2003 HLRS                               //
// Author: Uwe Woessner, Ruth Lang                                        //
// Date:  30.10.03  V1.0                                                  */

#if defined(_WIN32) && !defined(NODLL)
#define COIMPORT __declspec(dllimport)
#define COEXPORT __declspec(dllexport)

#elif defined(__GNUC__) && __GNUC__ >= 4 && !defined(CO_ia64icc)
#define COEXPORT __attribute__((visibility("default")))
#define COIMPORT COEXPORT

#else
#define COIMPORT
#define COEXPORT
#endif

#if defined(COVISE_FILE)
#define FILEEXPORT COEXPORT
#else
#define FILEEXPORT COIMPORT
#endif

#endif
