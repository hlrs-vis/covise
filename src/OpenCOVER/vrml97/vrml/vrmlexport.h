/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRML_EXPORT_H
#define VRML_EXPORT_H

/* ---------------------------------------------------------------------- */

/* defines for windows dll export */

#if defined(_WIN32) && !defined(NODLL)
#if defined(VRML_EXPORT) && defined(_WIN32)
#define VRMLEXPORT __declspec(dllexport)
#else
#ifdef _WIN32
#define VRMLEXPORT __declspec(dllimport)
#endif
#endif

#elif defined(__GNUC__) && __GNUC__ >= 4 && !defined(CO_ia64icc)
#define VRMLEXPORT __attribute__((visibility("default")))
#define VRMLIMPORT VRMLEXPORT

#else
#define VRMLEXPORT
#endif
#endif /* VRML_EXPORT_H */
