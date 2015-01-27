/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * Please do NOT introduce the usual #ifndef COIDENT_H_INCLUDED stuff
 * at the beginnig of this file to prevent multiple inclusions, because
 * it is designed to be multiple included!
 */
namespace covise
{

#ifdef COIDENT
#if !defined(NOIDENT) && !defined(CO_hp1020) && !defined(_WIN32) && !defined(__hpux)
#if !(__GNUC__)
#ident COIDENT /* at best use the simple #ident ... */
#elif(__GNUC__ < 4) || (__GNUC_MINOR__ < 4) /* since gcc v4.4, #ident is deprecated */
#ident COIDENT
#endif
#elif defined(CO_hp1020) /* ... which is unknown to HP 10.20  */
#pragma VERSIONID COIDENT
#elif !defined(COIDENT_H_INCLUDED) /* at least an indent in the c files */
#define COIDENT_H_INCLUDED
static const char *cov_ident = COIDENT; /* hopefully not optimized   */
#endif
#undef COIDENT
#endif
}
