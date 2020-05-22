/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-----------------------------------------------------------------------------
/** @file  ObservationLog.h
*	@brief This file provides the exported methods.
*
*   This file provides the exported methods which are available outside of the library.
*/
//-----------------------------------------------------------------------------

#pragma once

#if defined(observationCOVER_EXPORTS)
#  define OBSERVATION_COVER_SHARED_EXPORT Q_DECL_EXPORT   //! Export of the dll-functions
#else
#  define OBSERVATION_COVER_SHARED_EXPORT Q_DECL_IMPORT   //! Import of the dll-functions
#endif

#include "Interfaces/observationInterface.h"


