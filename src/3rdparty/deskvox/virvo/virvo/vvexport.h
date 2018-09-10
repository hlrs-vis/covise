// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#ifndef VV_EXPORT_H
#define VV_EXPORT_H

#include "vvcompiler.h"

//-------------------------------------------------------------------------------------------------
// virvo
//

#ifndef VIRVO_STATIC
#   ifdef virvo_EXPORTS
#       define VVAPI VV_DLL_EXPORT
#   else
#       define VVAPI VV_DLL_IMPORT
#   endif
#else
#   define VVAPI
#endif

#define VIRVOEXPORT VVAPI


//-------------------------------------------------------------------------------------------------
// virvo_fileio
//

#ifndef VIRVO_STATIC
#   ifdef virvo_fileio_EXPORTS
#       define VVFILEIOAPI VV_DLL_EXPORT
#   else
#       define VVFILEIOAPI VV_DLL_IMPORT
#   endif
#else
#   define VVFILEIOAPI
#endif

#define VIRVO_FILEIOEXPORT VVFILEIOAPI

#endif // VV_EXPORT_H
