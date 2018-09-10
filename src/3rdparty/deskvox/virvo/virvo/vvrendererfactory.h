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

#ifndef _VVRENDERERFACTORY_H_
#define _VVRENDERERFACTORY_H_

#include <map>
#include <string>

#include "vvrenderer.h"

/** Create vvRenderers from textual description
 */
class VIRVOEXPORT vvRendererFactory
{
  public:
    typedef std::map<std::string, std::string> Options;

    /**
     * @param vd volume description
     * @param rs renderer state
     * @param t renderer type or vvTexRend's geometry type
     * @param opt options for renderer or vvTexRend's voxel type
     */
    static vvRenderer *create(vvVolDesc *vd,
        const vvRenderState &rs,
        const char *type,
        const Options &opt);

    /**
     * @param vd volume description
     * @param rs renderer state
     * @param t renderer type or vvTexRend's geometry type
     * @param o options for renderer or vvTexRend's voxel type, specify in this format: option1=value1,option2=value2
     * do not use if you want to specify a server side filename
     */
    static vvRenderer *create(vvVolDesc *vd,
        const vvRenderState &rs,
        const char *type=NULL,
        const char *options=NULL);

    /**
     * @param name renderer name string
     * @param arch architecture string further describing the renderer (optional)
     */
    static bool hasRenderer(const std::string& name, std::string const& arch = "");
    static bool hasRenderer(vvRenderer::RendererType type);
};
#endif

//============================================================================
// End of File
//============================================================================
/* vim: set ts=2 sw=2 tw=0: */
