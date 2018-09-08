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

#ifndef VV_REQUESTMANAGEMENT_H
#define VV_REQUESTMANAGEMENT_H

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#include <string>
#include <vector>

#if VV_HAVE_BONJOUR
#include "vvbonjour/vvbonjourentry.h"
#endif
#include "vvinttypes.h"
#include "vvrenderer.h"
#include "vvtcpsocket.h"

struct vvServerInfo 
{
  /** Comma separated list of renderers
   */
  std::string renderers;
};

class VIRVOEXPORT vvGpu
{
public:
  struct vvGpuInfo
  {
    size_t freeMem;
    size_t totalMem;

    template<class A>
    void serialize(A& a, unsigned /*version*/)
    {
      a & freeMem;
      a & totalMem;
    }
  };

  /**
    Get a list of known gpus available for this process configured in a config file.
    @return vector-list of vvGpus
    */
  static std::vector<vvGpu*> list();
  /**
    Get the current gpu infos (in kb) of a gpu from either list() or createGpu()
    @return vvGpuInfo with values up to date or set to -1 if not available
    */
  static vvGpuInfo getInfo(vvGpu *gpu);
  /**
    Create a gpu object from a configured string with notation like "key=value,key=value,..."
    @return corresponding vvGpu object or NULL on error
    */
  static vvGpu* createGpu(std::string& data);
  /**
    Clear all gpu entries found in config file and added via createGpu().
    If list() or createGpu() was ever called, this should be called before
    termination of program to prevent memory leaks
    */
  static void clearGpus();

private:
  vvGpu();
  ~vvGpu();
  vvGpu(const vvGpu& rhs);
  vvGpu& operator = (const vvGpu& src);
  bool operator == (const vvGpu& other) const;

  class GpuData;

  GpuData *_data;
};

struct vvResource
{
public:
  vvResource()
  {
    upToDate = true;
    local    = false;
  }

  bool           upToDate;
  bool           local;
  std::string    hostname;
  ushort         port;
#if VV_HAVE_BONJOUR
  vvBonjourEntry bonjourEntry;
#endif
  std::vector<vvGpu::vvGpuInfo> ginfos;

  // vars for future use
  ushort numCPUs;
  uint   cpuMemSize;
};

struct vvRequest
{
  vvRequest()
    : type(vvRenderer::TEXREND)
    , niceness(0)
  {}

  vvRenderer::RendererType type;  ///< requested rendering type
  int         niceness;           ///< niceness priority ranging from -20 to 20
  typedef int numgpus;
  std::vector<numgpus> nodes;     ///< requested amount of nodes with corresponding number of gpus
  std::vector<vvResource*> resources;

  template<class A>
  void serialize(A& a, unsigned /*version*/)
  {
    a & type;
    a & niceness;
    a & nodes;
#if 0 // not serialized!!!
    a & resources;
#endif
  }

  struct Compare
  {
    bool operator ()(const vvRequest* lhs, const vvRequest* rhs) const
    {
      if(lhs->niceness != rhs->niceness)
      {
        return lhs->niceness < rhs->niceness;
      }
      else
      {
        // sort more node-requests first
        return lhs->nodes.size() > rhs->nodes.size();
      }
    }
  };
};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
