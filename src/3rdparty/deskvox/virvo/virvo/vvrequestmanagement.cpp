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

#include "vvdebugmsg.h"
#include "vvrequestmanagement.h"
#include "vvrendercontext.h"
#include "vvtoolshed.h"

#include <algorithm>
#include <GL/glew.h>
#include <sstream>
#include <istream>
#include <fstream>
#include <string>

#define GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX 0x9048
#define GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX 0x9049

class vvGpu::GpuData
{
public:
  GpuData()
  {
    glName = "";
    Xdsp   = "";
    cuda   = false;
    openGL = false;
    cudaDevice = -1;
    wSystem = vvRenderContext::VV_NONE;
  }
  std::string glName;
  std::string Xdsp;
  bool        cuda;
  bool        openGL;
  int         cudaDevice;
  vvRenderContext::WindowingSystem wSystem;

  bool operator == (const GpuData& other) const
  {
    return this->glName     == other.glName
        && this->Xdsp       == other.Xdsp
        && this->cuda       == other.cuda
        && this->openGL     == other.openGL
        && this->cudaDevice == other.cudaDevice
        && this->wSystem    == other.wSystem;
  }
};

std::vector<vvGpu*> gpus;

std::vector<vvGpu*> vvGpu::list()
{
  vvDebugMsg::msg(3, "vvGpu::list() Enter");

  const char* serverEnv = "VV_SERVER_PATH";
  if (getenv(serverEnv))
  {
    std::string filepath = std::string(getenv(serverEnv));
#ifdef WIN32
    filepath = filepath + std::string("\\vserver.config");
#else
    filepath = filepath + std::string("/vserver.config");
#endif

    std::ifstream fin(filepath.c_str());

    if(!fin.is_open())
    {
      std::string errmsg = std::string("vvGpu::list() could not open config file ")+filepath;
      vvDebugMsg::msg(0, errmsg.c_str());
    }

    uint lineNum = 0;
    std::string line;
    while(fin.good())
    {
      lineNum++;
      std::getline(fin, line);

      std::vector<std::string> subStrs = vvToolshed::split(line, "=");
      if(subStrs.size() < 2)
      {
        vvDebugMsg::msg(2, "vvGpu::list() nothing to parse in config file line ", (int)lineNum);
      }
      else
      {
        if(vvToolshed::strCompare("gpu", subStrs[0].c_str()) == 0)
        {
          line.erase(0,line.find_first_of("=",0)+1);
          vvGpu::createGpu(line);
        }
        else if(vvToolshed::strCompare("node", subStrs[0].c_str()) == 0)
        {
          vvDebugMsg::msg(3, "vvGpu::list() skipping unappendand entry \"node\"");
        }
        else
        {
          std::string errmsg = std::string("vvGpu::list() parse error: unknown attribute near: ")+std::string(subStrs[0]);
          vvDebugMsg::msg(0, errmsg.c_str());
        }
      }
    }
  }
  else
  {
    std::string errmsg = std::string("vvGpu::list() Environment variable ")+std::string(serverEnv)+std::string(" not set.");
    vvDebugMsg::msg(0, errmsg.c_str());
  }

  return gpus;
}

vvGpu::vvGpuInfo vvGpu::getInfo(vvGpu *gpu)
{
  vvDebugMsg::msg(3, "vvGpu::getInfo() Enter");

  vvGpuInfo inf = { 0, 0 };

  if(gpu->_data->openGL)
  {
    vvContextOptions co;
    co.displayName = gpu->_data->Xdsp;
    co.doubleBuffering = false;
    co.height = 1;
    co.width = 1;
    co.type = vvContextOptions::VV_WINDOW;

    vvRenderContext context = vvRenderContext(co);
    context.makeCurrent();

    // TODO: this is an NVIDIA extension
    int totalkb = 0;
    glGetIntegerv(GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX,
                  &totalkb);
    inf.totalMem = static_cast<size_t>(totalkb) * 1024;

    int freekb;
    glGetIntegerv(GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX,
                  &freekb);
    inf.freeMem = static_cast<size_t>(freekb) * 1024;
  }
  else if(gpu->_data->cuda)
  {
    // TODO: Implement cuda-case here!
  }

  return inf;
}

vvGpu* vvGpu::createGpu(std::string& data)
{
  vvDebugMsg::msg(3, "vvGpu::createGpu() Enter");

  std::vector<std::string> attributes = vvToolshed::split(data, ",");

  if(attributes.size() > 0)
  {
    vvGpu* gpu = new vvGpu();

    for(std::vector<std::string>::iterator attrib = attributes.begin(); attrib != attributes.end(); attrib++)
    {
      std::vector<std::string> nameValue = vvToolshed::split(*attrib, "=");

      if(nameValue.size() < 2)
      {
        vvDebugMsg::msg(0, "vvGpu::parseGpuData() parse error: attribute value missing");
        continue;
      }

      const char* attribNames[] =
      {
        "name",           // 1
        "xdsp",           // 2
        "cuda",           // 3
        "opengl",         // 4
        "windowingsystem" // 5
      };

      uint attrName = std::find(attribNames, attribNames+5, (nameValue[0])) - attribNames;
      attrName = (attrName < 12) ? (attrName + 1) : 0;

      switch(attrName)
      {
      case 1:
        gpu->_data->glName = nameValue[1];
        break;
      case 2:
        gpu->_data->Xdsp = nameValue[1];
        break;
      case 3:
        gpu->_data->cuda = vvToolshed::strCompare(nameValue[1].c_str(), "true") == 0 ? true : false;
        break;
      case 4:
        gpu->_data->openGL = vvToolshed::strCompare(nameValue[1].c_str(), "true") == 0 ? true : false;
        break;
      case 5:
        if(vvToolshed::strCompare(nameValue[1].c_str(), "X11"))
        {
          gpu->_data->wSystem = vvRenderContext::VV_X11;
        }
        else if(vvToolshed::strCompare(nameValue[1].c_str(), "WGL"))
        {
          gpu->_data->wSystem = vvRenderContext::VV_WGL;
        }
        else if(vvToolshed::strCompare(nameValue[1].c_str(), "COCOA"))
        {
          gpu->_data->wSystem = vvRenderContext::VV_COCOA;
        }
        else
        {
          vvDebugMsg::msg(0, "vvGpu::parseGpuData() parse error: unknown windowingsystem type");
          gpu->_data->wSystem = vvRenderContext::VV_NONE;
        }
        break;
      default:
        std::string errmsg = std::string("vvGpu::createGpu() parse error: unknown attribute near: ")+std::string(nameValue[0]);
        vvDebugMsg::msg(0, errmsg.c_str());
        delete gpu;
        return NULL;
      }
    }

    // check if gpu already known
    vvGpu *found = NULL;
    for(std::vector<vvGpu*>::iterator g = gpus.begin(); g != gpus.end(); g++)
    {
      if(**g == *gpu)
        found = *g;
    }

    if(found)
    {
      delete gpu;
      return found;
    }
    else
    {
      gpus.push_back(gpu);
      return gpu;
    }
  }
  else
  {
    return NULL;
  }
}

void vvGpu::clearGpus()
{
  for(std::vector<vvGpu*>::iterator g = gpus.begin(); g!=gpus.end(); g++)
  {
    delete *g;
  }
  gpus.clear();
}

vvGpu::vvGpu()
{
  _data = new GpuData;
}

vvGpu::~vvGpu()
{
  delete _data;
}

vvGpu& vvGpu::operator = (const vvGpu& src)
{
  (void)src;
  return *this;
}

bool vvGpu::operator == (const vvGpu& other) const
{
  return *this->_data == *other._data;
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
