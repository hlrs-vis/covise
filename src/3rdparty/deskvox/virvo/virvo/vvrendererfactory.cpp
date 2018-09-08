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

#include "vvrendererfactory.h"

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#include "vvdebugmsg.h"
#include "vvdynlib.h"
#include "vvvoldesc.h"
#include "vvtexrend.h"
#include "vvsoftsw.h"
#include "vvibrclient.h"
#include "vvimageclient.h"
#ifdef HAVE_VOLPACK
#include "vvrendervp.h"
#endif
#include "vvparbrickrend.h"
#include "vvserbrickrend.h"
#include "vvsocketmap.h"
#include "vvtcpsocket.h"
#include "vvcompiler.h"

#ifdef HAVE_CUDA
#include "cuda/utils.h"
#endif

#include "math/simd/intrinsics.h" // VV_ARCH
#include "private/vvlog.h"

#include <map>
#include <string>
#include <cstring>
#include <algorithm>
#include <vector>
#include <sstream>

#include <boost/lexical_cast.hpp>

#define VV_STRINGIFY(X) VV_STRINGIFY2(X)
#define VV_STRINGIFY2(X) #X

//--------------------------------------------------------------------------------------------------

namespace
{
  template <class Map, class Key>
  typename Map::mapped_type const* Lookup(Map const& map, Key const& key)
  {
    typename Map::const_iterator it = map.find(key);

    if (it == map.end())
      return 0;

    return &it->second;
  }
}

//--------------------------------------------------------------------------------------------------

#ifdef DESKVOX_USE_ASIO

//
// TODO:
// Move into socket map... ?!?!
//

#include "private/connection.h"
#include "private/connection_manager.h"

struct MyConnectionManager
{
    virvo::ConnectionManagerPointer manager;

    MyConnectionManager()
        : manager(virvo::makeConnectionManager())
    {
        manager->run_in_thread();
    }
};

static virvo::ConnectionPointer GetConnection(std::string const& host, unsigned short port)
{
    static MyConnectionManager M;

    // Look for an existing connection
    virvo::ConnectionPointer conn = M.manager->find(host, port);

    if (conn.get() == 0)
    {
        // No connection found.
        // Create a new connection.
        conn = M.manager->connect(host, port);
    }

    return conn;
}

static virvo::ConnectionPointer GetConnection(vvRendererFactory::Options const& options)
{
    // Default host
    std::string host = "127.0.0.1";
    // Default port
    unsigned short port = 31050;

    if (std::string const* s = Lookup(options, "host"))
        host = *s;

    if (std::string const* s = Lookup(options, "port"))
        port = boost::lexical_cast<unsigned short>(*s);

    // Get the connection to the host
    return GetConnection(host, port);
}

#endif

//--------------------------------------------------------------------------------------------------

#define EAX 0x0
#define EBX 0x1
#define ECX 0x2
#define EDX 0x3

#if VV_CXX_MSVC || VV_CXX_MINGW

#include <intrin.h>

static void get_cpuid(int reg[4], int type)
{
  __cpuidex(reg, type, 0);
}

#elif VV_ARCH == VV_ARCH_ARM
// TODO
#else

static void get_cpuid(int reg[4], int type)
{
  __asm__ __volatile__
  (
#ifdef __i386__
   "xchgl %%ebx, %k1; cpuid; xchgl %%ebx, %k1": "=a" (reg[EAX]), "=&r" (reg[EBX]), "=c" (reg[ECX]), "=d" (reg[EDX]) : "a" (type), "c" (0)
#else
    "cpuid": "=a" (reg[EAX]), "=b" (reg[EBX]), "=c" (reg[ECX]), "=d" (reg[EDX]) : "a" (type), "c" (0)
#endif
  );
}

#endif

namespace {

typedef std::map<std::string, vvRenderer::RendererType> RendererTypeMap;
typedef std::map<std::string, std::string> RendererAliasMap;

RendererAliasMap rendererAliasMap;
RendererTypeMap rendererTypeMap;
std::vector<std::string> rayRendArchs;

void init()
{
  if(!rendererTypeMap.empty())
    return;

  vvDebugMsg::msg(3, "vvRendererFactory::init()");

  // used in vview
  rendererAliasMap["0"] = "default";
  rendererAliasMap["1"] = "slices";
  rendererAliasMap["2"] = "cubic2d";
  rendererAliasMap["3"] = "planar";
  rendererAliasMap["4"] = "spherical";
  rendererAliasMap["5"] = "bricks";
  rendererAliasMap["6"] = "soft";
  rendererAliasMap["8"] = "volpack";
  rendererAliasMap["9"] = "rayrendcuda";
  rendererAliasMap["10"] = "rayrendfpu";
  rendererAliasMap["11"] = "rayrendsse2";
  rendererAliasMap["12"] = "rayrendsse4_1";
  rendererAliasMap["13"] = "rayrendavx";
  rendererAliasMap["14"] = "rayrendavx2";
  rendererAliasMap["20"] = "serbrick";
  rendererAliasMap["21"] = "parbrick";
  rendererAliasMap["30"] = "ibr";
  rendererAliasMap["31"] = "image";
  // used in COVER and Inventor renderer
  rendererAliasMap["tex2d"] = "slices";
  rendererAliasMap["slices2d"] = "slices";
  rendererAliasMap["preint"] = "planar";
  rendererAliasMap["fragprog"] = "planar";
  rendererAliasMap["tex"] = "planar";
  rendererAliasMap["tex3d"] = "planar";
  rendererAliasMap["brick"] = "serbrick";

  // TexRend
  rendererTypeMap["default"] = vvRenderer::TEXREND;
  rendererTypeMap["slices"] = vvRenderer::TEXREND;
  rendererTypeMap["cubic2d"] = vvRenderer::TEXREND;
  rendererTypeMap["planar"] = vvRenderer::TEXREND;
  rendererTypeMap["spherical"] = vvRenderer::TEXREND;

  // other renderers
  rendererTypeMap["generic"] = vvRenderer::GENERIC;
  rendererTypeMap["soft"] = vvRenderer::SOFTSW;
  rendererTypeMap["rayrend"] = vvRenderer::RAYREND;
  rendererTypeMap["rayrendcuda"] = vvRenderer::RAYREND;
  rendererTypeMap["rayrendfpu"] = vvRenderer::RAYREND;
  rendererTypeMap["rayrendsse2"] = vvRenderer::RAYREND;
  rendererTypeMap["rayrendsse4_1"] = vvRenderer::RAYREND;
  rendererTypeMap["rayrendavx"] = vvRenderer::RAYREND;
  rendererTypeMap["rayrendavx2"] = vvRenderer::RAYREND;
  rendererTypeMap["volpack"] = vvRenderer::VOLPACK;
  rendererTypeMap["image"] = vvRenderer::REMOTE_IMAGE;
  rendererTypeMap["ibr"] = vvRenderer::REMOTE_IBR;
  rendererTypeMap["serbrick"] = vvRenderer::SERBRICKREND;
  rendererTypeMap["parbrick"] = vvRenderer::PARBRICKREND;

  // ray rend architectures
  rayRendArchs.push_back("cuda");
  rayRendArchs.push_back("fpu");
  rayRendArchs.push_back("sse2");
  rayRendArchs.push_back("sse4_1");
  rayRendArchs.push_back("avx");
  rayRendArchs.push_back("avx2");
}

static bool test_bit(int value, int bit)
{
  return (value & (1 << bit)) != 0;
}


static bool archSupported(std::string const& arch)
{
  if (arch == "fpu")
  {
    return true;
  }

  if (arch == "cuda")
  {
#if VV_HAVE_CUDA
    return virvo::cuda::deviceCount() > 0;
#else
    return false;
#endif
  }

#if VV_ARCH == VV_ARCH_ARM
// TODO
#else

  int reg[4];
  get_cpuid(reg, 0);
  int nids = reg[EAX];

  if (nids >= 1)
  {
    get_cpuid(reg, 1);

    if (arch == "mmx")
      return test_bit(reg[EDX], 23);
    if (arch == "sse")
      return test_bit(reg[EDX], 25);
    if (arch == "sse2")
      return test_bit(reg[EDX], 26);
    if (arch == "sse3")
      return test_bit(reg[ECX], 0);
    if (arch == "ssse3")
      return test_bit(reg[ECX], 9);
    if (arch == "sse4_1")
      return test_bit(reg[ECX], 19);
    if (arch == "sse4_2")
      return test_bit(reg[ECX], 20);
    if (arch == "avx")
      return test_bit(reg[ECX], 28);
  }

  if (nids >= 7)
  {
    get_cpuid(reg, 7);

    if (arch == "avx2")
      return test_bit(reg[EBX], 5);
    if (arch == "avx512f")
      return test_bit(reg[EBX], 16);
    if (arch == "avx512pf")
      return test_bit(reg[EBX], 26);
    if (arch == "avx512er")
      return test_bit(reg[EBX], 27);
    if (arch == "avx512cd")
      return test_bit(reg[EBX], 28);
    if (arch == "avx512vl")
      return test_bit(reg[EBX], 31);
    if (arch == "avx512bw")
      return test_bit(reg[EBX], 30);
    if (arch == "avx512dq")
      return test_bit(reg[EBX], 17);
    if (arch == "avx512ifma")
      return test_bit(reg[EBX], 21);
    if (arch == "avx512vbmi")
      return test_bit(reg[ECX], 1);
  }

#endif

  return false;
}


std::string findRayRendPlugin(std::string const& plugindir, std::string const& arch)
{
  std::stringstream namestr;
#if defined(_WIN32) && !VV_CXX_MINGW
  namestr << "rayrend";
#else
  namestr << "librayrend";
#endif
  if (arch == "best")
  {
    // TODO: determine *best available*
    namestr << "sse4_1";
  }
  else
  {
    namestr << arch;
  }
#define DO_EXPAND(VAL)  VAL ## 1
#define EXPAND(VAL)     DO_EXPAND(VAL)

#if defined(VV_SHARED_LIB_POSTFIX) && (EXPAND(VV_SHARED_LIB_POSTFIX) != 1)
  namestr << VV_STRINGIFY(VV_SHARED_LIB_POSTFIX);
#endif
  namestr << ".";
#if defined(_WIN32) // TODO: resolve issues with cross compilation etc.
  namestr << "dll";
#elif defined __APPLE__
  namestr << "dylib";
#else
  namestr << "so";
#endif
  std::string name = namestr.str();

  VV_LOG(1) << "Locating plugin " << name << " in " << plugindir;

  bool found = false;
  std::vector<std::string> entrylist = virvo::toolshed::entryList(plugindir);
  for (std::vector<std::string>::const_iterator it = entrylist.begin();
       it != entrylist.end(); ++it)
  {
    if (*it == name)
    {
      found = true;
      break;
    }
  }

  if (found)
  {
    std::stringstream pathstr;
    pathstr << plugindir;
#ifdef _WIN32
    pathstr << "\\";
#else
    pathstr << "/";
#endif
    pathstr << name;
    std::string path = pathstr.str();
    return path;
  }
  else
  {
    return std::string();
  }
}


static bool hasRayRenderer(std::string const& arch)
{
  const char* pluginEnv = "VV_PLUGIN_PATH";
  char* pluginPath = getenv(pluginEnv);
  std::string ppath = ".";
  if (pluginPath) {
    ppath = pluginPath;
  }
#define STRINGIFY(x) #x
#ifdef VIRVO_PLUGIN_DIR
  else
  {
	  ppath = STRINGIFY(VIRVO_PLUGIN_DIR) ;
  }
#endif

  if (!archSupported(arch))
  {
    return false;
  }

  // if VV_PLUGIN_PATH not set, try "."
  return !findRayRendPlugin(ppath, arch).empty(); // TODO: e. g. try to open a symbol for validation
}



std::vector<std::string> split(const std::string &s, char delim)
{
  std::vector<std::string> elems;
  std::stringstream ss(s);
  std::string item;
  while(std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

struct ParsedOptions
{
  vvRendererFactory::Options options;
  std::string voxeltype;
  std::vector<vvTcpSocket*> sockets;
  std::string arch; // fpu|sse2|sse4_1|avx|avx2|best (default)
  std::vector<std::string> filenames;
  size_t bricks;
  std::vector<std::string> displays;
  std::string brickrenderer;

  ParsedOptions()
    : voxeltype("default")
    , arch("best")
    , bricks(1)
    , brickrenderer("planar")
  {

  }

  ParsedOptions(std::string str)
    : voxeltype("default")
    , arch("best")
    , bricks(1)
    , brickrenderer("planar")
  {
    std::vector<std::string> optlist = split(str, ',');
    for(std::vector<std::string>::iterator it = optlist.begin();
        it != optlist.end();
        ++it)
    {
      std::vector<std::string> list = split(*it, '=');
      if(list.empty())
        continue;

      std::string &option = list[0];
      std::transform(option.begin(), option.end(), option.begin(), ::tolower);
      switch(list.size())
      {
      case 1:
        singleOption(list[0], "");
        break;
      case 2:
        singleOption(list[0], list[1]);
        break;
      default:
        vvDebugMsg::msg(1, "option value not handled for: ", list[0].c_str());
        break;
      }
    }
  }

  ParsedOptions(const vvRendererFactory::Options &options)
    : options(options)
    , voxeltype("default")
    , arch("best")
    , bricks(1)
    , brickrenderer("planar")
  {
    for(std::map<std::string, std::string>::const_iterator it = options.begin();
        it != options.end();
        ++it)
    {
      singleOption(it->first, it->second);
    }
  }

  bool singleOption(const std::string &opt, const std::string &val)
  {
    if(val.empty())
    {
      voxeltype = val;
    }
    else
    {
      if(opt == "voxeltype")
      {
        voxeltype = val;
      }
      else if(opt == "sockets")
      {
        sockets.clear();
        std::vector<std::string> sockstr = split(val, ',');
        for (std::vector<std::string>::const_iterator it = sockstr.begin();
             it != sockstr.end(); ++it)
        {
          vvTcpSocket* sock = static_cast<vvTcpSocket*>(vvSocketMap::get(atoi((*it).c_str())));
          sockets.push_back(sock);
        }
      }
      else if(opt == "arch")
      {
        arch = val;
      }
      else if(opt == "filename")
      {
        filenames = split(val, ',');
      }
      else if(opt == "bricks")
      {
        bricks = atoi(val.c_str());
      }
      else if (opt == "displays")
      {
        displays = split(val, ',');
      }
      else if (opt == "brickrenderer")
      {
        brickrenderer = val;
      }
      else
      {
        vvDebugMsg::msg(1, "option not handled: ", opt.c_str());
        return false;
      }
    }
    return true;
  }
};

vvRenderer *create(vvVolDesc *vd, const vvRenderState &rs, const char *t, const ParsedOptions &options)
{
  init();

  if(!t || strcmp(t, "") == 0 || strcmp(t, "default") == 0)
    t = getenv("VV_RENDERER");
  if(!t)
    t = "default";
  std::string type(t);
  std::transform(type.begin(), type.end(), type.begin(), ::tolower);
  RendererAliasMap::iterator ait = rendererAliasMap.find(type);
  if(ait != rendererAliasMap.end())
    type = ait->second.c_str();
  RendererTypeMap::iterator it = rendererTypeMap.find(type);
  if(it == rendererTypeMap.end())
  {
    type = "default";
    it = rendererTypeMap.find(type);
  }
  assert(it != rendererTypeMap.end());

  vvTcpSocket* sock = NULL;
  std::string filename;
  std::string arch = options.arch;

  if (options.sockets.size() > 0)
  {
    sock = options.sockets[0];
  }

  if (options.filenames.size() > 0)
  {
    filename = options.filenames[0];
  }

  switch(it->second)
  {
  case vvRenderer::SERBRICKREND:
    return new vvSerBrickRend(vd, rs, options.bricks, options.brickrenderer, options.options);
  case vvRenderer::PARBRICKREND:
  {
    std::vector<vvParBrickRend::Param> params;

    size_t numbricks = options.displays.size();
    numbricks = std::max(options.sockets.size(), numbricks);
    numbricks = std::max(options.filenames.size(), numbricks);

    for (size_t i = 0; i < numbricks; ++i)
    {
      vvParBrickRend::Param p;
      if (options.displays.size() > i)
      {
        p.display = options.displays[i];
      }

      if (options.sockets.size() > i)
      {
        p.sockidx = vvSocketMap::getIndex(options.sockets[i]);
      }

      if (options.filenames.size() > i)
      {
        p.filename = options.filenames[i];
      }

      params.push_back(p);
    }
    return new vvParBrickRend(vd, rs, params, options.brickrenderer, options.options);
  }
  case vvRenderer::GENERIC:
    return new vvRenderer(vd, rs);
  case vvRenderer::REMOTE_IMAGE:
    {
#ifdef DESKVOX_USE_ASIO
    return new vvImageClient(vd, rs, GetConnection(options.options), filename);
#endif
    }
  case vvRenderer::REMOTE_IBR:
    {
#ifdef DESKVOX_USE_ASIO
    return new vvIbrClient(vd, rs, GetConnection(options.options), filename);
#endif
    }
  case vvRenderer::SOFTSW:
    return new vvSoftShearWarp(vd, rs);
#ifdef HAVE_VOLPACK
  case vvRenderer::VOLPACK:
    return new vvVolPack(vd, rs);
#endif
  case vvRenderer::RAYREND:
  {
    // if not specified, try to deduce arch from type string
    if (arch.empty() || arch == "best")
    {
      std::string a;
      size_t idx = type.find("rayrend", 0);
      if (idx != std::string::npos)
      {
        a = type;
        a.replace(idx, 7, "");
      }

      if (!a.empty())
      {
        arch = a;
      }
    }
    const char* pluginEnv = "VV_PLUGIN_PATH";
    char* pluginPath = getenv(pluginEnv);
    std::string ppath = pluginPath == NULL ? "." : pluginPath;

    // if VV_PLUGIN_PATH not set, try "."
    std::string path = findRayRendPlugin(ppath, arch);
    if (!path.empty())
    {
      VV_SHLIB_HANDLE handle = vvDynLib::open(path.c_str(), 1 /* RTLD_LAZY */);
      vvRenderer* (*create)(vvVolDesc*, vvRenderState const&);
      *(void**)(&create) = vvDynLib::sym(handle, "createRayCaster");
      if (create != NULL)
      {
        return (*create)(vd, rs);
      }
      else
      {
        VV_LOG(0) << vvDynLib::error();
      }
    }
    else
    {
      VV_LOG(0) << "No ray casting plugin loaded for architecture " << arch << std::endl;
    }
  }
  // fall through
  case vvRenderer::TEXREND:
  default:
    return new vvTexRend(vd, rs);
  }
  return NULL; // fix warning
}

} // namespace

vvRenderer *vvRendererFactory::create(vvVolDesc *vd, const vvRenderState &rs, const char *t, const char *o)
{
  vvDebugMsg::msg(3, "vvRendererFactory::create: type=", t);
  vvDebugMsg::msg(3, "vvRendererFactory::create: options=", o);

  if(!o)
    o = "default";
  ParsedOptions options(o);

  return ::create(vd, rs, t, options);
}



vvRenderer *vvRendererFactory::create(vvVolDesc *vd,
    const vvRenderState &rs,
    const char *t,
    const vvRendererFactory::Options &opts)
{
  vvDebugMsg::msg(3, "vvRendererFactory::create: type=", t);

  ParsedOptions options(opts);

  return ::create(vd, rs, t, options);
}

bool vvRendererFactory::hasRenderer(const std::string& name, std::string const& arch)
{
  init();

  std::string str = name;
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);

  if (str == "rayrend")
  {
    if (arch.empty())
    {
      return hasRayRenderer("best");
    }
    return hasRayRenderer(arch);
  }
#ifdef HAVE_OPENGL
  else if (str == "planar" || str == "texture" || str == "texrend")
  {
    return hasRenderer(vvRenderer::TEXREND);
  }
#endif
  else if (str == "serbrick")
  {
    return hasRenderer(vvRenderer::SERBRICKREND);
  }
  else if (str == "parbrick")
  {
    return hasRenderer(vvRenderer::PARBRICKREND);
  }
  else
  {
    return false;
  }
}

bool vvRendererFactory::hasRenderer(vvRenderer::RendererType type)
{
  init();

  switch (type)
  {
  case vvRenderer::RAYREND:
  {
    const char* pluginEnv = "VV_PLUGIN_PATH";
    char* pluginPath = getenv(pluginEnv);
    std::string ppath = pluginPath == NULL ? "." : pluginPath;

    for (std::vector<std::string>::const_iterator it = rayRendArchs.begin();
         it != rayRendArchs.end(); ++it)
    {
      // at least one plugin for an arbitrary architecture available?
      if (!findRayRendPlugin(ppath, *it).empty())
      {
        return true;
      }
    }
    return false;
  }
  case vvRenderer::TEXREND:
#ifdef HAVE_OPENGL
    return true;
#else
    return false;
#endif
  case vvRenderer::VOLPACK:
#ifdef HAVE_VOLPACK
    return true;
#else
    return false;
#endif
  default:
    return true;
  }
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0

