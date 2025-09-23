// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/memory.h>

#include <fstream>

#if WIN32
  #include <Windows.h>
#else
  #include <sys/sysinfo.h>
#endif

namespace lamure {

const size_t 
get_total_memory()
{
#if WIN32
  MEMORYSTATUSEX memInfo;
  memInfo.dwLength = sizeof(MEMORYSTATUSEX);
  GlobalMemoryStatusEx(&memInfo);
  DWORDLONG totalVirtualMem = memInfo.ullTotalPageFile;
  return totalVirtualMem;
#else
  struct sysinfo mem;
  sysinfo(&mem);
  return mem.totalram * size_t(mem.mem_unit);
#endif
}

const size_t 
get_available_memory(const bool use_buffers_cache)
{
#if WIN32
  MEMORYSTATUSEX statex;
  statex.dwLength = sizeof (statex);
  GlobalMemoryStatusEx (&statex);
  return statex.ullAvailPhys;
#else
    size_t cached_mem = 0;
    if (use_buffers_cache) {
        // get cached memory from Linux's meminfo
        std::ifstream ifs("/proc/meminfo", std::ios::in);
        if (ifs.is_open())
            while (true) {
                std::string s;
                ifs >> s;
                if (ifs.eof()) break;
                if (s == "cached:") {
                    ifs >> cached_mem;
                    break;
                }
            } 
    }

    struct sysinfo mem;
    sysinfo(&mem);  
    
    if (use_buffers_cache) 
        return size_t(mem.freeram + mem.bufferram) * size_t(mem.mem_unit) + 
                     (cached_mem * 1024u);
    else
        return mem.freeram * mem.mem_unit;
#endif
}

const size_t 
get_process_used_memory()
{
#if WIN32
  return get_total_memory() - get_available_memory();
#else
    size_t rss_mem = 0;

    // get physical memory used by the process
    std::ifstream ifs("/proc/self/status", std::ios::in);
    if (ifs.is_open())
        while (true) {
            std::string s;
            ifs >> s;
            if (ifs.eof()) break;
            if (s == "VmRSS:") {
                ifs >> rss_mem;
                break;
            }
        } 
    return rss_mem * 1024u;
#endif
}

} // namespace lamure

