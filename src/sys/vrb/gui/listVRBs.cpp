#include "listVRBs.h"
#include <util/string_util.h>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/allocators/allocator.hpp>

#include <iostream>
using namespace boost::interprocess;



#ifdef _WIN32
#include <windows.h>
#include <Lmcons.h>
#else
    #include <sys/types.h>
    #include <pwd.h>
#endif

const char *shmSegName = "CoviseVrbShmSeg";
const char *shmVecName = "VrbPortList";

std::string getUserName()
{
#ifdef _WIN32
char username[UNLEN+1];
DWORD username_len = UNLEN+1;
GetUserName(username, &username_len);
return username;
#else
    auto id = getuid();
    auto pwd = getpwuid(id);
    return pwd->pw_name;
#endif
}

MyShmStringVector *getShmPortList()
{
    try
    {
        managed_shared_memory shm(open_only, shmSegName);
        auto x = shm.find<MyShmStringVector>(shmVecName);
        std::cerr << "shm.find<MyShmStringVector>(shmVecName)" << std::endl;
        return x.first;
    }
    catch (const interprocess_exception &e)
    {
        return nullptr;
    }
}

shm_remove::shm_remove(int port) : m_port(port) {}
shm_remove::~shm_remove()
{
    if (m_port)
    {
        try
        {
            managed_shared_memory shm(open_only, shmSegName);
            auto list = shm.find<MyShmStringVector>(shmVecName).first;
            assert(list);
            for (auto i = list->begin(); i != list->end();)
            {
                auto l = split(i->c_str());
                if (l[0] == std::to_string(m_port).c_str())
                {
                    i = list->erase(i);
                }
                else
                    ++i;
            }
            if (list->empty())
            {
                shared_memory_object::remove(shmSegName);
            }
        }
        catch (const interprocess_exception &e)
        {
            return;
        }
    }
}
shm_remove::shm_remove(shm_remove &&other) : m_port(other.m_port) { other.m_port = 0; }
shm_remove &shm_remove::operator=(shm_remove &&other)
{
    m_port = other.m_port;
    other.m_port = 0;
    return *this;
}

shm_remove placeSharedProcessInfo(int tcpPort)
{
    const size_t size = 10000;
    managed_shared_memory shm(open_or_create, shmSegName, size);
    shm_remove remover(tcpPort);
    // Create allocators
    CharAllocator charallocator(shm.get_segment_manager());
    StringAllocator stringallocator(shm.get_segment_manager());


    // shared memory
    MyShmString mystring(charallocator);
    std::string s = std::to_string(tcpPort) + " " + std::to_string( boost::interprocess::ipcdetail::get_current_process_id()) + " " + getUserName();
    mystring = s.c_str();

    MyShmStringVector *myshmvector =
        shm.find_or_construct<MyShmStringVector>(shmVecName)(stringallocator);
    myshmvector->push_back(mystring);
    return std::move(remover);
}

void cleanShm()
{
    shared_memory_object::remove(shmSegName);
}

void listShm()
{
    try
    {
        managed_shared_memory shm(open_only, shmSegName);
        auto list = shm.find<MyShmStringVector>(shmVecName).first;
        if (list)
        {
            for (const auto &p : *list)
            {
                if (p.c_str())
                {
                    auto l = split(p.c_str());
                    std::cerr << "VRB running on port " << l[0] << ", pid: " << l[1] << ", from "  << l[2] << std::endl;
                }
            }
        }/*  */
    }
    catch (const interprocess_exception &e)
    {
        std::cerr << "No VRB running." << std::endl;
    }
}