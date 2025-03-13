#include "listVRBs.h"
#include <util/string_util.h>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <signal.h>
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

bool processExists(int processID) {
#ifdef _WIN32
    HANDLE hProcess = OpenProcess(SYNCHRONIZE, FALSE, processID);
    if (hProcess != NULL) {
        DWORD w;
        auto retval = WaitForSingleObject(hProcess, 0);
        if (retval == WAIT_TIMEOUT)
        {
            CloseHandle(hProcess);
            return true;
        }
        CloseHandle(hProcess);
        return false;
    }
    // If the error code is access denied, the process exists but we don't have access to open a handle to it.
    std::cerr << "process no access" << std::endl;

    return GetLastError() == ERROR_ACCESS_DENIED;
#else
    return !(kill(processID, SIGUSR1) && errno != EPERM);

#endif
}

MyShmStringVector *getShmPortList()
{
    try
    {
        managed_shared_memory shm(open_only, shmSegName);
        auto x = shm.find<MyShmStringVector>(shmVecName);
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

std::unique_ptr<shm_remove> placeSharedProcessInfo(int tcpPort)
{
    auto remover = std::make_unique<shm_remove>(tcpPort);
    try
    {
        const size_t size = 10000;
        boost::interprocess::permissions unrestricted_permissions;
        unrestricted_permissions.set_unrestricted();
        managed_shared_memory shm(open_or_create, shmSegName, size, 0, unrestricted_permissions);
        // Create allocators
        CharAllocator charallocator(shm.get_segment_manager());
        StringAllocator stringallocator(shm.get_segment_manager());


        // shared memory
        MyShmString mystring(charallocator);
        std::string s = std::to_string(tcpPort) + " " +
                        std::to_string(boost::interprocess::ipcdetail::get_current_process_id()) + " " + getUserName();
        mystring = s.c_str();

        MyShmStringVector *myshmvector = shm.find_or_construct<MyShmStringVector>(shmVecName)(stringallocator);
        myshmvector->push_back(mystring);
    }
    catch (boost::interprocess::interprocess_exception &e)
    {
        std::cerr << "could not register VRB in SHM: " << e.what() << std::endl;
    }
    return remover;
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
            for (auto i = list->begin(); i < list->end();)
            {
                auto p = *i;
                if (p.c_str())
                {
                    auto l = split(p.c_str());
                    if(!processExists(std::stoi(l[1])))
                        i = list->erase(i);
                    else{
                        std::cerr << "VRB running on port " << l[0] << ", pid: " << l[1] << ", from "  << l[2] << std::endl;
                        ++i;
                    }
                }else
                    ++i;
            }
        }
        if (list->empty())
            throw interprocess_exception{""};
    }
    catch (const interprocess_exception &e)
    {
        std::cerr << "No VRB running." << std::endl;
    }
}
