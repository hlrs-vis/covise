#ifndef COVISE_LIST_VRBS_H
#define COVISE_LIST_VRBS_H
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <memory>


// Typedefs
typedef boost::interprocess::allocator<char, boost::interprocess::managed_shared_memory::segment_manager>
    CharAllocator;
typedef boost::interprocess::basic_string<char, std::char_traits<char>, CharAllocator>
    MyShmString;
typedef boost::interprocess::allocator<MyShmString, boost::interprocess::managed_shared_memory::segment_manager>
    StringAllocator;
typedef boost::interprocess::vector<MyShmString, StringAllocator>
    MyShmStringVector;



MyShmStringVector *getShmPortList();

struct shm_remove
{
    shm_remove(int port);
    ~shm_remove();

    shm_remove(shm_remove &&other);
    shm_remove(const shm_remove &other) = delete;
    shm_remove &operator=(shm_remove &&other);
    shm_remove &operator=(const shm_remove &other) = delete;

private:
    int m_port = 0;
};

std::unique_ptr<shm_remove> placeSharedProcessInfo(int tcpPort);
void cleanShm();
void listShm();

#endif // COVISE_LIST_VRBS_H
