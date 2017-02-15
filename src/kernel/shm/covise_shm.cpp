/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include <util/unixcompat.h>
#include "covise_shm.h"
#include <config/CoviseConfig.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>

#ifdef SHARED_MEMORY
#define SYSV_SHMEM
#define POSIX_SHMEM

#ifdef __APPLE__
static bool use_posix = true;
#else
static bool use_posix = true;
#endif
#else
#define USE_PAGEFILE
#endif

#ifdef SHARED_MEMORY
#include <sys/mman.h>
#ifdef SYSV_SHMEM
#include <sys/ipc.h>
#include <sys/shm.h>
#endif

#if defined(POSIX_SHMEM)
#include <fcntl.h>
#include <sys/stat.h>
#endif
#endif

#if defined(__alpha) || defined(_AIX)
extern "C" {
extern unsigned int sleep(unsigned int);
}
#endif

#undef DEBUG

/***********************************************************************\
 **                                                                     **
 **   Shared Memory Class Routines                Version: 1.0          **
 **                                                                     **
 **                                                                     **
 **   Description  : The ShredMemory class handles the operating system **
 **		    part of the shared memory management.              **
 **                                                                     **
 **   Classes      : SharedMemory                                       **
 **                                                                     **
 **   Copyright (C) 1993     by University of Stuttgart                 **
 **                             Computer Center (RUS)                   **
 **                             Allmandring 30                          **
 **                             7000 Stuttgart 80                       **
 **                                                                     **
 **                                                                     **
 **   Author       : A. Wierse   (RUS)                                  **
 **                                                                     **
 **   History      :                                                    **
 **                  15.04.93  Ver 1.0                                  **
 **                  26.05.93           Return-values modified          **
 **                                                                     **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

using namespace covise;

List<MMapEntry> *Malloc_tmp::mmaplist = new List<MMapEntry>;
ShmConfig *ShmConfig::theShmConfig = NULL;
int shmlist_exists = 0;
List<SharedMemory> *SharedMemory::shmlist = 0L;
SharedMemory **SharedMemory::shm_array = 0L;
int SharedMemory::global_seq_no = 0;
SharedMemory *ShmAccess::shm = 0L;
SharedMemory *coShmPtr::shmptr = 0L;
SharedMemory *coShmArray::shmptr = 0L;
//int Message::new_count = 0;
//int Message::delete_count = 0;

covise::SharedMemory *covise::get_shared_memory()
{
    if (shmlist_exists)
    {
        if (covise::SharedMemory::shmlist)
        {
            return covise::SharedMemory::shmlist->get_first();
        }
    }
    else
        return 0L;
    return 0L;
}

ShmConfig *ShmConfig::the()
{
    if (!theShmConfig)
        theShmConfig = new ShmConfig;

    return theShmConfig;
}

size_t ShmConfig::getMallocSize()
{
    return the()->minSegSize;
}

ShmConfig::ShmConfig()
{
// set minimal allocation sizes in bytes
#if defined(__alpha) || defined(_SX)
    minSegSize = 400000;
#elif(defined(__linux__) || defined(__APPLE__))
    if (sizeof(void *) == 8)
        minSegSize = 32 * 1024 * 1024 - 8;
    else
        minSegSize = 8388606;
#elif defined(_sgi64)
    minSegSize = 67108862; // 64MB - 2 (size)
#else
    minSegSize = 16777214; // 16MB - 2
#endif
    bool haveShmSizeConfig = false;
    int minSegSizeConfig = coCoviseConfig::getInt("System.ShmSize", minSegSize, &haveShmSizeConfig);
    if (haveShmSizeConfig)
    {
        minSegSize = minSegSizeConfig;
    }

#ifdef SHARED_MEMORY
    bool haveShmConfig = false;
    std::string shmkind = coCoviseConfig::getEntry("System.ShmKind", &haveShmConfig);
    if (haveShmConfig)
    {
        if (shmkind == "posix")
        {
            use_posix = true;
        }
        else if (shmkind == "sysv")
        {
            use_posix = false;
        }
        else
        {
            std::cerr << "Unknow Shm kind: " << shmkind
                      << ", valid valus are: sysv, posix - using "
                      << (use_posix ? "posix" : "sysv") << std::endl;
        }
    }
#endif

    // try to get out host's config
    char hostname[1024];
    if (gethostname(hostname, 1023) == 0)
    {
        // try to find HostConfig entry of host name

        std::string hostName = hostname;
        std::string hostEntry = std::string("HostConfig.") + hostName;
        std::string memInfo = coCoviseConfig::getEntry(hostEntry);

        // not found: try to remove domain names from back to front
        std::string::size_type lastDot = hostName.find_last_of('.');

        while (memInfo == "" && lastDot != std::string::npos)
        {
            hostName = hostName.substr(0, lastDot);
            memInfo = coCoviseConfig::getEntry("HostConfig." + hostName);
            lastDot = hostName.find_last_of('.', lastDot);
        }

        if (memInfo != "")
        {
            char memory[100], execMode[100], unit[64];
            int timeout;
            float size;

            int readElem = sscanf(memInfo.c_str(), "%s%s%d%f%s", memory, execMode, &timeout, &size, unit);
            if (readElem == 5)
            {
                if (strstr(unit, "M") || strstr(unit, "m"))
                    size *= 1048576;
                else if (strstr(unit, "K") || strstr(unit, "k"))
                    size *= 1024;
                else if (strstr(unit, "G") || strstr(unit, "g"))
                    size *= 1073741824;
            }
            if (readElem >= 4)
                minSegSize = (int)size;
        }
    }
}

#define PERMS 0666
extern int shmlist_exists;
shmCallback *SharedMemory::shmC = NULL;

SharedMemory::SharedMemory(int shm_key, shmSizeType shm_size, int nD)
{
    SharedMemory **tmp_array;
    SharedMemory *last_shm;
    int tmp_perm;
    noDelete = nD;
    data = 0L;
    size = shm_size + 2 * sizeof(int);// seq_nr and key
    next = (SharedMemory *)0L;
    shmstate = invalid;
    global_seq_no++;
#ifdef CRAY
    print_comment(__LINE__, __FILE__, "no shared memory available");
    print_exit(__LINE__, __FILE__, 1);
#else
    key = shm_key;
    tmp_perm = PERMS;
    print_comment(__LINE__, __FILE__, "PERMS = %d", tmp_perm);
#ifdef SHARED_MEMORY
#if defined(SYSV_SHMEM)
    if (!use_posix)
    {
        while ((shmid = shmget(key, size, tmp_perm)) < 0)
        {
            switch (errno)
            {
            case ENOENT:
                cerr << "shared memory key no. " << key << " does not exist\n";
                cerr << "waiting\n";
                sleep(1);
                break;
            case EEXIST:
                cerr << "shared memory key no. " << key << " already in use\n";
                print_exit(__LINE__, __FILE__, 1);
                break;
            case ENOSPC:
                cerr << "maximum number of allowed shared memory "
                     << "identifiers system wide to low\n";
                print_exit(__LINE__, __FILE__, 1);
                break;
            case EINVAL:
                cerr << "system has not appropriate shared memory size: " << size << "\n";
                print_exit(__LINE__, __FILE__, 1);
                break;
            default:
                cerr << "can't get shared memory\n";
                print_exit(__LINE__, __FILE__, 1);
                break;
            }
        }
    }
#endif
#if defined(POSIX_SHMEM)
    if (use_posix)
    {
        char tmp_str[255];
        sprintf(tmp_str, "/covise_shm_%0x", key);
        while ((shmfd = shm_open(tmp_str, O_RDWR, S_IRUSR | S_IWUSR)) == -1)
        {
            cerr << "shm_open file " << key << " does not exist\n";
            cerr << "waiting\n";
            sleep(1);
        }
    }
#endif
#else
#ifdef USE_PAGEFILE
    handle = INVALID_HANDLE_VALUE;
#else
    char tmp_str[255];
    sprintf(tmp_str, "%s\\%d", getenv("tmp"), key);
    while ((handle = CreateFile(tmp_str, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL)) == INVALID_HANDLE_VALUE)
    {
        cerr << "CreateFile file " << key << " does not exist\n";
        cerr << "waiting\n";
        Sleep(1000);
    }
#endif
#endif
    print_comment(__LINE__, __FILE__, "new SharedMemory; key: %x  size: %d", (unsigned)key, size);
    shmstate = valid;
#ifdef SHARED_MEMORY
#if defined(SYSV_SHMEM)
    if (!use_posix)
    {
        data = (char *)shmat(shmid, NULL, 0);
        //if ( data == (char *)-1)
        if (data == NULL)
        {
            perror("Problem with shmat call in new SharedMemory");
            printf("can't attach shared memory\nEXITING!!!\n");
            shmstate = detached;
            exit(0);
        }
        else
            shmstate = attached;
    }
#endif
#if defined(POSIX_SHMEM)
    if (use_posix)
    {
        data = (char *)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, shmfd, 0);
        if (data == MAP_FAILED)
        {
            perror("Problem with mmap call in new SharedMemory 1");
            printf("can't attach shared memory\nEXITING!!!\n");
            shmstate = detached;
            exit(0);
        }
        else
            shmstate = attached;
    }
#endif
#else
    const char *name = NULL;
#ifdef USE_PAGEFILE
    std::stringstream str;
    str << "Local\\covise_shm_" << key;
    std::string s = str.str();
    name = s.c_str();
#endif
    filemap = CreateFileMapping(handle, NULL, PAGE_READWRITE, 0, size, name);
    if (!(data = (char *)MapViewOfFile(filemap, FILE_MAP_ALL_ACCESS, 0, 0, size)))
    {
        print_error(__LINE__, __FILE__, "Not enough disk space for CreateFileMapping file");
        print_exit(__LINE__, __FILE__, 1);
    }
#endif
    seq_no = *(int *)data;
    //    if(global_seq_no != seq_no) {
    //    	print_comment(__LINE__, __FILE__, "wrong SharedMemory seq_no");
    //	print_exit(__LINE__, __FILE__, 1);
    //    }
    if (!shmlist)
    {
        shmlist = new List<SharedMemory>;
        shmlist_exists = 1;
    }
    else
    {
        last_shm = shmlist->get_last();
        if (last_shm) // can be NULL, if all segments have been removed
        {
            last_shm->next = this;
        }
    }
    print_comment(__LINE__, __FILE__, "shmlist->add(this);");
    shmlist->add(this);
    tmp_array = new SharedMemory *[global_seq_no];
    for (int i = 0; i < global_seq_no - 1; i++)
        tmp_array[i] = shm_array[i];
    tmp_array[global_seq_no - 1] = this;
    delete[] shm_array;
    shm_array = tmp_array;

    if (shmC)
    {
        (*shmC)(key, size, data);
    }
#endif
    //   fprintf(stderr,"Proc(%ld) SharedMemory::SharedMemory attach[%d]: KEY=%x, SEGSZ=%d, noDelete=%d\n",
    //                    getpid(), global_seq_no, shm_key,shm_size,nD);
}

SharedMemory::SharedMemory(int *shm_key, shmSizeType shm_size)
{
    SharedMemory *tmpshm;
    SharedMemory **tmp_array;
    SharedMemory *last_shm;
    char tmp_str[255];

    noDelete = 0;
    data = 0L;
    size = shm_size + 2 * sizeof(int); // seq_nr and key
    next = NULL;
    shmstate = invalid;
    seq_no = ++global_seq_no;
    key = *shm_key;
#ifdef CRAY
    shmstate = attached;
    data = new char[size];
    if (data == 0L)
    {
        print_comment(__LINE__, __FILE__, "malloc returns null");
        print_exit(__LINE__, __FILE__, 1);
    }
#ifdef DEBUG
    sprintf(tmp_str, "new SharedMemory with %d bytes", size);
    print_comment(__LINE__, __FILE__, tmp_str);
#endif
#else
    if (key == 0)
    {
        shmlist->reset();
        tmpshm = shmlist->next();
        key = tmpshm->key + global_seq_no - 1;
    }
#ifdef SHARED_MEMORY
#ifdef SYSV_SHMEM
    if (!use_posix)
    {
        while ((shmid = shmget(key, size, PERMS | IPC_CREAT | IPC_EXCL)) < 0)
        {
            switch (errno)
            {
            ///////////////////////////////////////////////////
            case EINVAL:
                cerr << "Shared Memory Problem EINVAL for " << size << " Bytes:\n"
                     << "  The allocated size is larger than the maximum value \n"
                     << "  of the kernel." << endl;
#ifdef __linux__
                cerr << "The administratur can change the maximal shared memory\n"
                     << "size of the system by setting the kernel variable\n"
                     << "kernel.shmmax with the 'sysctl' command without reboot.\n"
                     << endl;
#elif __sgi
                cerr << "The administratur can change the maximal shared memory\n"
                     << "size of the system by setting the kernel variable\n"
                     << "shmmax with the 'systune' command and rebooting.\n"
                     << endl;
#endif
                print_exit(__LINE__, __FILE__, 1);
                break;

            ////////////////////////////////////////////////////
            case EEXIST:
            case EACCES:
                key += 1;
#ifdef DEBUG
                cerr << "shared memory key no. " << (key - 1) << " already in use\n";
                cerr << "incrementing key\n";
#endif
                break;

            ///////////////////////////////////////////////////
            case ENOSPC:
                cerr << "Problem ENOSPC for " << size << " Bytes:\n"
                     << "  A shared memory identifier is to be created but the\n"
                     << "  system-imposed limit on the maximum number of allowed\n"
                     << "  shared memory identifiers system wide would be exceeded."
                     << endl;

                print_exit(__LINE__, __FILE__, 1);
                break;

            ///////////////////////////////////////////////////
            case ENOMEM:
                cerr << "Problem ENOMEM allocating segment for " << size << " Bytes:\n"
                     << "  A shared memory identifier and associated shared memory\n"
                     << "  segment are to be created but the amount of available\n"
                     << "  memory is not sufficient to fill the request."
                     << endl;
                print_exit(__LINE__, __FILE__, 1);
                break;

            ///////////////////////////////////////////////////
            default:
                cerr << "can't allocate get shared memory: " << strerror(errno);
                print_exit(__LINE__, __FILE__, 1);
                break;
            }
        }

        // write shared meory key into file for removal after crash
        char tmp_fname[100];
        sprintf(tmp_fname, "/tmp/covise_shm_%d", getuid());
        FILE *hdl = fopen(tmp_fname, "a+");
        if (hdl)
        {
            fprintf(hdl, "%d %x %d\n", shmid, key, size);
            fclose(hdl);
        }
    }
#endif
#if defined(POSIX_SHMEM)
    if (use_posix)
    {
        char buf[255];
        sprintf(buf, "/covise_shm_%0x", key);
        while ((shmfd = shm_open(buf, O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR)) == -1)
        {
            key++;
            sprintf(buf, "/covise_shm_%0x", key);
        }

        // write shared meory key into file for removal after crash
        char tmp_fname[100];
        sprintf(tmp_fname, "/tmp/covise_shm_%d", getuid());
        FILE *hdl = fopen(tmp_fname, "a+");
        if (hdl)
        {
            fprintf(hdl, "%d %x %d\n", -1, key, size);
            fclose(hdl);
        }
    }
#endif
#else
    sprintf(tmp_str, "%s\\%d", getenv("tmp"), key);
#ifdef USE_PAGEFILE
    handle = INVALID_HANDLE_VALUE;
#else
    while ((handle = CreateFile(tmp_str, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL)) == INVALID_HANDLE_VALUE)
    {
        key++;
        sprintf(tmp_str, "%s\\%d", getenv("tmp"), key);
    }
#endif
#endif
    *shm_key = key;
#ifdef DEBUG
    sprintf(tmp_str, "new SharedMemory; key: %x  size: %d", key, size);
    print_comment(__LINE__, __FILE__, tmp_str);

#endif
    shmstate = valid;
#ifdef SHARED_MEMORY
#ifdef SYSV_SHMEM
    if (!use_posix)
    {
        data = (char *)shmat(shmid, NULL, 0);
        //if ( (int *) (data == (int *) -1)
        if (data == NULL)
        {
            fprintf(stderr, "Error in shmat call: %d = %s",
                    errno, strerror(errno));
            perror("Problem with shmat call in new SharedMemory");
            printf("can't attach shared memory\nEXITING!!!\n");
            shmstate = detached;
            exit(0);
        }
        else
            shmstate = attached;
    }
#endif
#if defined(POSIX_SHMEM)
    if (use_posix)
    {
        if (ftruncate(shmfd, size) == -1)
        {
            perror("ftruncate for shmem failed");
            exit(1);
        }
        data = (char *)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, shmfd, 0);
        if (data == MAP_FAILED)
        {
            perror("Problem with mmap call in new SharedMemory 2");
            printf("can't attach shared memory\nEXITING!!!\n");
            shmstate = detached;
            exit(0);
        }
        else
            shmstate = attached;
    }
#endif
#else
    const char *name = NULL;
#ifdef USE_PAGEFILE
	for (;;)
	{
		std::stringstream str;
		str << "Local\\covise_shm_" << key;
		std::string s = str.str();
		name = s.c_str();
#endif
		filemap = CreateFileMapping(handle, NULL, PAGE_READWRITE, 0, size, name);
		if (filemap == NULL)
		{
			print_error(__LINE__, __FILE__, "Not enough disk space for CreateFileMapping file");
			print_exit(__LINE__, __FILE__, 1);
		}
#ifdef USE_PAGEFILE
		if (GetLastError() != ERROR_ALREADY_EXISTS)
			break;
		++key;
	}
	*shm_key = key;
#endif
	if (!(data = (char *)MapViewOfFile(filemap, FILE_MAP_ALL_ACCESS, 0, 0, size)))
	{
		print_error(__LINE__, __FILE__, "Not enough disk space for CreateFileMapping file");
		print_exit(__LINE__, __FILE__, 1);
	}
#endif
#endif
    *(int *)data = seq_no;
    *(int *)(&data[sizeof(int)]) = key;

    if (!shmlist)
    {
        shmlist = new List<SharedMemory>;
        shmlist_exists = 1;
    }
    else
    {
        last_shm = shmlist->get_last();
        last_shm->next = this;
    }
    shmlist->add(this);
    tmp_array = new SharedMemory *[global_seq_no];
    for (int i = 0; i < global_seq_no - 1; i++)
        tmp_array[i] = shm_array[i];
    tmp_array[global_seq_no - 1] = this;
    delete[] shm_array;
    shm_array = tmp_array;
    if (shmC)
    {
        (*shmC)(key, size, data);
    }

    //    fprintf(stderr,"Proc(%ld) SharedMemory::SharedMemory (create): KEY=%x, SEGSZ=%d\n",
    //                    getpid(),key,size);
}

SharedMemory::~SharedMemory()
{
#ifdef CRAY
    print_comment(__LINE__, __FILE__, "in ~SharedMemory");
    delete[] data;
#else
    SharedMemory *ptr;
#ifdef SHARED_MEMORY
#ifdef SYSV_SHMEM
    if (!use_posix)
    {
        if (shmdt(data) < 0)
        {
            print_comment(__LINE__, __FILE__, "can't detach shared memory");
        }
        else
        {
            shmstate = detached;
        }
        if (!noDelete)
        {
            if (shmctl(shmid, IPC_RMID, (struct shmid_ds *)0) < 0)
            {
                print_comment(__LINE__, __FILE__, "can't remove shared memory");
            }
        }
    }
#endif
#if defined(POSIX_SHMEM)
    if (use_posix)
    {
        if (munmap(data, size) == -1)
        {
            perror("error in munmap()\n");
        }
        else
        {
            shmstate = detached;
        }
        close(shmfd);

        if (!noDelete)
        {
            char tmp_str[255];
            sprintf(tmp_str, "/covise_shm_%0x", key);
            if (shm_unlink(tmp_str))
            {
                fprintf(stderr, "can't remove shared mem, key=%s: %s\n", tmp_str, strerror(errno));
            }
        }
    }
#endif
#else
    bool ierr;
    ierr = (UnmapViewOfFile(data) != 0);
    if (ierr == 0)

    {
        cerr << "can't detach shared memory (3)  " << GetLastError() << endl;
        //print_comment(__LINE__, __FILE__, "can't detach shared memory");
    }
    else
    {
        shmstate = detached;
    }
    bool pBuf;
    pBuf = (CloseHandle(filemap) != 0);
    if (pBuf == false)
    {
        cerr << "can't close handle filemap " << GetLastError() << endl;
        //print_comment(__LINE__, __FILE__, "can't remove shared memory");
    }
    if (handle != INVALID_HANDLE_VALUE)
    {
        pBuf = (CloseHandle(handle) != 0);
        if (pBuf == false)
        {
            cerr << "can't close handle  " << GetLastError() << endl;
            //print_comment(__LINE__, __FILE__, "can't remove shared memory");
        }
    }

    // try to delete shared memory
    // can only be done if all other processes locking the shared memeory are dead
    char tmp_str[255];
    sprintf(tmp_str, "%s\\%d", getenv("tmp"), key);
    if (!noDelete)
    {
        for (int itt = 0; itt < 5; itt++)
        {
            pBuf = (DeleteFile(tmp_str) != 0);
            if (pBuf == false)
            {
                cerr << "can't remove shared memory (4)  " << GetLastError() << endl;
                sleep(1);
            }

            else
                break;
        }

        if (pBuf == 0)
        {
            // popup a message
            /*   TCHAR szBuf[80];
		   LPVOID lpMsgBuf;
		   DWORD dw = GetLastError();
		   LPTSTR lpszFunction[10000];

		   FormatMessage(
			   FORMAT_MESSAGE_ALLOCATE_BUFFER |
			   FORMAT_MESSAGE_FROM_SYSTEM,
			   NULL,
			   dw,
			    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
			   (LPTSTR) &lpMsgBuf,
			   0, NULL );

		   wsprintf(szBuf,
			   "%s failed with error %d: %s",
			   lpszFunction, dw, lpMsgBuf);

		   MessageBox(NULL, szBuf, "Error", MB_OK);

         print_comment(__LINE__, __FILE__, "can't remove shared memory");*/
        }
    }
#endif

    //    cerr << "in ~SharedMemory\n";

    //    shmlist->reset();
    //    while(ptr = shmlist->next())
    //	if(ptr->seq_no == seq_no)
    //	    shmlist->remove(ptr);
    if (shm_array[global_seq_no - 1] != this)
    {
        // oops, this is not the last shared memory segment, so search for it and set its value in shm_array to NULL
        int i;
        for (i = 0; i < global_seq_no - 1; i++)
        {
            if (shm_array[i] == this)
            {
                shm_array[i] = NULL;
            }
        }
    }
    else
    {
        global_seq_no--;
    }

    shmlist->remove(this);
    shmlist->reset();
    ptr = shmlist->next();
    delete ptr;
#endif
}

#if defined(__hpux) || defined(_SX)
void *SharedMemory::get_pointer(int no)
{
    if (SharedMemory::shmlist)
    {
        return &(shm_array[no - 1]->data[2 * sizeof(int)]);
    }
    print_comment(__LINE__, __FILE__, "getting pointer: 0x0");
    return (void *)0L;
}
#endif

int SharedMemory::detach()
{
#ifdef CRAY
    shmstate = detached;
    return 1;
#else
#ifdef SHARED_MEMORY
#ifdef SYSV_SHMEM
    if (!use_posix)
    {
        if (shmdt(data) < 0)
        {
            print_comment(__LINE__, __FILE__, "can't detach shared memory");
            return 0;
        }
        else
        {
            shmstate = detached;
            return 1;
        }
    }
#endif
#if defined(POSIX_SHMEM)
    if (use_posix)
    {
        if (munmap(data, size))
        {
            print_comment(__LINE__, __FILE__, "can't detach shared memory");
            return 0;
        }
        else
        {
            shmstate = detached;
        }
        close(shmfd);
        return 1;
    }
#endif
#else
    if (UnmapViewOfFile(data))
    {
        print_comment(__LINE__, __FILE__, "can't detach shared memory");
    }
    else
    {
        shmstate = detached;
    }
    CloseHandle(filemap);
    if (handle != INVALID_HANDLE_VALUE)
        CloseHandle(handle);
    return (1);
#endif
#endif
    return 0;
}

void SharedMemory::get_shmlist(int *ptr)
{
    int i = 0;
#ifdef DEBUG
    char tmp_str[255];
#endif
    SharedMemory *shmptr;

    shmlist->reset();
    while ((shmptr = shmlist->next()))
    {
        ptr[++i] = shmptr->key;
        ptr[++i] = shmptr->size - 2 * sizeof(int); // pointer to a shared memory which is two integers larger seq_nr and key
#ifdef DEBUG
        sprintf(tmp_str, "get_shmlist(%d, %d)", shmptr->key, shmptr->size);
        print_comment(__LINE__, __FILE__, tmp_str);
#endif
    }
    ptr[0] = i / 2;
}

void *Malloc_tmp::large_new(long size)
{
#if defined(CRAY) || defined(_WIN32) || defined(_SX)
    return new char[size];
#else
    MMapEntry *mapentry;

    mapentry = new MMapEntry;
    mapentry->size = size;
    mapentry->fd = open("/dev/zero", O_RDWR, 0666);
    mapentry->ptr = (char *)mmap(NULL, size, (PROT_READ | PROT_WRITE),
                                 MAP_PRIVATE, mapentry->fd, 0);
    mmaplist->add(mapentry);
    return mapentry->ptr;
#endif
}

void Malloc_tmp::large_delete(void *ptr)
{
#if defined(CRAY) || defined(_WIN32) || defined(_SX)
    delete[](char *)ptr;
#else
    MMapEntry *tmpptr;

    mmaplist->reset();
    tmpptr = mmaplist->current();
    while (tmpptr)
    {
        if (tmpptr->ptr == ptr)
        {
            munmap(tmpptr->ptr, tmpptr->size);
            mmaplist->remove(tmpptr);
            return;
        }
        tmpptr = mmaplist->next();
    }
    print_comment(__LINE__, __FILE__, "Malloc_tmp::delete with wrong pointer");
#endif
    return;
}
