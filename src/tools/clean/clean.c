/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#define _XOPEN_SOURCE 500
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <string.h>
#include <limits.h>
#ifndef CRAY
#include <sys/shm.h>
#include <sys/mman.h>
#endif
#include <signal.h>
#include <fcntl.h>
#include <errno.h>
#define PERMS 0666

int main(int argc, char *argv[])
{
    int key, shmid;
    int size;
    char tmp_fname[100];
    int kill_id;
    FILE *hdl;
    int ihdl;

#ifndef CRAY
    fprintf(stderr, "checking remaining shm segments...\n");
    sprintf(tmp_fname, "/tmp/covise_shm_%d", getuid());
    hdl = fopen(tmp_fname, "r");
    if (hdl)
    {
        while (fscanf(hdl, "%d %x %d", &shmid, &key, &size) != EOF)
        {
            if (shmid == -1)
            {
                // posix shmem
                char tmp_str[100];
                sprintf(tmp_str, "/covise_shm_%x", key);
                if (shm_unlink(tmp_str))
                {
                    printf("removal of %s (key=%x) failed: %s\n", tmp_str, key, strerror(errno));
                }
                else
                {
                    printf("removal of %s (key=%x) succeeded\n", tmp_str, key);
                }
            }
            else
            {
                // sysv shmem
                if (shmctl(shmid, IPC_RMID, (struct shmid_ds *)0) < 0)
                {
                    printf("removal of id=%5d, key=%x failed!!\n", shmid, key);
                }
                else
                {
                    printf("removal of id=%5d, key=%x succeeded!!\n", shmid, key);
                }
            }
        }
    }

    /* make file empty */
    ihdl = open(tmp_fname, O_TRUNC, 0644);
    if (ihdl)
        close(ihdl);

#endif

    if (argc == 2 && !strcmp(argv[1], "-f"))
    {
        printf("trying to remove shared memory\n");
        // try to remove every possible posix shm segment
        unsigned int i;
        for (i = 0xea000000; i < UINT_MAX; ++i)
        {
            char tmp_str[100];
            sprintf(tmp_str, "/covise_shm_%0x", i);
            printf("\r%s", tmp_str);
            if (!shm_unlink(tmp_str))
            {
                printf(": success\n");
            }
        }

        for (i = 0; i < 0xea000000; ++i)
        {
            char tmp_str[100];
            sprintf(tmp_str, "/covise_shm_%0x", i);
            printf("\r%s", tmp_str);
            if (!shm_unlink(tmp_str))
            {
                printf(": success\n");
            }
        }
    }

    fprintf(stderr, "checking remaining processes...\n");

    sprintf(tmp_fname, "/tmp/kill_ids_%d", getuid());
    hdl = fopen(tmp_fname, "r");
    if (hdl)
    {
        while (fscanf(hdl, "%d", &kill_id) != EOF)
        {
            if (kill(kill_id, SIGTERM) == 0)
            {
                printf("killing process no. %d succeeded\n", kill_id);
            }
            else
            {
                if (getpgid(kill_id) != -1 || (errno != ESRCH))
                {
                    printf("killing process no. %d failed, trying harder\n", kill_id);
                    sleep(1);
                    if (kill(kill_id, SIGKILL) == 0)
                    {
                        printf("killing process no. %d with SIGKILL succeeded\n", kill_id);
                    }
                }
            }
        }
        fclose(hdl);
    }
    ihdl = open(tmp_fname, O_TRUNC, 0644);
    if (ihdl)
        close(ihdl);

    return 0;
}
