/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

typedef struct messageBuf
{
    int ts1;
    int ts2;
    int ts3;
    int ts4;
    unsigned char type;
    unsigned char val[7];
} usbMessage;

int main(int argc, char **argv)
{
    if (argc > 1)
    {
        fprintf(stderr, "opening  %s\n", argv[1]);
        int filedes = open(argv[1], O_RDONLY);
        int values[6];
        float fvalues[6];
        if (filedes >= 0)
        {
            fprintf(stderr, "successfully opened  %s %d\n", argv[1], filedes);
            struct stat statbuf;
            fstat(filedes, &statbuf);
            int bufSize = statbuf.st_blksize;
            int num = 20;
            while (num)
            {
                unsigned char buf[14096];
                memset(buf, 0, bufSize);
                int numRead = read(filedes, buf, bufSize);
                if (numRead > 0)
                {

                    int i = 0;
                    while (i < numRead)
                    {
                        usbMessage *message = (usbMessage *)(buf + i);
                        if (message->type == 1) //Key
                        {
                            fprintf(stderr, "Button %d ", message->val[1]);
                            if (message->val[3])
                            {
                                fprintf(stderr, "Pressed \n");
                            }
                            else
                            {
                                fprintf(stderr, "Released \n");
                            }
                        }
                        else if (message->type == 2) //Motion
                        {

                            int axis = message->val[1];
                            values[axis] = *((int *)(&message->val[3]));
                            if (values[axis] > 9)
                            {
                                if (values[axis] > 410)
                                    values[axis] = 410;
                                fvalues[axis] = (values[axis] - 10) / 400.0;
                            }
                            else if (values[axis] < -9)
                            {
                                if (values[axis] < -410)
                                    values[axis] = -410;
                                fvalues[axis] = (values[axis] + 10) / 400.0;
                                values[axis] = 0;
                            }
                            else
                            {
                                fvalues[axis] = 0.0;
                                values[axis] = 0;
                            }
                            //fprintf(stderr,"Axis %d %d\n",message->val[1],value);
                            for (int n = 0; n < 6; n++)
                            {
                                fprintf(stderr, "%03.3f ", fvalues[n]);
                            }
                        }

                        fprintf(stderr, "\n");

                        i += sizeof(usbMessage);
                    }
                    //num--;
                }
                else
                {
                    perror("read");
                }
            }
            close(filedes);
        }
    }
}
