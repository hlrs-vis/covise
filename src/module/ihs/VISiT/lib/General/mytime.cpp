#ifndef WIN32
#include <sys/time.h>
#include <unistd.h>
#else
#include <WinSock2.h>
#include <windows.h>
#endif
#include <stdio.h>

double timediff(struct timeval *time1, struct timeval *time2)
{
   double res;

   res = (double) ( time2->tv_sec - time1->tv_sec );
   res += (double) 0.000001 * ( time2->tv_usec - time1->tv_usec );

   return (res);
}
