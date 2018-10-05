/* client.c */
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <errno.h>
#define SERVER_PORT 1234
#include <wiringPi.h>

#define double2ll(value) (*(uint64_t *)(&value))
inline void byteSwap(double &value)
{
    double2ll(value) =
        ((double2ll(value) & 0x00000000000000ffll) << 56) | ((double2ll(value) & 0x000000000000ff00ll) << 40) | ((double2ll(value) & 0x0000000000ff0000ll) << 24) | ((double2ll(value) & 0x00000000ff000000ll) << 8) | ((double2ll(value) & 0x000000ff00000000ll) >> 8) | ((double2ll(value) & 0x0000ff0000000000ll) >> 24) | ((double2ll(value) & 0x00ff000000000000ll) >> 40) | ((double2ll(value) & 0xff00000000000000ll) >> 56);
}
struct FGControl
{
    double elevator;
    double aileron;
} fgcontrol;


int main (int argc, char **argv) {
  int s, rc, i;
  double leftLine=0.5, rightLine=0.5;
  int factorR=1,factorL=-1;
  double elevatorMin=-1,elevatorMax=1,aileronMax=1;
  struct sockaddr_in cliAddr, remoteServAddr;
  struct hostent *h;
  /* Kommandozeile auswerten */
  wiringPiSetup () ;
  pinMode (8, INPUT) ;
  pinMode (9, INPUT) ;
  pinMode (7, INPUT) ;
  pinMode (15, INPUT) ;
  pinMode (0, INPUT) ;
  pinMode (2, INPUT) ;
  pinMode (3, INPUT) ;
  pullUpDnControl (8,PUD_UP);
  pullUpDnControl (9,PUD_UP);
  pullUpDnControl (7,PUD_UP);
  pullUpDnControl (15,PUD_UP);
  pullUpDnControl (0,PUD_UP);
  pullUpDnControl (2,PUD_UP);
  pullUpDnControl (3,PUD_UP);
  if (argc < 2) {
    printf ("Usage: %s <server> <data1> ... <dataN> \n",
       argv[0] );
    exit (EXIT_FAILURE);
  }
  FGControl tmpData ;
  /* IP-Adresse vom Server überprüfen */
  h = gethostbyname (argv[1]);
  if (h == NULL) {
    printf ("%s: unbekannter Host '%s' \n", 
       argv[0], argv[1] );
    exit (EXIT_FAILURE);
  }
  printf ("%s: sende Daten an '%s' (IP : %s) \n",
     argv[0], h->h_name,
     inet_ntoa (*(struct in_addr *) h->h_addr_list[0]) );
  remoteServAddr.sin_family = h->h_addrtype;
  memcpy ( (char *) &remoteServAddr.sin_addr.s_addr,
           h->h_addr_list[0], h->h_length);
  remoteServAddr.sin_port = htons (SERVER_PORT);
  /* Socket erzeugen */
  s = socket (AF_INET, SOCK_DGRAM, 0);
  if (s < 0) {
     printf ("%s: Kann Socket nicht öffnen (%s) \n",
        argv[0], strerror(errno));
     exit (EXIT_FAILURE);
  }
  /* Jeden Port bind(en) */
  cliAddr.sin_family = AF_INET;
  cliAddr.sin_addr.s_addr = htonl (INADDR_ANY);
  cliAddr.sin_port = htons (0);
  rc = bind ( s, (struct sockaddr *) &cliAddr,
              sizeof (cliAddr) );
  if (rc < 0) {
     printf ("%s: Konnte Port nicht bind(en) (%s)\n",
        argv[0], strerror(errno));
     exit (EXIT_FAILURE);
  }
  double oldTime=0;
  struct timeval tv;
  bool middleA;
  bool middleB;
  bool middleI;
  bool leftA;
  bool leftB;
  bool rightA;
  bool rightB;
  bool middleAo=false;
  bool middleBo=false;
  bool middleIo=false;
  bool leftAo=false;
  bool leftBo=false;
  bool rightAo=false;
  bool rightBo=false;
  long counterLeft=0;
  long counterRight=0;
  long counterMiddle=0;
  /* Daten senden */
  while (1) {
    gettimeofday(&tv,NULL);
    double currentTime = tv.tv_sec + ((float)tv.tv_usec) / 1000000.0;

    rightA = digitalRead(8);
    rightB= digitalRead(9);
    leftA = digitalRead(7);
    leftB = digitalRead(15);
    middleA  = digitalRead(2);
    middleB  = digitalRead(3);
    middleI  = digitalRead(0);
    if(middleI !=middleIo)
    {
       if(middleI)
       {
           fprintf(stderr,"Index\n");
           counterMiddle = 5;
       }
       middleIo = middleI;
    }
    if(middleA !=middleAo)
    {
       if(middleA)
       {
          if(middleB)
              counterMiddle--;
          else
              counterMiddle++;
       }
       middleAo = middleA;
    }
    if(leftA !=leftAo)
    {
/*
       if(leftA)
       {
          if(leftB)
              counterLeft--;
          else
              counterLeft++;
       }
*/

       if (leftA != leftB)
            counterLeft--;
       else 
            counterLeft++;
       leftAo = leftA;
    }
    if(rightA !=rightAo)
    {
       if(rightA != rightB)
              counterRight--;
          else
              counterRight++;
       rightAo = rightA;
    }
    
    if(currentTime > oldTime + 0.1)
    {
       leftLine = (counterLeft/1240.0); 
       rightLine =( counterRight/1240.0); 
	       float middleValue = -(counterMiddle/200.0); 
	oldTime = currentTime;
    if (leftLine>1.0)
       leftLine=1.0;
    if (leftLine<0.0)
       leftLine=0.0;
    if (rightLine>1.0)
       rightLine=1.0;
    if (rightLine<0.0)
       rightLine=0.0;
  fgcontrol.elevator=-((leftLine+rightLine)/2*(elevatorMax-elevatorMin)+elevatorMin);

  fgcontrol.aileron=(-leftLine+rightLine+middleValue)*aileronMax;
    tmpData = fgcontrol;
    byteSwap(tmpData.elevator);
    byteSwap(tmpData.aileron);
    rc = sendto (s, &tmpData, sizeof(fgcontrol), 0,
                 (struct sockaddr *) &remoteServAddr,
                 sizeof (remoteServAddr));
    printf ("Data Send cL %d cR %d ll %f rl %f m %d \n",counterLeft,counterRight,leftLine,rightLine,counterMiddle);
    if (rc < 0) {
       printf ("%s: Konnte Daten nicht senden %d\n",
          argv[0], i-1 );
       close (s);
       exit (EXIT_FAILURE);
    }
    }
  }
  return EXIT_SUCCESS;
}
