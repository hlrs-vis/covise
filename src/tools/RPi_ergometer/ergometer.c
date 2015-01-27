/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <wiringPi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <signal.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <string.h>

// ------------------------------------------------------------

#define GPIO_SPEED 0
#define GPIO_BRAKE 2
#define GPIO_SYNC 7

// ------------------------------------------------------------

#define SRV_IP "192.168.1.2"
#define BUFLEN 512
#define NPACK 10
#define PORT 9930
#define BUF 512
#define LOCAL_SERVER_PORT 9931

// ------------------------------------------------------------

static volatile bool pin0_value;
static volatile int accel;
static volatile int speed_counter;
static volatile int speed;

// network udp listener

int slist, rc, n, len;
struct sockaddr_in cliAddr, servAddr;
char puffer[BUF];
time_t time1;
char loctime[BUF];
char *ptr;
const int y = 1;

// network

struct sockaddr_in si_other;
int s, i, slen = sizeof(si_other);
char buf[BUFLEN];

void sendUDPData(void);

// ------------------------------------------------------------
// diep
// helper
// ------------------------------------------------------------

void diep(char *s)
{
    perror(s);
    exit(1);
}

// ------------------------------------------------------------
// signal SIGINT
// we have to make sure that the brake signal is low, if the
// programm is being terminated by an ctrl-c
// ------------------------------------------------------------
void signalfunc(int sig)
{
    if (sig != SIGINT)
    {
        return;
    }
    else
    {
        digitalWrite(GPIO_BRAKE, LOW);
        exit(0);
    }
}

// ------------------------------------------------------------
// timing thread
// gets the no. of edges in the speed signal (rpm) every second
// second
// ------------------------------------------------------------
PI_THREAD(thrdTiming)
{
    while (1)
    {
        speed = speed_counter * 10;
        printf("speed =   %d\n", speed);
        printf("accel =            %d\n", accel);
        speed_counter = 0;

        sendUDPData();

        delayMicroseconds(100000);
    }
}

// ------------------------------------------------------------
// ------------------------------------------------------------
PI_THREAD(thrdBreak)
{
    int oldaccel;

    while (1)
    {
        memset(puffer, 0, BUF);
        len = sizeof(cliAddr);

        n = recvfrom(slist, puffer, BUF, 0, (struct sockaddr *)&cliAddr, (socklen_t *)&len);

        if (n >= 0)
        {
            //printf ("debug: got something\n");

            if (speed <= 0)
            {
                // if not driving wh should not set the break
                accel = 0;
            }
            else
            {
                oldaccel = accel;

                sscanf(puffer, "%d", &accel);

                // we should get something in {0,10000}
                // 0 - no break, 10000 - full break
                accel = (0 - accel) + 5000;

                // Mittel nehmen
                accel = ((oldaccel * 15) + accel) / 16;

                //accel -= 1000;

                if (accel > 5000)
                    accel = 5000;
                if (accel < -5000)
                    accel = -5000;
            }
        }

        /* Zeitangaben präparieren */
        //      time(&time1);
        //      strncpy(loctime, ctime(&time1), BUF);
        //      ptr = strchr(loctime, '\n' );
        //      *ptr = '\0';
        /* Erhaltene Nachricht ausgeben */
        //      printf ("%s: Daten erhalten von %s:UDP%u : %s \n",
        //              loctime, inet_ntoa (cliAddr.sin_addr),
        //              ntohs (cliAddr.sin_port), puffer);
    }
}

// ----------------------------------------------------------
// isr GPIO_SPEED
// raised on the falling edge of the speed signal
// ---------------------------------------------------------
void isrRPMEdgeDetected(void)
{
    speed_counter++;
}

// ---------------------------------------------------------
// isr GPIO_SYNC
// raised on the rising or falling edge of the sync signal
// ---------------------------------------------------------
void isrSyncEdgeDetected(void)
{
    // read current sync level
    pin0_value = digitalRead(GPIO_SYNC);

    if (pin0_value)
    {
        // rising edge => positive half-wave
        // acceleration is set if positive
        if (accel >= 0)
        {
            digitalWrite(GPIO_BRAKE, HIGH);
            delayMicroseconds(accel);
            digitalWrite(GPIO_BRAKE, LOW);
        }
    }
    else
    {
        // falling edge => negative half-wave
        // acceleration is set if negative
        if (accel <= 0)
        {
            digitalWrite(GPIO_BRAKE, HIGH);
            delayMicroseconds(abs(accel));
            digitalWrite(GPIO_BRAKE, LOW);
        }
    }
}

// ---------------------------------------------------------
// sendData()
// send an udp diagram to a server application
// ---------------------------------------------------------
void sendUDPData(void)
{
    sprintf(buf, "%i\n", speed);

    if (sendto(s, buf, BUFLEN, 0, (struct sockaddr *)&si_other, slen) == -1)
    {
        diep("sendto()");
    }
}

// ---------------------------------------------------------
// main
// ---------------------------------------------------------
int main(void)
{
    // ergometer

    accel = 0;
    speed_counter = 0;
    speed = 0;

    // pi setup

    if (wiringPiSetup() == -1)
    {
        exit(1);
    }

    signal(SIGINT, signalfunc);

    // state machine

    int state = 0;

    while (1)
    {
        switch (state)
        {
        case 0:
        {
            printf("gpio setup ...\n");

            //generate PWM on GPIO 1 for testing purposes
            //pinMode (1, PWM_OUTPUT);
            //pwmSetMode(PWM_MODE_MS);
            //pwmSetClock(400);
            //pwmSetRange(1000);
            //pwmWrite(1, 400);

            pinMode(GPIO_BRAKE, OUTPUT);
            pinMode(GPIO_SPEED, INPUT);
            pinMode(GPIO_SYNC, INPUT);

            digitalWrite(GPIO_BRAKE, LOW);

            wiringPiISR(GPIO_SYNC, INT_EDGE_BOTH, &isrSyncEdgeDetected);
            wiringPiISR(GPIO_SPEED, INT_EDGE_FALLING, &isrRPMEdgeDetected);

            piThreadCreate(thrdTiming);
            piThreadCreate(thrdBreak);

            state = 1;
            break;
        }
        case 1:
        {
            printf("network setup ... (tx)\n");

            if ((s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1)
            {
                diep("socket");
            }

            memset((char *)&si_other, 0, sizeof(si_other));
            si_other.sin_family = AF_INET;
            si_other.sin_port = htons(PORT);
            if (inet_aton(SRV_IP, &si_other.sin_addr) == 0)
            {
                fprintf(stderr, "inet_aton() failed\n");
                exit(1);
            }

            printf("network setup ... (rx)\n");

            slist = socket(AF_INET, SOCK_DGRAM, 0);
            if (slist < 0)
            {
                printf("debug: Kann Socket nicht öffnen ...\n");
                exit(EXIT_FAILURE);
            }
            /* Lokalen Server Port bind(en) */
            servAddr.sin_family = AF_INET;
            servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
            servAddr.sin_port = htons(LOCAL_SERVER_PORT);
            setsockopt(slist, SOL_SOCKET, SO_REUSEADDR, &y, sizeof(int));
            rc = bind(slist, (struct sockaddr *)&servAddr,
                      sizeof(servAddr));
            if (rc < 0)
            {
                printf("debug: Kann Portnummern nicht binden\n");
                exit(EXIT_FAILURE);
            }
            printf("debug: Wartet auf Daten am Port (UDP) %u\n",
                   LOCAL_SERVER_PORT);

            state = 5;
            break;
        }
        case 2:
        {
            // increase accel
            accel += 1;
            delay(1);

            if (accel >= 5000)
            {
                state = 3;
            }
            break;
        }
        case 3:
        {
            // decrease accel
            accel -= 1;
            delay(1);

            if (accel <= -5000)
            {
                state = 2;
            }
            break;
        }
        case 4:
        {
            // map speed:     0 ... 1000
            // to  accel  -5000 ... 5000
            accel = 2000; // ((speed * 10 * 1.5) - 5000);

            if (accel > 5000)
            {
                accel = 5000;
            }
            delay(1);

            break;
        }
        case 5:
        {
            printf("sm doing nothing ...\n");

            delayMicroseconds(1000000);

            break;
        }
        }
    }

    return 0;
}

// ---------------------------------------------------------

/*     
     
   //create a UDP socket
   if ((s=socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1)
   {
      die("socket");
   }
     
   // zero out the structure
   memset((char *) &si_me, 0, sizeof(si_me));
     
   si_me.sin_family = AF_INET;
   si_me.sin_port = htons(PORT);
   si_me.sin_addr.s_addr = htonl(INADDR_ANY);
     
   //bind socket to port
   if( bind(s , (struct sockaddr*)&si_me, sizeof(si_me) ) == -1)
   {
      die("bind");
   }
     
   //keep listening for data
   while(1)
   {
      printf("Waiting for data...");
      fflush(stdout);
         
      //try to receive some data, this is a blocking call
      if ((recv_len = recvfrom(s, buf, BUFLEN, 0, (struct sockaddr *) &si_other, &slen)) == -1)
      {
         die("recvfrom()");
      }
         
      //print details of the client/peer and the data received
      printf("Received packet from %s:%d\n", inet_ntoa(si_other.sin_addr), ntohs(si_other.sin_port));
      printf("Data: %s\n" , buf);
         
      //now reply the client with the same data
      if (sendto(s, buf, recv_len, 0, (struct sockaddr*) &si_other, slen) == -1)
      {
         die("sendto()");
      }
   }
 
*/
