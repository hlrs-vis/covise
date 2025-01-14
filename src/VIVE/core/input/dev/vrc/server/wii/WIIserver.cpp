/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// PROGRAM WIIserver
//
// handling server for WII joysticks sending UDP packets for events
//
// Initial version: 2007-02-22 [cs]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2007 by VISENSO GmbH
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:  1.0 Initial version
//
//
#include <covise/covise.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#ifndef WIN32
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/prctl.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#else
#endif

#include <wiimote.h>

#include <util/UDP_Sender.h>
#include <util/ArgsParser.h>

struct acc
{
    uint8_t x;
    uint8_t y;
    uint8_t z;
};
void mesgCallback(int id, int mesg_count, union wiimote_mesg *mesg[]);
class joystate
{
public:
    joystate(const char *mode);
    void init();
    void show();
    void turnX(float alpha);
    void turnY(float alpha);
    void turnZ(float alpha);
    bool turning;
    int Button;
    int X;
    int Y;
    int Z;
    enum modeType
    {
        button,
        joystick
    };
    int mode;
    float alphaX;
    float alphaY;
    float alphaZ;
    float matrix[3][3];
    struct acc acc_zero, acc_one;
};

void joystate::show()
{

    printf("\n");
    printf("joy: acc_zero= %d %d %d, acc_one=%d %d %d\n", acc_zero.x, acc_zero.y, acc_zero.z, acc_one.x, acc_one.y, acc_one.z);
    //    int i,j;
    //    for(i=0;i<3;i++) {
    //       for(j=0;j<3;j++) {
    //          printf("%4f ", matrix[i][j]);
    //       }
    //       printf("\n");
    //    }
}

void initmatrix(float matrix[3][3])
{
    int i, j;
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            if (i == j)
                matrix[i][j] = 1.0;
            else
                matrix[i][j] = 0.0;
}

joystate::joystate(const char *mode)
{
    if (0 == strcmp(mode, "joystick"))
    {
        this->mode = joystick;
    }
    else
    {
        this->mode = button;
    }
    this->init();
}

void joystate::init()
{
    X = Y = Z = 0;
    Button = 0;
    alphaX = alphaY = alphaZ = 0.0;
    turning = false;
    initmatrix(matrix);
}

const float pi = 3.14159265;

void matmult(float a[3][3], float b[3][3], float c[3][3])
{
    int i, j, k;
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
        {
            c[i][j] = 0.0;
            for (k = 0; k < 3; k++)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
}

void matmult2(float a[3][3], float b[3][3])
{
    float c[3][3] = { { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } };
    int i, j, k;
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            for (k = 0; k < 3; k++)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    //Now copy the result into a
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            a[i][j] = c[i][j];
}

void turnX(float a[3][3], float alpha)
{
    float cosalpha = cos(alpha);
    float sinalpha = sin(alpha);
    float c[3][3] = {
        { cosalpha, 0.0, -sinalpha },
        { 0.0, 1.0, 0.0 },
        { sinalpha, 0.0, cosalpha }
        //      cosalpha,  0.0,   sinalpha,
        //      0.0,       1.0,   0.0,
        //      -sinalpha, 0.0,   cosalpha
    };
    matmult2(a, c);
}

void turnY(float a[3][3], float alpha)
{
    float cosalpha = cos(alpha);
    float sinalpha = sin(alpha);
    float c[3][3] = {
        { 1.0, 0.0, 0.0 },
        { 0.0, cosalpha, sinalpha },
        { 0.0, -sinalpha, cosalpha }
        //       1.0, 0.0,         0.0,
        //       0.0, cosalpha, -sinalpha,
        //       0.0, sinalpha, cosalpha
    };
    matmult2(a, c);
}

void turnZ(float a[3][3], float alpha)
{
    float cosalpha = cos(alpha);
    float sinalpha = sin(alpha);
    float c[3][3] = {
        { cosalpha, -sinalpha, 0.0 },
        { sinalpha, cosalpha, 0.0 },
        { 0.0, 0.0, 1.0 }
    };
    matmult2(a, c);
}

void joystate::turnX(float alpha)
{
    ::turnX(matrix, alpha);
}
void joystate::turnY(float alpha)
{
    ::turnY(matrix, alpha);
}
void joystate::turnZ(float alpha)
{
    ::turnZ(matrix, alpha);
}

joystate *joy = NULL;

// signal handler: correctly shut down flock
void sigHandler(int signo)
{
    fprintf(stderr, "Signal %d caught by Handler\n", signo);
    exit(0);
}

static const char *Ps2ServerVersion = "1.0";

/// ++++ Forward-declaration of functions defined later in the text ++++
// show help
void displayHelp(const char *progName);
void showbuffer(float x, float y, float z, float matrix[3][3], int button, UDP_Sender &sender, int stationID);
///////// program starts here

void evalProc(joystate *);

///Values measured by manual, one time calibration
float ax0 = 135;
float ay0 = 135;
float az0 = 135;
float denominator = 27;

float ax(int raw)
{
    return ((float)raw - ax0) / denominator;
}
float ay(int raw)
{
    return ((float)raw - ay0) / denominator;
}
float az(int raw)
{
    return ((float)raw - az0) / denominator;
}

int main(int argc, char *argv[])
{

    ArgsParser arg(argc, argv);

    //at least one stations has to be connected
    if (argc < 2
        || 0 == strcasecmp(argv[1], "-h")
        || 0 == strcasecmp(argv[1], "--help"))
    {
        displayHelp(argv[0]);
        exit(-1);
    }

    // ----- create default values for all options

    const char *target = arg.getOpt("-t", "--target", "localhost:7777");
    //We have two modes button and joystick
    const char *mode = arg.getOpt("-m", "--mode", "button");

    // if a bluetooth device address is given, then dont search for one
    const char *bdaddr_str = arg.getOpt("-a", "--address", NULL);

    if (arg.numArgs() < 1)
    {
        cerr << "\nCover station ID missing" << endl;
        exit(0);
    }

    //
    int stationID;

    sscanf(arg[0], "%d", &stationID);

    ////// Echo it
    fprintf(stderr, "\n");
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "  + VRC WIIserver %-10s (C) $LastChangedDate: 2007-02-15 11:59:23 +0100 (Do, 15 Feb 2007) $ VISENSO GmbH +\n", Ps2ServerVersion);
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "  + Settings:                                           +\n");
    fprintf(stderr, "  +   UDP Target:        %-30s +\n", target);
    fprintf(stderr, "  +   COVER station ID:  %-2d                             +\n", stationID);
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "\n");

    /// establish some signal handlers
    signal(SIGINT, sigHandler);
    signal(SIGPIPE, sigHandler);
    signal(SIGCHLD, sigHandler);
    signal(SIGTERM, sigHandler);
    signal(SIGHUP, sigHandler);
#ifndef __linux
    prctl(PR_TERMCHILD); // Exit when parent does
#else
    prctl(PR_SET_PDEATHSIG, SIGTERM);
#endif

    /// Open UDP sender
    UDP_Sender sender(target);
    if (sender.isBad())
    {
        cerr << "Could not start UDP server t" << endl;
        return -1;
    }

    joy = new joystate(mode);

    perror("");

    bdaddr_t bdaddr = *BDADDR_ANY;

    if (bdaddr_str == NULL)
    {
        printf("Looking for a Wiimote... (please press 1+2 on the Wiimote)\n");
        if (wiimote_find_wiimote(&bdaddr, -1))
        {
            exit(1);
        }
        char s[100];
        ba2str(&bdaddr, s);
        printf("Using Wiimote at address %s\n", s);
    }
    else
    {
        printf("Waiting for Wiimote %s... (please press 1+2 on the Wiimote)\n", bdaddr_str);
        str2ba(bdaddr_str, &bdaddr);
    }
    int addr;
    wiimote_t *wii = wiimote_connect(&bdaddr, mesgCallback, &addr);
    if (NULL == wii)
    {
        exit(1);
    }

    wiimote_command(wii, WIIMOTE_CMD_RPT_MODE, WIIMOTE_RPT_ACC | WIIMOTE_RPT_STATUS | WIIMOTE_RPT_BTN | WIIMOTE_RPT_IR);
    //   wiimote_read(wii,  WIIMOTE_RW_EEPROM, 0x16, 7,bu);

    int X, Y, Z, Button = 0;
    float alphaX = 0.0, alphaY = 0.0, alphaZ = 0.0;
    X = Y = Z = 0;
    float matrix[3][3] = { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } };
    Button = -1;
    while (true)
    {
        usleep(10000);
        Button = joy->Button;
        switch (Button)
        {
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
            //The angles are increased by the value delivered by the joystick
            //if the angle exceeds pi or -pi it is corrected
            alphaX = joy->alphaX;
            if (alphaX > pi)
                alphaX -= 2 * pi;
            if (alphaX < -pi)
                alphaX += 2 * pi;
            alphaY = joy->alphaY;
            if (alphaY > pi)
                alphaY -= 2 * pi;
            if (alphaY < -pi)
                alphaY += 2 * pi;
            alphaZ = joy->alphaZ;
            if (alphaZ > pi)
                alphaZ -= 2 * pi;
            if (alphaZ < -pi)
                alphaZ += 2 * pi;
            if (joy->alphaX != 0.0)
                turnX(matrix, alphaX);
            if (joy->alphaY != 0.0)
                turnY(matrix, alphaY);
            if (joy->alphaZ != 0.0)
                turnZ(matrix, alphaZ);
            //
            X += joy->X;
            Y += joy->Y;
            Z += joy->Z;
            break;
        case 8:
            break;
        case 9:
            //Reset
            //This is the HOME button
            //we use it for reset
            X = Y = Z = Button = 0;
            alphaX = alphaY = alphaZ = 0.0;
            initmatrix(matrix);
            joy->init();
            break;
        case 10:
            break;
        case 11:
            break;
        default:
            break;
        }
        showbuffer((float)X / 3.0, (float)Y / 3.0, (float)Z / 3.0, joy->matrix, Button, sender, stationID);
    }
    return 0;
}

float integralx = 0.0;
float lastintegralx = 0.0;

float integraly = 0.0;
float lastintegraly = 0.0;

float integralz = 0.0;
float lastintegralz = 0.0;
void mesgCallback(int id, int mesg_count, union wiimote_mesg *mesg[])
{
    int i, j;
    int valid_source;
    /////   joy->show();
    for (i = 0; i < mesg_count; i++)
    {
        switch (mesg[i]->type)
        {
        case WIIMOTE_MESG_STATUS:
            printf("Status Report: id=%d battery=%d extension=", id,
                   mesg[i]->status_mesg.battery);
            switch (mesg[i]->status_mesg.extension)
            {
            case WIIMOTE_EXT_NONE:
                printf("none");
                break;
            case WIIMOTE_EXT_NUNCHUK:
                printf("Nunchuk");
                break;
            case WIIMOTE_EXT_CLASSIC:
                printf("Classic Controller");
                break;
            default:
                printf("Unknown Extension");
                break;
            }
            printf("\n");
            break;
        case WIIMOTE_MESG_BTN:
            ////            printf("Button Report: %.4X\n", mesg[i]->btn_mesg.buttons);
            if (0 == mesg[i]->btn_mesg.buttons)
            {
                joy->Button = 0;
                joy->X = joy->Y = joy->Z = 0;
            }
            if (mesg[i]->btn_mesg.buttons & 0x0004)
            {
                joy->Button = 1;
            }
            if (mesg[i]->btn_mesg.buttons & 0x0008)
            {
                joy->Button = 2;
            }
            if (mesg[i]->btn_mesg.buttons & 0x0080)
            {
                joy->Button = 9;
            }

            if (mesg[i]->btn_mesg.buttons & 0x100)
            {
                if (joy->mode == joystate::joystick)
                    joy->X -= 1;
                else
                    joy->Button = 3;
            }
            if (mesg[i]->btn_mesg.buttons & 0x200)
            {
                if (joy->mode == joystate::joystick)
                    joy->X += 1;
                else
                    joy->Button = 3;
            }
            if (mesg[i]->btn_mesg.buttons & 0x400)
            {
                if (joy->mode == joystate::joystick)
                    joy->Y -= 1;
                else
                    joy->Button = 3;
            }
            if (mesg[i]->btn_mesg.buttons & 0x800)
            {
                if (joy->mode == joystate::joystick)
                    joy->Y += 1;
                else
                    joy->Button = 3;
            }

            if (mesg[i]->btn_mesg.buttons & 0x0010)
            {
                joy->Z -= 1;
            }
            if (mesg[i]->btn_mesg.buttons & 0x1000)
            {
                joy->Z += 1;
            }
            break;
        case WIIMOTE_MESG_ACC:
            float pitch, roll;
            double a_x, a_y, a_z;
            a_x = ((double)mesg[i]->acc_mesg.x - joy->acc_one.x);
            a_y = ((double)mesg[i]->acc_mesg.y - joy->acc_one.y);
            a_z = ((double)mesg[i]->acc_mesg.z - joy->acc_one.z);
            integralx += ax(mesg[i]->acc_mesg.x);
            integraly += ay(mesg[i]->acc_mesg.y);
            integralz += az(mesg[i]->acc_mesg.z);
            joy->turning = true;
            initmatrix(joy->matrix);
            pitch = asin(ax(mesg[i]->acc_mesg.x));
            roll = asin(ay(mesg[i]->acc_mesg.y));
            joy->turnZ(pitch);
            joy->turnY(-roll);
            break;
        case WIIMOTE_MESG_IR:
            ////            printf("IR Report: ");
            valid_source = 0;
            for (j = 0; j < WIIMOTE_IR_SRC_COUNT; j++)
            {
                if (mesg[i]->ir_mesg.src[j].valid)
                {
                    valid_source = 1;
                    printf("%d:(%d,%d)\n ", j, mesg[i]->ir_mesg.src[j].x,
                           mesg[i]->ir_mesg.src[j].y);
                }
            }
            if (!valid_source)
            {
                ////               printf("no sources detected");
            }
            ////            printf("\n");
            break;
        case WIIMOTE_MESG_NUNCHUK:
            printf("Nunchuk Report: btns=%.2X stick=(%d,%d) acc.x=%d acc.y=%d "
                   "acc.z=%d\n",
                   mesg[i]->nunchuk_mesg.buttons,
                   mesg[i]->nunchuk_mesg.stick_x,
                   mesg[i]->nunchuk_mesg.stick_y, mesg[i]->nunchuk_mesg.acc_x,
                   mesg[i]->nunchuk_mesg.acc_y, mesg[i]->nunchuk_mesg.acc_z);
            break;
        case WIIMOTE_MESG_CLASSIC:
            printf("Classic Report: btns=%.4X l_stick=(%d,%d) r_stick=(%d,%d) "
                   "l=%d r=%d\n",
                   mesg[i]->classic_mesg.buttons,
                   mesg[i]->classic_mesg.l_stick_x,
                   mesg[i]->classic_mesg.l_stick_y,
                   mesg[i]->classic_mesg.r_stick_x,
                   mesg[i]->classic_mesg.r_stick_y,
                   mesg[i]->classic_mesg.l, mesg[i]->classic_mesg.r);
            break;

        default:
            printf("Unknown Report");
            break;
        }
    }
}

void evalProc(joystate *joy)
{
    ////   wiimote_mesg_callback_t *mesg_callback;

    int i;
    //the  value of this variable
    //is toggled by pressing button 8

    while (true)
    {
    }
    unsigned char bytes[32];
    while (true)
    {
        for (i = 0; i < 8; i++)
            bytes[i] = 0;
        if (bytes[6] == 1)
        {
            //Button event
            if (bytes[4] == 0)
            {
                //A button was released
                joy->Button = 0;
            }
            else
            {
                //A button was pressed
                joy->Button = bytes[7];
                if (joy->Button == 8)
                {
                    joy->turning = !joy->turning;
                }
                if (joy->Button == 10)
                    joy->Button = 1;
                if (joy->Button == 11)
                    joy->Button = 2;
                if (joy->Button == 0)
                    joy->Button = 3;
            }
        }
        else if (bytes[6] == 2)
        {
            short *sp;
            sp = (short *)(bytes + 4);

            if (joy->turning)
            {
                //we don't move but turn
                float alpha;
                alpha = pi * ((float)*sp) / 32768.0 / 30.0;
                switch (bytes[7])
                {
                case 0:
                    joy->alphaX = alpha;
                    break;
                case 1:
                    joy->alphaY = alpha;
                    break;
                case 2:
                    joy->alphaZ = alpha;
                }
            }
            else
            {
                switch (bytes[7])
                {
                case 0:
                    joy->X = (int)*sp;
                    break;
                case 1:
                    joy->Y = (int)*sp;
                    break;
                case 2:
                    joy->Z = (int)*sp;
                }
            }
        }
        ///      printf("\nevalProcButton=%2d  X=%5d, Y=%5d, Z=%5d\n", joy->Button, joy->X,joy->Y,joy->Z);
    }
}

// Send the data to the port and display it on the screen
void showbuffer(float x, float y, float z, float matrix[3][3], int button, UDP_Sender &sender, int stationID)
{
    //   int button1;
    char sendbuffer[2048];
    static int often = 0;
    often++;
    sprintf(sendbuffer, "VRC %d %3d [%f %f %f] - [%f %f %f %f %f %f %f %f %f] - [0 0]",
            stationID, button, x, y, z,
            matrix[0][0], matrix[0][1], matrix[0][2],
            matrix[1][0], matrix[1][1], matrix[1][2],
            matrix[2][0], matrix[2][1], matrix[2][2]);
    //   if(often%100==0)
    //      fprintf(stderr,"%s\n",sendbuffer);
    sender.send(sendbuffer, strlen(sendbuffer) + 1);
}
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Help

// show help
void displayHelp(const char *progName)
{
    cout << progName << " [options]  coverID\n"
         << "\n"
         << "   coverID = Station ID for COVER's BUTTON_ADDR config\n"
         << "\n"
         << "Options:\n"
         << "\n"
         << "   -t <host:port>          set target to send tracking UDP packets\n"
         << "   --target=host:port      (default: localhost:7777)\n"
         << "\n"
         << "   -m <mode>         define the mode the WII should work in\n"
         << "   --mode=mode      (default: button)\n"
         << "\n"
         << "   -a <address>         set the bluetooth address of the Wiimote to use\n"
         << "   --address=address       (default: none, search for a Wiimote)\n"
         << "\n"
         << "Examples:\n"
         << "\n"
         << "   " << progName << " 3             Use the WII in button mode and send\n"
         << "                           data to localhost:7777 with ID=3\n"
         << "\n"
         << "   " << progName << " -m joystick  -t visenso:6666 4\n"
         << "                           use th wii in joystick mode and send data\n"
         << "                           to Host \"visenso\" Port 6666 with ID=4\n"
         << endl;
}
