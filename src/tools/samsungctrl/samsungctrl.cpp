/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <iostream>
#include <vector>
#include <net/covise_host.h>
#include <net/covise_connect.h>
#include <net/covise_socket.h>
#include <string.h>

#ifndef _WIN32
#include <unistd.h>
#include <stdlib.h>
#else
#define strcasecmp stricmp
void usleep(unsigned int usec)
{
    Sleep(usec / 1000);
}
#endif
using namespace std;
using namespace covise;

enum
{
    STATUS = 0x00,
    TIME = 0x01,
    ON_TIME = 0x02,
    OFF_TIME = 0x03,
    VIDEO = 0x04,
    AUDIO = 0x05,
    RGB = 0x06,
    PIP_STATUS = 0x07,
    MAINTENANCE = 0x08,
    SERIAL_NUMBER = 0x0B,
    DISPLAY_STATUS = 0x0D,
    SW_VERSION = 0x0E,
    AUTO_MOTION_PLUS = 0x0F,
    MODEL_NUMBER = 0x10,
    POWER = 0x11,
    VOLUME = 0x12,
    MUTE = 0x13,
    INPUT_SOURCE = 0x14,
    PICTURE_SIZE = 0x15,
    DIRECT_CHANNEL = 0x17,
    SCREEN_MODE = 0x18,
    SCREEN_SIZE = 0x19,
    RED_OFFSET = 0x1A,
    GREEN_OFFSET = 0x1B,
    BLUE_OFFSET = 0x1C,
    MDC_CONNECTION_TYPE = 0x1D,
    IMAGE_RETENTION = 0x1E,
    CONTRAST = 0x24,
    BRIGHTNESS = 0x25,
    SHARPNESS = 0x26,
    COLOR = 0x27,
    TINT = 0x28,
    RED_GAIN = 0x29,
    GREEN_GAIN = 0x2A,
    BLUE_GAIN = 0x2B,
    TREBLE = 0x2C,
    BASS = 0x2D,
    BALANCE = 0x2E,
    COARSE = 0x2F,
    FINE = 0x30,
    H_POSITION = 0x31,
    V_POSITION = 0x32,
    CLEAR_MENU = 0x34,
    REMOTE = 0x36,
    RGB_CONTRAST = 0x37,
    RGB_BRIGHTNESS = 0x38,
    PIP_ON_OFF = 0x3C,
    AUTO_ADJUST = 0x3D,
    COLOR_TONE = 0x3E,
    COLOR_TEMPERATURE = 0x3F,
    PIP_SOURCE = 0x40,
    MAIN_PIP_SWAP = 0x41,
    PIP_SIZE = 0x42,
    PIP_LOCATION = 0x43,
    SOUND_SELECT = 0x47,
    PIXEL_SHIFT = 0x4C,
    VIDEO_WALL = 0x4F,
    AUTO_LAMP = 0x57,
    MANUAL_LAMP = 0x58,
    SAFETY_SCREEN_RUN = 0x59,
    SAFETY_SCREEN = 0x5B,
    VIDEO_WALL_MODE = 0x5C,
    SAFETY_LOCK = 0x5D,
    PANEL_LOCK = 0x5F,
    OSD_ON_OFF = 0x70,
    P_MODE = 0x71,
    S_MODE = 0x72,
    NR_MODE_SET = 0x73,
    PC_COLOR_TONE = 0x75,
    AUTO_ADJUSTMENT = 0x76,
    ALL_KEYS_LOCK = 0x77,
    SRS_TSXT = 0x78,
    FILM_MODE = 0x79,
    SIGNAL_BALANCE = 0x7A,
    SB_GAIN_R = 0x7B,
    SB_GAIN_G = 0x7C,
    SB_GAIN_B = 0x7D,
    SB_GAIN = 0x7E,
    SB_SHARPNESS = 0x7F,
    PANEL_ON_TIME = 0x83,
    VIDEO_WALL_ON = 0x84,
    TEMPERATURE = 0x85,
    BRIGHTNESS_SENSOR = 0x86,
    DYNAMIC_CONTRAST = 0x87,
    SAFETY_SCREEN_ON = 0x88,
    VIDEO_WALL_USER = 0x89,
    MODEL_NAME = 0x8A,
    HDMI_BLACK_LEVEL = 0x94,
    RJ45_SETTING = 0xA2,
    OSD_DISPLAY_TYPE = 0xA3,
    TIMER1 = 0xA4,
    TIMER2 = 0xA5,
    TIMER3 = 0xA6,
    CLOCK = 0xA7,
    HOLIDAY_ADD_DELETE = 0xA8,
    HOLIDAY_GET = 0xA9,
    MAGIC_NET_PIP = 0xE0,
    PANEL_ON_OFF = 0xF9,
    AMBIENT_BRIGHTNESS_MODE = 0xA1,
    FAN = 0x8F,
    GAMMA = 0x96,
    RESET = 0x9F,
    USER_AUTO = 0x45,
    STANDBY = 0x4A,
    FAN_SPEED = 0x44,
    VIDEO_PICTURE_POSITION = 0x4B,
    AUTO_POWER = 0x33,
};

class samsungDisplay
{
public:
    Host *display;
    SimpleClientConnection *conn;
    samsungDisplay(const char *hostname);
    ~samsungDisplay();
};
samsungDisplay::samsungDisplay(const char *hostname)
{
    display = new Host(hostname);
    conn = new SimpleClientConnection(display, 1515);
    if (!conn->is_connected())
    {
        std::cerr << "could not connect to: " << hostname << std::endl;
        _exit(-1);
    }
    std::cerr << "connected to: " << hostname << std::endl;
}
samsungDisplay::~samsungDisplay()
{
    delete conn;
    delete display;
}
class Displays
{
public:
    Displays();
    ~Displays();
    int sendCommand(int c, const unsigned char *data, int len);
    int sendVideoWallCommand(int c, const unsigned char *data, int len);
    int readReply();
    std::vector<samsungDisplay *> disp;
};
Displays::Displays()
{
}
Displays::~Displays()
{
    for (int i = 0; i < disp.size(); i++)
        delete disp[i];
}
int Displays::readReply()
{
    int numResponding = 0;
    for (int i = 0; i < disp.size(); i++)
    {
        unsigned char buf[100];
        int numRead = 0;
        //do {
        numRead = disp[i]->conn->getSocket()->Read(buf, 100);
        /* std::cerr << "Read" <<  numRead  << " of " << 100 << std::endl;
	    fprintf(stderr,"buffer:");
	for(int n=0;n<numRead;n++)
	{
	    fprintf(stderr,"%x",buf[n]);
	}
	    fprintf(stderr,"\n");
	//} while(numRead > 0 || (numRead < 0 && (errno == EAGAIN || errno == EINTR || errno==0)));*/
        if (numRead < 0)
        {
            perror("socket read:");
        }
        else
        {
            ++numResponding;

            if ((buf[0] == 0xAA) && (buf[1] == 0xFF))
            {
                int ID = buf[2];
                int len = buf[3];
                int Ack_Nak = buf[4];
                int cmd = buf[5];
                if (Ack_Nak == 0x00)
                {
                    fprintf(stderr, "command failed, error code %x\n", buf[6]);
                }
                else if (cmd == 0x00 && len == 9) // status
                {
                    fprintf(stderr, "Power %x\n", buf[6]);
                    fprintf(stderr, "Volume%x\n", buf[7]);
                    fprintf(stderr, "Mute %x\n", buf[8]);
                    fprintf(stderr, "Input %x\n", buf[9]);
                    fprintf(stderr, "Aspect %x\n", buf[10]);
                }
                else if (cmd == 0x04 && len == 0x0A) // video control
                {
                    fprintf(stderr, "Contrast %x\n", buf[6]);
                    fprintf(stderr, "Brightness %x\n", buf[7]);
                    fprintf(stderr, "Sharpness %x\n", buf[8]);
                    fprintf(stderr, "Color %x\n", buf[9]);
                    fprintf(stderr, "Tint %x\n", buf[10]);
                    fprintf(stderr, "ColorTone %x\n", buf[11]);
                    fprintf(stderr, "ColorTemp %x\n", buf[12]);
                }
                else if (cmd == 0x0D && len == 0x08) // Display Status
                {
                    fprintf(stderr, "Lamp %x\n", buf[6]);
                    fprintf(stderr, "Temperature %x\n", buf[7]);
                    fprintf(stderr, "Bright_sensor %x\n", buf[8]);
                    fprintf(stderr, "No_Sync %x\n", buf[9]);
                    fprintf(stderr, "Cur_Temp %x\n", buf[10]);
                    fprintf(stderr, "Fan %x\n", buf[12]);
                }
                else if (cmd == 0x08 && len == 0x15) // video control
                {
                    fprintf(stderr, "Power%x\n", buf[6]);
                    fprintf(stderr, "P.Size %x\n", buf[7]);
                    fprintf(stderr, "P.Source %x\n", buf[8]);
                    fprintf(stderr, "LMax_H %x\n", buf[9]);
                    fprintf(stderr, "LMax_M %x\n", buf[10]);
                    fprintf(stderr, "LMax_AP %x\n", buf[11]);
                    fprintf(stderr, "LMaxValue %x\n", buf[12]);
                    fprintf(stderr, "LMin_H %x\n", buf[13]);
                    fprintf(stderr, "LMin_M %x\n", buf[14]);
                    fprintf(stderr, "LMin_AP %x\n", buf[15]);
                    fprintf(stderr, "LMinValue %x\n", buf[16]);
                    fprintf(stderr, "LampValue %x\n", buf[17]);
                    fprintf(stderr, "ScreenInterval %x\n", buf[18]);
                    fprintf(stderr, "ScreenTime %x\n", buf[19]);
                    fprintf(stderr, "ScreenType %x\n", buf[20]);
                    fprintf(stderr, "V.Wall %x\n", buf[21]);
                    fprintf(stderr, "V.WallFormat %x\n", buf[22]);
                    fprintf(stderr, "V.WallDivid %x\n", buf[23]);
                    fprintf(stderr, "V.WallSet %x\n", buf[24]);
                }
            }
        }
    }
    return numResponding;
}
int Displays::sendCommand(int c, const unsigned char *data, int len)
{
    unsigned char *buf;
    buf = new unsigned char[len + 100];
    size_t pos = 0;
    buf[pos++] = 0xAA;
    buf[pos++] = c;
    buf[pos++] = 0x00;
    buf[pos++] = len;
    for (int i = 0; i < len; i++)
    {
        buf[pos] = data[i];
        pos++;
    }
    buf[pos] = 0x0; // checksum
    for (int i = 1; i < pos; i++)
    {
        buf[pos] += buf[i];
    }
    pos++;
    for (int i = 0; i < disp.size(); i++)
    {
        int numWritten = disp[i]->conn->getSocket()->write(buf, (unsigned int)pos);
        if (numWritten < pos)
        {
            std::cerr << "could not send all bytes, only" << numWritten << " of " << pos << std::endl;
            return -1;
        }
        else
            std::cerr << "sent" << numWritten << " of " << pos << std::endl;
    }
    delete[] buf;
    return 0;
}

int Displays::sendVideoWallCommand(int c, const unsigned char *data, int len)
{
    unsigned char *buf;
    buf = new unsigned char[len + 100];
    size_t pos = 0;
    buf[pos++] = 0xAA;
    buf[pos++] = c;
    buf[pos++] = 0x00;
    buf[pos++] = len;
    for (int i = 0; i < len; i++)
    {
        buf[pos] = data[i];
        pos++;
    }
    buf[pos] = 0x0; // checksum
    for (int i = 1; i < pos; i++)
    {
        buf[pos] += buf[i];
    }
    for (int i = 0; i < disp.size(); i++)
    {
        buf[5] = (unsigned char)(disp.size() - i);
        buf[pos] = 0x0; // checksum
        for (int n = 1; n < pos; n++)
        {
            buf[pos] += buf[n];
        }
        fprintf(stderr, "%x\n", buf[5]);
        int numWritten = disp[i]->conn->getSocket()->write(buf, (unsigned int)(pos + 1));
        if (numWritten < pos)
        {
            std::cerr << "could not send all bytes, only" << numWritten << " of " << pos + 1 << std::endl;
            return -1;
        }
        else
            std::cerr << "sent" << numWritten << " of " << pos + 1 << std::endl;
    }
    delete[] buf;
    return 0;
}

int main(int argc, char **argv)
{
    // check arguments
    if (argc < 3)
    {
        cerr << "Wrong argument number" << endl;
        cerr << "Usage: samsungdctrl command value DISPLAY_IP[s]  " << endl;
        return 1;
    }

    if (strcasecmp(argv[1], "power") == 0)
    {
        if (strcasecmp(argv[2], "on") == 0)
        {
            std::vector<std::string> macs;
            if (argc < 4) {
                macs.push_back("48:44:f7:90:69:f2");
                macs.push_back("48:44:f7:8f:2d:e2");
                macs.push_back("48:44:f7:90:69:f3");
                macs.push_back("48:44:f7:90:69:f5");
                macs.push_back("48:44:f7:d9:39:d3");
                macs.push_back("48:44:f7:d9:39:cf");
                macs.push_back("48:44:f7:90:69:e9");
                macs.push_back("48:44:f7:90:69:eb");
                macs.push_back("48:44:f7:90:69:f4");
            }
            else
            {
                for (int i = 0; i < (argc - 3); i++)
                {
                    macs.push_back(argv[i+3]);
                }
            }

            for (std::vector<std::string>::iterator it = macs.begin(); it != macs.end(); ++it)
            {
                std::string cmd = std::string("sudo /usr/sbin/ether-wake ").append(*it);
                if (system(cmd.c_str()) == -1) {
                    return 1;
                }
            }
            return (0);
        }
    }
    Displays displays;
    if (argc < 4)
    {
        displays.disp.push_back(new samsungDisplay("141.58.8.61"));
        displays.disp.push_back(new samsungDisplay("141.58.8.62"));
        displays.disp.push_back(new samsungDisplay("141.58.8.63"));
        displays.disp.push_back(new samsungDisplay("141.58.8.64"));
        displays.disp.push_back(new samsungDisplay("141.58.8.65"));
        displays.disp.push_back(new samsungDisplay("141.58.8.66"));
        displays.disp.push_back(new samsungDisplay("141.58.8.67"));
        displays.disp.push_back(new samsungDisplay("141.58.8.68"));
        displays.disp.push_back(new samsungDisplay("141.58.8.69"));
    }
    else
    {
        for (int i = 0; i < (argc - 3); i++)
        {
            displays.disp.push_back(new samsungDisplay(argv[i + 3]));
        }
    }
    unsigned char val[100];
    usleep(100000);
    if (strcasecmp(argv[1], "power") == 0)
    {
        val[0] = 0;
        if (strcasecmp(argv[2], "on") == 0)
            val[0] = 1;
        displays.sendCommand(POWER, val, 1);
    }
    if (strcasecmp(argv[1], "VIDEO_WALL") == 0)
    {

        val[0] = 0;
        if (strcasecmp(argv[2], "on") == 0)
        {
            val[0] = 1;
            displays.sendCommand(VIDEO_WALL_ON, val, 1);
            displays.readReply();
            usleep(100000);
            val[0] = 0x33;
            val[1] = 0x4;
            displays.sendVideoWallCommand(VIDEO_WALL_USER, val, 2);

            displays.readReply();
            usleep(100000);
            val[0] = 0x18; //DVI
            displays.sendCommand(INPUT_SOURCE, val, 1);
        }
        else
        {
            displays.sendCommand(VIDEO_WALL_ON, val, 1);
            displays.readReply();
            usleep(100000);
            val[0] = 0x21; //HDMI
            displays.sendCommand(INPUT_SOURCE, val, 1);
        }
    }
    if (strcasecmp(argv[1], "BRIGHTNESS_SENSOR") == 0)
    {
        val[0] = 0;
        if (strcasecmp(argv[2], "on") == 0)
            val[0] = 1;
        displays.sendCommand(BRIGHTNESS_SENSOR, val, 1);
    }
    if (strcasecmp(argv[1], "AUTO_MOTION_PLUS") == 0)
    {
        val[0] = 0;
        if (strcasecmp(argv[2], "Clear") == 0)
            val[0] = 0x01;
        if (strcasecmp(argv[2], "Standard") == 0)
            val[0] = 0x02;
        if (strcasecmp(argv[2], "Smooth") == 0)
            val[0] = 0x03;
        if (strcasecmp(argv[2], "Custom") == 0)
            val[0] = 0x04;
        if (strcasecmp(argv[2], "Demo") == 0)
            val[0] = 0x05;
        val[1] = 5;
        val[2] = 5;
        displays.sendCommand(AUTO_MOTION_PLUS, val, 3);
    }
    if (strcasecmp(argv[1], "STATUS") == 0)
    {
        val[0] = 0;
        displays.sendCommand(STATUS, val, 0);
    }
    if (strcasecmp(argv[1], "DISPLAY_STATUS") == 0)
    {
        val[0] = 0;
        displays.sendCommand(DISPLAY_STATUS, val, 0);
    }
    if (strcasecmp(argv[1], "VIDEO") == 0)
    {
        val[0] = 0;
        displays.sendCommand(VIDEO, val, 0);
    }
    if (strcasecmp(argv[1], "MAINTENANCE") == 0)
    {
        val[0] = 0;
        displays.sendCommand(MAINTENANCE, val, 0);
    }
    if (strcasecmp(argv[1], "AUDIO") == 0)
    {
        val[0] = 0;
        displays.sendCommand(AUDIO, val, 0);
    }
    if (strcasecmp(argv[1], "DYNAMIC_CONTRAST") == 0)
    {
        val[0] = 0;
        if (strcasecmp(argv[2], "on") == 0)
            val[0] = 1;
        displays.sendCommand(DYNAMIC_CONTRAST, val, 1);
    }
    if (strcasecmp(argv[1], "PANEL") == 0)
    {
        val[0] = 0;
        if (strcasecmp(argv[2], "OFF") == 0)
            val[0] = 1;
        displays.sendCommand(PANEL_ON_OFF, val, 1);
    }
    if (strcasecmp(argv[1], "INPUT_SOURCE") == 0)
    {
        val[0] = 0;
        if (strcasecmp(argv[2], "PC") == 0)
            val[0] = 0x14;
        else if (strcasecmp(argv[2], "BNC") == 0)
            val[0] = 0x1E;
        else if (strcasecmp(argv[2], "DVI") == 0)
            val[0] = 0x18;
        else if (strcasecmp(argv[2], "AV") == 0)
            val[0] = 0x0C;
        else if (strcasecmp(argv[2], "S-VIDEO") == 0)
            val[0] = 0x04;
        else if (strcasecmp(argv[2], "COMPONENT") == 0)
            val[0] = 0x08;
        else if (strcasecmp(argv[2], "MAGICNET") == 0)
            val[0] = 0x20;
        else if (strcasecmp(argv[2], "HDMI") == 0)
            val[0] = 0x21;
        else if (strcasecmp(argv[2], "TV") == 0)
            val[0] = 0x30;
        else if (strcasecmp(argv[2], "DTV") == 0)
            val[0] = 0x40;
        displays.sendCommand(INPUT_SOURCE, val, 1);
    }

    displays.readReply();
}
