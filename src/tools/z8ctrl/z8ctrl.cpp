/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <iostream>
#include <util/UDP_Sender.h>

using namespace std;
using namespace covise;

enum Type{
    INTEGER, UNSIGNED_SHORT
};

const char *usage = "Usage: z8ctrl [3Don|3Doff|preset n|brightness n|test n|black|wakeup]\n"
                    "  3Don: turn 3D on\n"
                    "  3Doff: turn 3D off\n"
                    "  preset n: set preset to n 0 = HDMI1, 1 = HDMI2, 2=STEREO(DP1), 3=DP1, 4=DP2; stereo has to be toggled independently\n"
                    "  brightness n: set brightness to n (0-100)\n"
                    "  test n: set testpattern to n  with n [0-14] n=0 for normal image\n"
                    "  black: set black screen, revert with wakeup\n"
                    "  wakeup: wake up\n";

struct Command
{
    const char *name;
    std::vector<unsigned char> buf;
    int argpos = -1;
    Type type = INTEGER;
};

static const auto commands = {
     //                    00     01    02    03    04    05    06    07    08    09    10    11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28    29    30    31    32    33    34    35    36    37    38    39    40    41    42    43    44    45    46    47    48    49    50    51    62    63    64    65    66    67   68   69   70
    Command{"HDMI",      {0x20, 0x10, 0x00, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x00, 0x1b,  0x00, 0x20, 0x00 }},
    Command{"DP",      {0x20, 0x10, 0x00, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x00, 0x17,  0x00, 0x21, 0x00 }},
    Command{"3DON",       {0x80, 0x10, 0x01, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01 }},
    Command{"3DOFF",      {0x80, 0x10, 0x01, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }},
    Command{"PRESET",     {0x02, 0x10, 0x00, 0x13, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }, 17, UNSIGNED_SHORT},
    //Command{"BRIGHTNESS", {0x50, 0x10, 0x00, 0x33, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }, 49, UNSIGNED_SHORT},
    Command{"BRIGHTNESS", {0x50, 0x10, 0x00, 0x13, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }, 17, UNSIGNED_SHORT},
    Command{"TEST",       {0x12, 0x10, 0x00, 0x13, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00 }, 17},
    Command{"BLACK", {0x10, 0x10, 0x00, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01 }},
    Command{"WAKEUP", {0x10, 0x10, 0x00, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }}
};

int main(int argc, char **argv)
{
    UDP_Sender z8("192.168.1.192", 9099);
    std::unique_ptr<Command> cmd;
    for (int i = 1; i < argc; i++)
    {
        string arg = argv[i];
        transform(arg.begin(), arg.end(), arg.begin(), ::toupper);

        if(cmd)
        {
            int num = 0;
            try
            {   
                num = std::stoi(arg);
            }
            catch(const std::exception&)
            {
                std::cerr << "invalid argument" << std::endl;
                return 1;
            }
            if(cmd->type == INTEGER)
            {
                cmd->buf[cmd->argpos] = num;
            } else {
                unsigned short i = num;
                *((unsigned short *)(&(cmd->buf[cmd->argpos]))) = num;
            }
            z8.send(cmd->buf.data(), cmd->buf.size());
            std::cerr << "setting " << cmd->name << " to " << num << std::endl;
            return 0;
        }
        for(const auto command : commands)
        {
            if(arg == command.name)
            {
                if(command.argpos != -1)
                {
                    cmd = std::make_unique<Command>(command);
                    break;
                }
                else
                {
                    z8.send(command.buf.data(), command.buf.size());
                    std::cerr << "setting " << command.name << std::endl;
                    return 0;
                }
            }
        }
    }
    std::cerr << usage << std::endl;
    return 1;
 }
