/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "XenomaiSocketCan.h"

#include <cstring>
#include <sstream>
#include <native/task.h>

void XenomaiSocketCan::printFrame(const char *text, const can_frame &frame)
{
    fprintf(stderr, text);
    if (frame.can_id & CAN_ERR_FLAG)
    {
        fprintf(stderr, "!0x%x!", (frame.can_id & CAN_ERR_MASK));
    }
    else if (frame.can_id & CAN_EFF_FLAG)
    {
        fprintf(stderr, "<0x%x>", (frame.can_id & CAN_EFF_MASK));
    }
    else
    {
        fprintf(stderr, "<0x%x>", (frame.can_id & CAN_SFF_MASK));
    }

    fprintf(stderr, "[%d]", (int)frame.can_dlc);

    if (!(frame.can_id & CAN_RTR_FLAG))
    {
        for (int i = 0; i < frame.can_dlc; i++)
        {
            fprintf(stderr, " %02x", (int)frame.data[i]);
        }
    }

    if (frame.can_id & CAN_ERR_FLAG)
    {
        fprintf(stderr, " ERROR: ");
        if (frame.can_id & CAN_ERR_BUSOFF)
            fprintf(stderr, "bus-off");
        if (frame.can_id & CAN_ERR_CRTL)
            fprintf(stderr, "controller problem");
    }
    else if (frame.can_id & CAN_RTR_FLAG)
    {
        fprintf(stderr, "remote request");
    }
    fprintf(stderr, "\n");
}

std::ostream &operator<<(std::ostream &out, const can_frame &frame)
{
    if (frame.can_id & CAN_ERR_FLAG)
    {
        out << "!0x" << std::hex << (frame.can_id & CAN_ERR_MASK) << "!";
    }
    else if (frame.can_id & CAN_EFF_FLAG)
    {
        out << "<0x" << std::hex << (frame.can_id & CAN_EFF_MASK) << ">";
    }
    else
    {
        out << "<0x" << std::hex << (frame.can_id & CAN_SFF_MASK) << ">";
    }

    out << " [" << std::dec << (int)frame.can_dlc << "]";

    if (!(frame.can_id & CAN_RTR_FLAG))
    {
        for (int i = 0; i < frame.can_dlc; i++)
        {
            out << " " << std::hex << (int)frame.data[i];
        }
    }

    if (frame.can_id & CAN_ERR_FLAG)
    {
        out << " ERROR: ";
        if (frame.can_id & CAN_ERR_BUSOFF)
            out << "bus-off";
        if (frame.can_id & CAN_ERR_CRTL)
            out << "controller problem";
    }
    else if (frame.can_id & CAN_RTR_FLAG)
    {
        out << " remote request";
    }

    return out;
}

XenomaiSocketCan::XenomaiSocketCan(const std::string &setDevice)
    : socket(-1)
    , device(setDevice)
{
    socket = rt_dev_socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (socket < 0)
    {
        std::cerr << "XenomaiSocketCan::XenomaiSocketCan(): rt_dev_socket: " << strerror(-socket) << std::endl;
        return;
    }

    ifreq ifr;
    strncpy(ifr.ifr_name, device.c_str(), IFNAMSIZ);
    int ret_ioctl = rt_dev_ioctl(socket, SIOCGIFINDEX, &ifr);
    if (ret_ioctl < 0)
    {
        std::cerr << "XenomaiSocketCan::XenomaiSocketCan(): rt_dev_ioctl(GET_IFINDEX): " << strerror(-ret_ioctl) << std::endl;
        return;
    }

    sockaddr_can recv_addr;
    recv_addr.can_family = AF_CAN;
    recv_addr.can_ifindex = ifr.ifr_ifindex;
    int ret_bind = rt_dev_bind(socket, (sockaddr *)&recv_addr, sizeof(sockaddr_can));
    if (ret_bind < 0)
    {
        std::cerr << "XenomaiSocketCan::XenomaiSocketCan(): rt_dev_bind: " << strerror(-ret_bind) << std::endl;
        return;
    }

    can_err_mask_t err_mask = CAN_ERR_MASK;
    int ret_filter = rt_dev_setsockopt(socket, SOL_CAN_RAW, CAN_RAW_ERR_FILTER, &err_mask, sizeof(err_mask));
    if (ret_filter < 0)
    {
        std::cerr << "XenomaiSocketCan::XenomaiSocketCan(): rt_dev_setsockopt: " << strerror(-ret_filter) << "!" << std::endl;
    }
}

XenomaiSocketCan::~XenomaiSocketCan()
{
    if (socket >= 0)
    {
        int ret_close = rt_dev_close(socket);
        if (ret_close)
        {
            std::cout << "XenomaiSocketCan::~XenomaiSocketCan(): rt_dev_close: " << strerror(-ret_close) << std::endl;
        }
    }
}

void XenomaiSocketCan::addRecvFilter(u_int32_t id, u_int32_t mask)
{
    can_filter recv_filter = { id, mask };
    recvFilterVector.push_back(recv_filter);
}

int XenomaiSocketCan::applyRecvFilters()
{
    can_filter *recvFilters = new can_filter[recvFilterVector.size()];
    copy(recvFilterVector.begin(), recvFilterVector.end(), recvFilters);

    int ret_filter = rt_dev_setsockopt(socket, SOL_CAN_RAW, CAN_RAW_FILTER, recvFilters, recvFilterVector.size() * sizeof(can_filter));
    if (ret_filter < 0)
    {
        std::cerr << "XenomaiSocketCan::addFilter(): rt_dev_setsockopt: " << strerror(-ret_filter) << "!" << std::endl;
    }

    delete[] recvFilters;

    return ret_filter;
}

int XenomaiSocketCan::setRecvTimeout(nanosecs_rel_t timeout)
{
    int ret_timeout = rt_dev_ioctl(socket, RTCAN_RTIOC_RCV_TIMEOUT, &timeout);
    if (ret_timeout)
    {
        std::cerr << "XenomaiSocketCan::setRecvTimeout(): rt_dev_ioctl: RCV_TIMEOUT: " << strerror(-ret_timeout) << std::endl;
    }

    return ret_timeout;
}

int XenomaiSocketCan::setSendTimeout(nanosecs_rel_t timeout)
{
    int ret_timeout = rt_dev_ioctl(socket, RTCAN_RTIOC_SND_TIMEOUT, &timeout);
    if (ret_timeout)
    {
        std::cerr << "XenomaiSocketCan::setSendTimeout(): rt_dev_ioctl: RCV_TIMEOUT: " << strerror(-ret_timeout) << std::endl;
    }

    return ret_timeout;
}

void XenomaiSocketCan::printFrame(const can_frame &frame)
{
    if (frame.can_id & CAN_ERR_FLAG)
    {
        std::cerr << "!0x" << std::hex << (frame.can_id & CAN_ERR_MASK) << "!";
    }
    else if (frame.can_id & CAN_EFF_FLAG)
    {
        std::cerr << "<0x" << std::hex << (frame.can_id & CAN_EFF_MASK) << ">";
    }
    else
    {
        std::cerr << "<0x" << std::hex << (frame.can_id & CAN_SFF_MASK) << ">";
    }

    std::cerr << " [" << std::dec << (int)frame.can_dlc << "]";

    if (!(frame.can_id & CAN_RTR_FLAG))
    {
        for (int i = 0; i < frame.can_dlc; i++)
        {
            std::cerr << " " << std::hex << (int)frame.data[i];
        }
    }

    if (frame.can_id & CAN_ERR_FLAG)
    {
        std::cerr << " ERROR: ";
        if (frame.can_id & CAN_ERR_BUSOFF)
            std::cerr << "bus-off";
        if (frame.can_id & CAN_ERR_CRTL)
            std::cerr << "controller problem";
    }
    else if (frame.can_id & CAN_RTR_FLAG)
    {
        std::cerr << " remote request";
    }

    std::cerr << std::endl;
}

std::string XenomaiSocketCan::frameToString(const can_frame &frame)
{
    std::stringstream frameStream;

    if (frame.can_id & CAN_ERR_FLAG)
    {
        frameStream << "!0x" << std::hex << (frame.can_id & CAN_ERR_MASK) << "!";
    }
    else if (frame.can_id & CAN_EFF_FLAG)
    {
        frameStream << "<0x" << std::hex << (frame.can_id & CAN_EFF_MASK) << ">";
    }
    else
    {
        frameStream << "<0x" << std::hex << (frame.can_id & CAN_SFF_MASK) << ">";
    }

    frameStream << " [" << std::dec << (int)frame.can_dlc << "]";

    if (!(frame.can_id & CAN_RTR_FLAG))
    {
        for (int i = 0; i < frame.can_dlc; i++)
        {
            frameStream << " " << std::hex << (int)frame.data[i];
        }
    }

    if (frame.can_id & CAN_ERR_FLAG)
    {
        frameStream << " ERROR: ";
        if (frame.can_id & CAN_ERR_BUSOFF)
            frameStream << "bus-off";
        if (frame.can_id & CAN_ERR_CRTL)
            frameStream << "controller problem";
    }
    else if (frame.can_id & CAN_RTR_FLAG)
    {
        frameStream << " remote request";
    }

    return frameStream.str();
}
