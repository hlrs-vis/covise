#include "NvTop.h"

extern "C" {
#include <nvtop/extract_gpuinfo.h>
}

#include <iostream>
#include <cover/coVRPluginSupport.h>
#include <cover/VRViewer.h>
#include <cover/coVRStatsDisplay.h>
#include <config/CoviseConfig.h>
#include <util/unixcompat.h>
#include <util/threadname.h>

using namespace opencover;


NvTop::NvTop()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

bool NvTop::init()
{
    init_gpu_info_extraction();
    m_numdevs = initialize_device_info(&m_threadDevInfos, size_t(0xffffffffL));

    m_deviceNum = covise::coCoviseConfig::getInt("device", "COVER.Plugin.NvTop", m_deviceNum);
    if (m_deviceNum >= m_numdevs)
        m_deviceNum = 0;

    auto stats = VRViewer::instance()->statsDisplay;
    if (m_numdevs>0 && stats) {
        std::cerr << "NvTop: enabling GPU stats for " << m_threadDevInfos[m_deviceNum].device_name << std::endl;
        stats->enableGpuStats(true, m_threadDevInfos[m_deviceNum].device_name);
        m_devinfos = new device_info[m_numdevs];
    }

    return m_numdevs>0;
}

NvTop::~NvTop()
{
    stopThread();

    delete[] m_devinfos;
    m_devinfos = nullptr;

    auto stats = VRViewer::instance()->statsDisplay;
    if (m_numdevs>0 && stats) {
        stats->enableGpuStats(false);
    }
    clean_device_info(m_numdevs, m_threadDevInfos);
    shutdown_gpu_info_extraction();
}

void NvTop::stopThread()
{
    if (!m_thread)
        return;

    {
        std::lock_guard<std::mutex> guard(m_mutex);
        m_needThread = false;
    }

    m_thread->join();
    m_thread.reset();
}

void NvTop::preFrame()
{
    auto stats = VRViewer::instance()->getViewerStats();
    if (!stats)
        return;

    if (m_numdevs == 0)
        return;

    bool collect = stats->collectStats("frame_rate");

    if (collect && !m_thread)
    {
        assert(m_needThread == false);
        m_needThread = true;

        m_thread.reset(new std::thread([this](){

            covise::setThreadName("NvTop");

            for (;;) {
                usleep(30000);

                {
                    std::lock_guard<std::mutex> guard(m_mutex);
                    if (!m_needThread)
                        return;
                }

                update_device_infos(m_numdevs, m_threadDevInfos);

                {
                    std::lock_guard<std::mutex> guard(m_mutex);
                    if (!m_needThread)
                        return;
                }

                std::lock_guard<std::mutex> guard(m_mutex);
                memcpy(m_devinfos, m_threadDevInfos, sizeof(m_devinfos[m_deviceNum])*m_numdevs);
            }
        }));
    }
    else if (!collect && m_thread)
    {
        stopThread();
        return;
    }

    std::lock_guard<std::mutex> guard(m_mutex);
    if (!m_devinfos)
        return;

    if (cover->debugLevel(3))
    {
        for (unsigned i=0; i<m_numdevs; ++i) {
            auto &dev = m_devinfos[i];
            std::cerr << "GPU " << i;
            if (IS_VALID(device_name_valid, dev.valid))
                std::cerr << " (" << dev.device_name << ")";
            std::cerr <<  ": ";
            if (IS_VALID(gpu_clock_speed_valid, dev.valid))
                std::cerr << "GPU " << dev.gpu_clock_speed << " MHz";
            if (IS_VALID(mem_clock_speed_valid, dev.valid))
                std::cerr << ", Mem " << dev.mem_clock_speed << " MHz";
            if (IS_VALID(pcie_rx_valid, dev.valid))
                std::cerr << ", PCEe upload " << dev.pcie_rx/1024 << " MB/s";
            std::cerr << std::endl;
        }
    }

    auto &dev = m_devinfos[m_deviceNum];

    if (stats && stats->collectStats("frame_rate"))
    {
        osg::FrameStamp *frameStamp = VRViewer::instance()->getViewerFrameStamp();

        if (IS_VALID(gpu_clock_speed_max_valid, dev.valid))
            stats->setAttribute(frameStamp->getFrameNumber(), "GPU Clock Max MHz", dev.gpu_clock_speed_max);
        if (IS_VALID(gpu_clock_speed_valid, dev.valid))
            stats->setAttribute(frameStamp->getFrameNumber(), "GPU Clock MHz", dev.gpu_clock_speed);
        if (IS_VALID(gpu_clock_speed_max_valid, dev.valid) && IS_VALID(gpu_clock_speed_valid, dev.valid))
            stats->setAttribute(frameStamp->getFrameNumber(), "GPU Clock Rate", (double)dev.gpu_clock_speed/dev.gpu_clock_speed_max);

        if (IS_VALID(mem_clock_speed_max_valid, dev.valid))
            stats->setAttribute(frameStamp->getFrameNumber(), "GPU Mem Clock Max MHz", dev.mem_clock_speed_max);
        if (IS_VALID(mem_clock_speed_valid, dev.valid))
            stats->setAttribute(frameStamp->getFrameNumber(), "GPU Mem Clock MHz", dev.mem_clock_speed);
        if (IS_VALID(mem_clock_speed_max_valid, dev.valid) && IS_VALID(mem_clock_speed_valid, dev.valid))
            stats->setAttribute(frameStamp->getFrameNumber(), "GPU Mem Clock Rate", (double)dev.mem_clock_speed/dev.mem_clock_speed_max);

        if (IS_VALID(pcie_rx_valid, dev.valid))
            stats->setAttribute(frameStamp->getFrameNumber(), "GPU PCIe rx KB/s", dev.pcie_rx);
        if (IS_VALID(pcie_tx_valid, dev.valid))
            stats->setAttribute(frameStamp->getFrameNumber(), "GPU PCIe tx KB/s", dev.pcie_tx);

        if (IS_VALID(gpu_util_rate_valid, dev.valid))
            stats->setAttribute(frameStamp->getFrameNumber(), "GPU Utilization", dev.gpu_util_rate/100.);

        if (IS_VALID(total_memory_valid, dev.valid))
            stats->setAttribute(frameStamp->getFrameNumber(), "GPU Mem Total", dev.total_memory);
        if (IS_VALID(used_memory_valid, dev.valid))
            stats->setAttribute(frameStamp->getFrameNumber(), "GPU Mem Used", dev.used_memory);
        if (IS_VALID(free_memory_valid, dev.valid))
            stats->setAttribute(frameStamp->getFrameNumber(), "GPU Mem Free", dev.free_memory);
    }
}

COVERPLUGIN(NvTop)
