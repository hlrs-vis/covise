#pragma once
#include "Parser.h"
#include "app/typedefs.h"
#include "GridRenderer.h"
#include <lib/core/interfaces/IEnergyGrid.h>
#include <lib/core/Logger.h>

struct GridParser : DataPackageParser<grid_ptr>
{
    GridParser(Logger logger, const std::string &name, GridRenderConfig &config)
        : m_logger(std::move(logger))
        , m_config(config)
    {
        m_logger.setPrefix(name);
    }

    GridRenderConfig m_config;
    Logger m_logger;
};
