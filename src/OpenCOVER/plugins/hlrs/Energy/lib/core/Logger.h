#pragma once

#include <string_view>
#include <string>
#include <functional>

class Logger
{
public:
    enum class Level
    {
        ERROR,
        WARN,
        INFO
    };

    using InfoStrategy = std::function<void(std::string_view)>;
    using ErrorStrategy = std::function<void(std::string_view)>;
    using WarnStrategy = std::function<void(std::string_view)>;

    explicit Logger(InfoStrategy info, ErrorStrategy error, WarnStrategy warn, std::string_view prefix = {}, int rank = 0)
        : info_(info)
        , error_(error)
        , warn_(warn)
        , prefix_(prefix)
        , rank_(rank)
    {
    }

    constexpr void info(std::string_view msg) const
    {
        log(Level::INFO, prefixMsg(msg));
    }

    constexpr void error(std::string_view msg) const
    {
        log(Level::ERROR, prefixMsg(msg));
    }

    constexpr void warn(std::string_view msg) const
    {
        log(Level::WARN, prefixMsg(msg));
    }
    
    void setPrefix(std::string_view prefix) {
        prefix_ = prefix;
    }

private:
    std::string prefixMsg(std::string_view msg) const
    {
        auto prefix { "[" + prefix_ + "] " };
        return prefix += msg;
    }

    constexpr void log(Level level, std::string_view msg) const
    {
        // display only for rank 0
        if (rank_ != 0)
            return;
        switch (level)
        {
        case Level::ERROR:
            error_(msg);
            break;
        case Level::WARN:
            warn_(msg);
            break;
        case Level::INFO:
        default:
            info_(msg);
        }
    }

    InfoStrategy info_;
    ErrorStrategy error_;
    WarnStrategy warn_;
    std::string prefix_;
    int rank_;
};
