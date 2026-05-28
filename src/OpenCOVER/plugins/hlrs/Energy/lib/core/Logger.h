#pragma once

#include <string_view>
#include <string>
#include <functional>

class Logger
{
public:
    enum class LogLevel
    {
        ERROR,
        WARN,
        INFO
    };
    
    enum class DebugLevel
    {
        ALL, // info + warn + error
        MINIMAL, // info
        NONE 
    };

    using Rank = int;
    using Message = std::string_view;
    using Prefix = std::string_view;
    using InfoStrategy = std::function<void(Message)>;
    using ErrorStrategy = std::function<void(Message)>;
    using WarnStrategy = std::function<void(Message)>;

    explicit Logger(InfoStrategy info, ErrorStrategy error, WarnStrategy warn, Prefix prefix = {}, Rank rank = 0, DebugLevel debug = DebugLevel::MINIMAL)
        : info_(info)
        , error_(error)
        , warn_(warn)
        , prefix_(prefix)
        , rank_(rank)
        , debug_(debug)
    {
    }

    constexpr void info(Message msg) const
    {
        if (debug_ == DebugLevel::NONE)
            return;
        log(LogLevel::INFO, prefixMsg(msg));
    }

    constexpr void error(Message msg) const
    {
        if (debug_ != DebugLevel::ALL)
            return;

        log(LogLevel::ERROR, prefixMsg(msg));
    }

    constexpr void warn(Message msg) const
    {
        if (debug_ != DebugLevel::ALL)
            return;

        log(LogLevel::WARN, prefixMsg(msg));
    }
    
    constexpr void setPrefix(Prefix prefix) {
        prefix_ = prefix;
    }

    constexpr void setDebugLeve(DebugLevel debug) {
        debug_ = debug;
    }

private:
    const std::string prefixMsg(Message msg) const
    {
        auto prefix { "[" + prefix_ + "] " };
        return prefix += msg;
    }

    constexpr void log(LogLevel level, Message msg) const
    {
        // display only for rank 0
        if (rank_ != 0)
            return;
        switch (level)
        {
        case LogLevel::ERROR:
            error_(msg);
            break;
        case LogLevel::WARN:
            warn_(msg);
            break;
        case LogLevel::INFO:
        default:
            info_(msg);
        }
    }

    InfoStrategy info_;
    ErrorStrategy error_;
    WarnStrategy warn_;
    std::string prefix_;
    Rank rank_;
    DebugLevel debug_;
};
