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

    using Rank = int;
    using Message = std::string_view;
    using Prefix = std::string_view;
    using InfoStrategy = std::function<void(Message)>;
    using ErrorStrategy = std::function<void(Message)>;
    using WarnStrategy = std::function<void(Message)>;

    explicit Logger(InfoStrategy info, ErrorStrategy error, WarnStrategy warn, Prefix prefix = {}, Rank rank = 0)
        : info_(info)
        , error_(error)
        , warn_(warn)
        , prefix_(prefix)
        , rank_(rank)
    {
    }

    constexpr void info(Message msg) const
    {
        log(LogLevel::INFO, prefixMsg(msg));
    }

    constexpr void error(Message msg) const
    {
        log(LogLevel::ERROR, prefixMsg(msg));
    }

    constexpr void warn(Message msg) const
    {
        log(LogLevel::WARN, prefixMsg(msg));
    }
    
    constexpr void setPrefix(Prefix prefix) {
        prefix_ = prefix;
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
};
