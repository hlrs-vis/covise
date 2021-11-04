#ifndef COVISE_UTIL_ASYNC_WAIT
#define COVISE_UTIL_ASYNC_WAIT
#include <future>
#include <functional>
#include <vector>
#include <cassert>
#include <algorithm>
#include <util/coExport.h>
namespace covise
{

    //An AsyncWait executes the Condition in a separate thread.
    //The main loop has to check for met conditions via handleAsyncWaits.
    //If the Condition of and AsyncWait is met the Response is executed in the main thread.
    //AsyncWaits can be chained together via the >> operator eg. waitForA >> waitForB.
    //In this example waitForB's condition is only checked after waitForB's Response is executed.
class AsyncWaitInterface
{
public:
    virtual bool operator()() = 0;
    virtual void wait() = 0;
    virtual void remove() = 0;
};

typedef std::vector<std::unique_ptr<AsyncWaitInterface>> AsyncWaits;
extern UTILEXPORT AsyncWaits asyncWaits;

//has to be called in main loop to execute the Responses of AsyncWaits that have their condition met
void UTILEXPORT handleAsyncWaits();

template <typename Param>
class AsyncWaitClass : public AsyncWaitInterface
{
public:
    typedef std::function<Param(void)> Condition;
    typedef std::function<bool(const Param &)> Response;
    AsyncWaitClass(Condition condition, Response response)
        : m_response(response), m_condition(condition){}
    ~AsyncWaitClass() = default;
    AsyncWaitClass(AsyncWaitClass &&other) = delete;
    AsyncWaitClass(const AsyncWaitClass &) = delete;
    AsyncWaitClass &operator=(AsyncWaitClass &&) = delete;
    AsyncWaitClass &operator=(const AsyncWaitClass &) = delete;

    //wait for the condition to accure
    void wait()
    {
#ifdef _WIN32
        m_future = std::async(std::launch::async, [this]() {
            return std::unique_ptr<Param>{new Param{ m_condition() }};
            });
#else
        m_future = std::async(std::launch::async, m_condition);
#endif
    }

    bool operator()() override
    {
        if (m_future.valid() && m_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        {
#ifdef _WIN32
            m_response(*m_future.get());
#else
            m_response(m_future.get());
#endif
            return true;
        }
        return false;
    }

    template <typename OtherParam>
    AsyncWaitClass<OtherParam> &operator>>(AsyncWaitClass<OtherParam> &c)
    {

        next = &c;
        Response oldRes = m_response;
        m_response = [this, oldRes](const Param &r)
        {
            if (oldRes(r))
            {
                assert(next);
                next->wait();
                next = nullptr;
                return true;
            }
            remove();
            return false;
        };
        return *dynamic_cast<AsyncWaitClass<OtherParam> *>(next);
    }

    void remove() override
    {
        if (next)
        {
            next->remove();
            asyncWaits.erase(std::remove_if(asyncWaits.begin(), asyncWaits.end(), [this](const std::unique_ptr<AsyncWaitInterface> &ptr)
                                            { return ptr.get() == next; }));
            next = nullptr;
        }
    }

private:
    Response m_response;
    Condition m_condition;
#ifdef _WIN32
    std::future<std::unique_ptr<Param>> m_future;
#else
    std::future<Param> m_future;
#endif
    AsyncWaitInterface *next = nullptr;
};

template <typename Param>
AsyncWaitClass<Param> &AsyncWait(std::function<Param(void)> condition, std::function<bool(const Param&)> response)
{
    auto r = new AsyncWaitClass<Param>{condition, response};
    asyncWaits.emplace(asyncWaits.end(),  std::unique_ptr<AsyncWaitClass<Param>>{r});
    return *r;
}
} //covise

#endif // COVISE_UTIL_ASYNC_WAIT
