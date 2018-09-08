// client.cpp


#include "vvclient.h"

#include <boost/bind.hpp>

#include <iomanip>
#include <iostream>
#include <string>
#include <thread>


using namespace virvo;


namespace boost { namespace uuids {

    std::ostream& operator <<(std::ostream& stream, uuid const& id)
    {
        for (size_t I = 0; I < 16; ++I)
            stream << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(id.data[I]);

        return stream;
    }

}}


class MyClient : public Client
{
public:
    MyClient(std::string const& host, unsigned short port)
    {
        connect(host, port);
    }

    //--- Client interface -----------------------------------------------------

    virtual void on_read(MessagePointer message)
    {
        std::cout << "CLIENT Message read: " << message->id() << " : \"" << message->deserialize<std::string>() << "\"" << std::endl;
    }

    virtual void on_write(MessagePointer message)
    {
        std::cout << "CLIENT Message sent: " << message->id() << std::endl;
    }

    void jippie(MessagePointer message)
    {
        std::cout << "My name is " << message->deserialize<std::string>() << std::endl;
    }
};


int main()
{
    try
    {
        // Create a new client
        MyClient client("127.0.0.1", 30000);

        // Start reading/writing messages
        std::thread runner(boost::bind(&Client::run, &client));

        std::string text;

        while (std::getline(std::cin, text))
        {
            if (text == "What's your name?")
            {
                client.write(makeMessage(0, text), boost::bind(&MyClient::jippie, &client, _1));
            }
            else
            {
                client.write(makeMessage(0/*type*/, text));
            }
        }

        // Stop processing messages
        client.stop();

        // Wait for the IO service to finish
        runner.join();
    }
    catch (std::exception& e)
    {
        std::cerr << "client: " << e.what() << std::endl;
    }
}
