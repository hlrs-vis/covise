#include "ConnectorZmq.h"

#include <iostream>

#include <net/tokenbuffer.h>
#include <traffic_simulator.pb.h>
#include <cover/coVRMSController.h>

#include "Traffic.h"

ConnectorZmq::ConnectorZmq()
    : zmq_command_socket(zmq_context, zmq::socket_type::req)
    , zmq_subscriber_socket(zmq_context, zmq::socket_type::sub)
{
    connect();
}

void ConnectorZmq::connect()
{
    zmq_command_socket.connect("tcp://localhost:4101");
    zmq_subscriber_socket.connect("tcp://localhost:4102");
    zmq_subscriber_socket.set(zmq::sockopt::subscribe, ""); // everything

    sendCommands();
}

bool ConnectorZmq::update(double deltaTime, double simulationDeltaTime)
{
    zmq::message_t msg;
    if (!zmq_subscriber_socket.recv(msg, zmq::recv_flags::dontwait))
    {
        return false;
    }

    std::stringstream s;
    s.write(reinterpret_cast<char *>(msg.data()), msg.size());

    traffic_simulator::SimulationState state;
    state.ParseFromIstream(&s);

    // std::cout << "Got message: status=" << state.status() << std::endl;
    // std::cout << "Got message: current_time=" << state.current_time() << std::endl;
    // std::cout << "Got message: delta_time=" << state.delta_time() << std::endl;
    // std::cout << "Got message: run_speed=" << state.run_speed() << std::endl;

    // read data

    vehicles.clear();
    vehicles.reserve(state.vehicles_size());

    for (auto v : state.vehicles())
    {
        vehicles.push_back(VehicleState {
            v.id(),
            "passenger",
            osg::Vec3d(v.x(), v.y(), v.z()),
            v.angle(),
        });
    }

    return true;
}

void ConnectorZmq::getSimulationState(SimulationState &state)
{
    state.vehicles = vehicles;
}

bool ConnectorZmq::isConnected() const
{
    return true;
}

void ConnectorZmq::sendCommands()
{
}
