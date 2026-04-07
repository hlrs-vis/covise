/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "AuralRealityPlugin.h"

#include <cover/coVRPluginSupport.h>

#include <grpc/grpc.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include "tmt_service.grpc.pb.h"

AuralRealityPlugin *AuralRealityPlugin::plugin = NULL;

using namespace opencover;

AuralRealityPlugin::AuralRealityPlugin()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , ui::Owner("AuralRealityPlugin", cover->ui)
{

    std::cout << "AuralReality boot" << std::endl;

    auto channel = grpc::CreateChannel("localhost:8080", grpc::InsecureChannelCredentials());
    auto stub = auralreality::TMTService::NewStub(channel);

    auralreality::Id request;
    request.set_id("hello");

    auralreality::Id response;

    grpc::ClientContext context;
    grpc::Status status = stub->Ping(&context, request, &response);
    if (status.ok())
    {
        std::cerr << "RPC pong: " << response.id() << std::endl;
    }
    else
    {
        std::cerr << "RPC failed: " << status.error_message() << std::endl
                  << "Error code: " << status.error_code() << std::endl;
    }
}

bool AuralRealityPlugin::update()
{
    return false;
}

AuralRealityPlugin::~AuralRealityPlugin()
{
}

COVERPLUGIN(AuralRealityPlugin)
