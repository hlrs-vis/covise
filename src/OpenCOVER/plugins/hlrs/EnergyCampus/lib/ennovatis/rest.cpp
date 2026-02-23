#include "rest.h"

#include <string>

#include "HTTPClient/CURL/methods.h"
#include "HTTPClient/CURL/request.h"
#include "building.h"
#include "date.h"

using namespace std;
using namespace opencover::httpclient::curl;
namespace ennovatis {

void rest_request_handler::fetchChannels(const ChannelGroup &group,
                                         const ennovatis::Building &b,
                                         ennovatis::rest_request req) {
  auto input = b.getChannels(group);
  for (auto &channel : input) {
    req.channelId = channel.id;
    worker.addThread(std::async(std::launch::async, rest::fetch_data, req));
  }
}

std::string rest_request::operator()() const {
  auto _dtf = date::time_point_to_str(dtf, date::dateformat);
  auto _dtt = date::time_point_to_str(dtt, date::dateformat);
  return url + "?projEid=" + projEid + "&dtf=" + _dtf + "&dtt=" + _dtt +
         "&ts=" + std::to_string(ts) + "&tsp=" + std::to_string(tsp) +
         "&tst=" + std::to_string(tst) + "&etst=" + std::to_string(etst) +
         "&cEid=" + channelId;
}

std::string rest::fetch_data(const rest_request &req) {
  std::string response = "";
  GET getRequest(req());
  if (!Request().httpRequest(getRequest, response))
    response = "[ERROR] Failed to fetch data from Ennovatis. With request: " + req();
  return response;
}
}  // namespace ennovatis
