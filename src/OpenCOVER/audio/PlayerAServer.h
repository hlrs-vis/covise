/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PLAYER_ASERVER_
#define _PLAYER_ASERVER_

#include "Player.h"

namespace opencover::audio
{

class Listener;

class COVRAUDIOEXPORT PlayerAServer : public Player
{
public:
    PlayerAServer(const Listener *listener, const std::string &host, int port);
    virtual ~PlayerAServer();
    virtual std::unique_ptr<Source> makeSource(Player *player, const Audio *audio);

    virtual int send_cmd(const char *cmd) const;
    virtual int send_data(const char *data, int size, bool swapped = false) const;
    virtual int read_answer(char *buf, int maxsize) const;

protected:
    void connect();
    class Source : public Player::Source
    {
    public:
        Source(Player *player, const Audio *audio);
        virtual ~Source();
        virtual void setAudio(const Audio *audio) override;
        virtual void play(double start) override;
        virtual void play() override;
        virtual void stop() override;

        virtual void setLoop(bool loop) override;

        virtual void update(const Player *player) override;

        int asHandle;
        float odirection;
        float ointensity;
        float opitch;
        glm::vec3 ov;

    private:
        virtual void loadAudio(const Audio *audio);
    };

    mutable int asFd;
    std::string asHost;
    int asPort;
};
}
#endif
