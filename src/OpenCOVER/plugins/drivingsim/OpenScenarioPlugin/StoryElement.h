#ifndef STORYELEMENT_H
#define STORYELEMENT_H


class StoryElement
{
public:
    StoryElement();
	virtual ~StoryElement() {};
    enum State
    {
        finished,
        running,
        stopped
    };

    virtual void stop();
    virtual void start();
    virtual void finish();

    bool isRunning();
    bool isStopped();
    bool isFinished();
private:

	State state;
};

#endif // STORYELEMENT_H
