#pragma once
typedef enum { TypeCarSound = 0, TypeSimpleSound = 1, TypeRemoteSound = 2, TypeRemoteSoundDelay, SOUND_NEW_SOUND, SOUND_DELETE_SOUND, SOUND_SOUND_FILE, SOUND_SOUND_ID, SOUND_CLIENT_INFO } SoundMessages;
typedef enum {Start=0,Stop, enableLoop, disableLoop, Volume, Pitch} remoteSoundActions;
#pragma pack(push, 1)
struct CarSoundData
{
    int msgType;
    float engineSpeed;
    float carSpeed;
    float torque;
    float slip;
};
struct SimpleSoundData
{
    int msgType;
    unsigned char action;
    unsigned char soundNum;
};
struct RemoteSoundData
{
    int msgType;
    int soundID;
    float value;
    unsigned char action;
};
struct RemoteSoundDelayData
{
    int msgType;
    int soundID;
    unsigned long long startValue;
    unsigned long long endValue;
    bool stopChannel;
};

#pragma pack(pop)