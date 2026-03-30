#pragma once
typedef enum
{
    TypeCarSound = 0,
    TypeSimpleSound = 1,
    _usused_1 = 2,
    _unused_2,
    SOUND_NEW_SOUND,
    SOUND_DELETE_SOUND,
    SOUND_SOUND_FILE,
    SOUND_SOUND_ID,
    SOUND_CLIENT_INFO
} SoundMessages;
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

#pragma pack(pop)
