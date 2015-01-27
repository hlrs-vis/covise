/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// FIXME this should be exported from the SDK!
#include "notetrck.h"

class NoteKey
{
public:
    TimeValue time;
    TSTR note;
    DWORD flags;

    NoteKey(TimeValue t, const TSTR &n, DWORD f = 0)
    {
        time = t;
        note = n;
        flags = f;
    }
    NoteKey(NoteKey &n)
    {
        time = n.time;
        note = n.note;
        flags = n.flags;
    }

    void SetFlag(DWORD mask)
    {
        flags |= (mask);
    }
    void ClearFlag(DWORD mask)
    {
        flags &= ~(mask);
    }
    BOOL TestFlag(DWORD mask)
    {
        return (flags & (mask) ? 1 : 0);
    }
};

class NoteKeyTab : public Tab<NoteKey *>
{
public:
    ~NoteKeyTab()
    {
        Clear();
    }
    void Clear();
    void DelKey(int i)
    {
        delete (*this)[i];
        Delete(i, 1);
    }
    NoteKeyTab &operator=(NoteKeyTab &keys);
    void KeysChanged();
};

class DefNoteTrack : public NoteTrack
{
public:
    NoteKeyTab keys;

    DefNoteTrack()
    {
    }
    DefNoteTrack(DefNoteTrack &n)
    {
        keys = n.keys;
    }
    DefNoteTrack &operator=(DefNoteTrack &track)
    {
        keys = track.keys;
        return *this;
    }
    void HoldTrack();

    Class_ID ClassID()
    {
        return Class_ID(NOTETRACK_CLASS_ID, 0);
    }

    // Tree view methods from animatable
    int NumKeys()
    {
        return keys.Count();
    }
    TimeValue GetKeyTime(int index)
    {
        return keys[index]->time;
    }
    void MapKeys(TimeMap *map, DWORD flags);
    void DeleteKeys(DWORD flags);
    void CloneSelectedKeys();
    void DeleteTime(Interval iv, DWORD flags);
    void ReverseTime(Interval iv, DWORD flags);
    void ScaleTime(Interval iv, float s);
    void InsertTime(TimeValue ins, TimeValue amount);
    void AddNewKey(TimeValue t, DWORD flags);
    int GetSelKeyCoords(TimeValue &t, float &val, DWORD flags);
    void SetSelKeyCoords(TimeValue t, float val, DWORD flags);
    int GetTrackVSpace(int lineHeight)
    {
        return 1;
    }
    BOOL CanCopyTrack(Interval iv, DWORD flags)
    {
        return 1;
    }
    BOOL CanPasteTrack(TrackClipObject *cobj, Interval iv, DWORD flags)
    {
        return cobj->ClassID() == ClassID();
    }
    TrackClipObject *CopyTrack(Interval iv, DWORD flags);
    void PasteTrack(TrackClipObject *cobj, Interval iv, DWORD flags);
    Interval GetTimeRange(DWORD flags);
    int HitTestTrack(TrackHitTab &hits, Rect &rcHit, Rect &rcTrack, float zoom, int scroll, DWORD flags);
    int PaintTrack(HDC hdc, Rect &rcTrack, Rect &rcPaint, float zoom, int scroll, DWORD flags);
    void SelectKeys(TrackHitTab &sel, DWORD flags);
    int NumSelKeys();
    void FlagKey(TrackHitRecord hit);
    int GetFlagKeyIndex();
    BOOL IsAnimated()
    {
        return keys.Count() > 1;
    }
    void EditTrackParams(TimeValue t, ParamDimensionBase *dim, TCHAR *pname, HWND hParent, IObjParam *ip, DWORD flags);
    int TrackParamsType()
    {
        return TRACKPARAMS_KEY;
    }
    BOOL SupportTimeOperations()
    {
        return TRUE;
    }

    IOResult Save(ISave *isave);
    IOResult Load(ILoad *iload);

    void DeleteThis()
    {
        delete this;
    }
    RefResult NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget,
                               PartID &partID, RefMessage message)
    {
        return REF_SUCCEED;
    }
};
