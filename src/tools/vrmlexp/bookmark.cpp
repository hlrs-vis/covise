/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: bookmark.cpp

    DESCRIPTION:  A Dialog box to get a URL
 
    CREATED BY: Charles Thaeler
 
    HISTORY: created 26 Mar. 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#include "vrml.h"
#include "bookmark.h"

// in prim.cpp  - The dll instance handle
extern HINSTANCE hInstance;

class Bookmark
{
public:
    TSTR url;
    TSTR desc;
};

static Tab<Bookmark *> bookmarks;

static TSTR url;
static TSTR desc;
static BOOL initBookmarks = TRUE;

void
SplitURL(TSTR in, TSTR *url, TSTR *camera)
{
    if (camera)
    {
        TSTR temp = in;
        int i = temp.last('#');
        if (i >= 0)
        {
            *camera = temp.Substr(i + 1, temp.Length() - i - 1);
            *url = temp.Substr(0, i);
        }
        else
        {
            *camera = _T(""); // no camera
            *url = in;
        }
    }
    else
        *url = in;
}

static void
LoadBookmarks(Interface *ip, TCHAR *filename, HWND hlist)
{
#define MAX_STR 1024
    if (filename)
    {
        FILE *fp = _tfopen(filename, _T("r"));
        if (fp == NULL)
        {
            // Warning box?
            return;
        }
        int inchar;
        while ((inchar = getc(fp)) != EOF)
        {
            // loop till we find an A HREF="
            if ((inchar == 'A') && (((inchar = getc(fp)) != EOF && inchar == ' ') && ((inchar = getc(fp)) != EOF && inchar == 'H') && ((inchar = getc(fp)) != EOF && inchar == 'R') && ((inchar = getc(fp)) != EOF && inchar == 'E') && ((inchar = getc(fp)) != EOF && inchar == 'F') && ((inchar = getc(fp)) != EOF && inchar == '=') && ((inchar = getc(fp)) != EOF && inchar == '"')))
            {
                TCHAR url[MAX_STR];
                TCHAR desc[MAX_STR];
                int count = 0;
                url[0] = desc[0] = '\0';

                while ((inchar = getc(fp)) != EOF)
                {
                    // loop till we find the closing " or are too big
                    if (inchar == '"' || count >= (MAX_STR - 2))
                    {
                        url[count] = '\0';
                        if (inchar == '"')
                            break;
                    }
                    else
                        url[count++] = inchar;
                }
                count = 0;
                if (inchar != EOF)
                {
                    // loop till we find a >
                    while ((inchar = getc(fp)) != EOF && inchar != '>')
                        ;
                    if (inchar != EOF)
                    {
                        // Everything till the next < is a description
                        while ((inchar = getc(fp)) != EOF)
                        {
                            // loop till we find the closing " or are too big
                            if (inchar == '<' || count >= (MAX_STR - 2))
                            {
                                desc[count] = '\0';
                                if (inchar == '<')
                                    break;
                            }
                            else
                                desc[count++] = inchar;
                        }
                    }
                }
                // OK I think we have have a valid URL and Description so we need
                // to add it to the bookmarks
                Bookmark *bm = new Bookmark();
                bm->url = url;
                bm->desc = desc;
                int pos = bookmarks.Append(1, &bm);
                if (hlist)
                {
                    LV_ITEM item;
                    item.mask = LVIF_TEXT | LVIF_STATE | LVIF_PARAM;
                    item.iItem = pos;
                    item.iSubItem = 0;
                    item.state = 0;
                    item.stateMask = 0;
                    item.lParam = (LPARAM)bookmarks[pos];
                    item.pszText = (TCHAR *)bm->url.data();
                    item.cchTextMax = (int)_tcslen(item.pszText) + 1;
                    ListView_InsertItem(hlist, &item);
                    ListView_SetItemText(hlist, pos, 1, (TCHAR *)bm->desc.data());
                    ListView_EnsureVisible(hlist, pos, FALSE);
                }
            }
        }
    }
    else
    {
// I wonder how big it should be?
#define BM_SIZE 32768
        //Load from the ini file.
        TCHAR bm[BM_SIZE], *pbm;

        int rc = GetPrivateProfileSection(BOOKMARK_SECTION, bm, BM_SIZE,
                                          ExportIniFilename(ip));
        // Clean out the old list if it exists
        for (int i = bookmarks.Count() - 1; i >= 0; i--)
        {
            delete bookmarks[i];
            bookmarks.Delete(i, 1);
        }
        if (rc > 0 && rc != (BM_SIZE - 2))
        {
            pbm = bm;
            while (*pbm)
            {
                Bookmark *nbm = new Bookmark();
                TCHAR *p = pbm;
                while (p)
                {
                    if (*p == '=')
                    {
                        *p = '\0';
                        break;
                    }
                    p++;
                }
                // the first string is a key -- url
                nbm->url = pbm;
                pbm += _tcslen(pbm) + 1; // Step beyond the null

                // the next string is a value -- description
                nbm->desc = pbm;
                pbm += _tcslen(pbm) + 1; // Step beyond the null
                bookmarks.Append(1, &nbm);
            }
        }
    }
}

static void
SaveBookmarks(Interface *ip)
{
    //Load from the ini file.
    // First Nuke the old ones
    WritePrivateProfileSection(BOOKMARK_SECTION, _T("\0\0"),
                               ExportIniFilename(ip));
    for (int i = 0; i < bookmarks.Count(); i++)
        WritePrivateProfileString(BOOKMARK_SECTION,
                                  bookmarks[i]->url, bookmarks[i]->desc,
                                  ExportIniFilename(ip));
}

static INT_PTR CALLBACK
    URLBookmarkDlgProc(HWND hDlg, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (msg == WM_INITDIALOG)
        SetWindowLongPtr(hDlg, GWLP_USERDATA, lParam);

    Interface *ip = (Interface *)GetWindowLongPtr(hDlg, GWLP_USERDATA);

    HWND hlist = GetDlgItem(hDlg, IDC_URL_LIST);

    switch (msg)
    {
    case WM_INITDIALOG:
    {
        int rc;
        LV_COLUMN col;

        col.mask = LVCF_FMT | LVCF_WIDTH | LVCF_TEXT;
        col.fmt = LVCFMT_LEFT;
        col.cx = 260; // allow for the URL, the remainder is the description
        col.pszText = GetString(IDS_PRS_URL);
        rc = ListView_InsertColumn(hlist, 0, &col);

        col.mask = LVCF_FMT | LVCF_WIDTH | LVCF_TEXT;
        col.fmt = LVCFMT_LEFT;
        col.cx = 260;
        col.pszText = GetString(IDS_PRS_DESCRIPTION);
        rc = ListView_InsertColumn(hlist, 1, &col);

        for (int i = 0; i < bookmarks.Count(); i++)
        {
            LV_ITEM item;
            item.mask = LVIF_TEXT | LVIF_STATE | LVIF_PARAM;
            item.iItem = i;
            item.iSubItem = 0;
            item.state = 0;
            item.stateMask = 0;
            item.lParam = (LPARAM)bookmarks[i];
            item.pszText = (TCHAR *)bookmarks[i]->url.data();
            item.cchTextMax = (int)_tcslen(item.pszText) + 1;
            rc = ListView_InsertItem(hlist, &item);
            ListView_SetItemText(hlist, i, 1, (TCHAR *)bookmarks[i]->desc.data());
        }

        SendMessage(GetDlgItem(hDlg, IDC_URL), WM_SETTEXT, 0, (LPARAM)url.data());
        EnableWindow(GetDlgItem(hDlg, IDC_URL), TRUE);

        SendMessage(GetDlgItem(hDlg, IDC_DESC), WM_SETTEXT, 0, (LPARAM)desc.data());
        EnableWindow(GetDlgItem(hDlg, IDC_DESC), TRUE);

        int numsel = ListView_GetSelectedCount(hlist);
        EnableWindow(GetDlgItem(hDlg, IDC_DELETE), numsel > 0);

        CenterWindow(hDlg, GetParent(hDlg));
    }
        return TRUE;
    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDC_IMPORT:
        {
            OPENFILENAME ofn;
            TCHAR filename[MAX_PATH];
            TCHAR *filterStr = _T("Bookmarks\0*.htm;*.html\0\0");
            filename[0] = '\0';
            ofn.lStructSize = sizeof(OPENFILENAME);
            ofn.hInstance = hInstance;
            ofn.hwndOwner = hDlg;
            ofn.lpstrFilter = filterStr;
            ofn.lpstrCustomFilter = NULL;
            ofn.nFilterIndex = 0;
            ofn.lpstrFile = filename;
            ofn.nMaxFile = MAX_PATH;
            ofn.lpstrFileTitle = NULL;
            ofn.nMaxFileTitle = 0;
            ofn.lpstrInitialDir = NULL;
            ofn.lpstrTitle = NULL;
            ofn.Flags = OFN_FILEMUSTEXIST | OFN_LONGNAMES | OFN_PATHMUSTEXIST;
            ofn.lpstrDefExt = _T("htm");

            if (GetOpenFileName(&ofn))
                LoadBookmarks(ip, ofn.lpstrFile, hlist);
            return TRUE;
        }
        case IDC_ADD:
        {
            Bookmark *bm = new Bookmark();

            int len = (int)SendDlgItemMessage(hDlg, IDC_URL, WM_GETTEXTLENGTH, 0, 0);
            TSTR temp;
            temp.Resize(len + 1);
            SendDlgItemMessage(hDlg, IDC_URL, WM_GETTEXT, len + 1, (LPARAM)temp.data());
            bm->url = temp;

            len = (int)SendDlgItemMessage(hDlg, IDC_DESC, WM_GETTEXTLENGTH, 0, 0);
            temp.Resize(len + 1);
            SendDlgItemMessage(hDlg, IDC_DESC, WM_GETTEXT, len + 1, (LPARAM)temp.data());
            bm->desc = temp;

            int pos = bookmarks.Append(1, &bm);

            LV_ITEM item;
            item.mask = LVIF_TEXT | LVIF_STATE | LVIF_PARAM;
            item.iItem = pos;
            item.iSubItem = 0;
            item.state = 0;
            item.stateMask = 0;
            item.lParam = (LPARAM)bookmarks[pos];
            item.pszText = (TCHAR *)bm->url.data();
            item.cchTextMax = (int)_tcslen(item.pszText) + 1;
            ListView_InsertItem(hlist, &item);
            ListView_SetItemText(hlist, pos, 1, (TCHAR *)bm->desc.data());
            ListView_EnsureVisible(hlist, pos, FALSE);
        }
            return TRUE;
        case IDC_DELETE:
        {
            int numsel = ListView_GetSelectedCount(hlist);
            if (numsel > 0)
            {
                // need to find and delete them, traverse backwards since the list shrinks
                for (int i = ListView_GetItemCount(hlist) - 1; i >= 0; i--)
                {
                    LV_ITEM item;
                    item.mask = LVIF_STATE | LVIF_PARAM;
                    item.iSubItem = 0;
                    item.state = 0;
                    item.stateMask = LVIS_SELECTED;
                    item.iItem = i;
                    ListView_GetItem(hlist, &item);
                    if (item.state & LVIS_SELECTED)
                    {
                        ListView_DeleteItem(hlist, i);
                        // Now find the bookmark and delete it
                        for (int j = 0; j < bookmarks.Count(); j++)
                        {
                            if (bookmarks[j] == (Bookmark *)item.lParam)
                            {
                                bookmarks.Delete(j, 1);
                                break;
                            }
                        }
                    }
                }
                EnableWindow(GetDlgItem(hDlg, IDC_DELETE), FALSE);
            }
        }
            return TRUE;
        case IDCANCEL:
            EndDialog(hDlg, FALSE);
            return TRUE;
        case IDOK:
        {
            int len = (int)SendDlgItemMessage(hDlg, IDC_URL, WM_GETTEXTLENGTH, 0, 0);
            TSTR temp;
            temp.Resize(len + 1);
            SendDlgItemMessage(hDlg, IDC_URL, WM_GETTEXT, len + 1, (LPARAM)temp.data());
            url = temp;

            len = (int)SendDlgItemMessage(hDlg, IDC_DESC, WM_GETTEXTLENGTH, 0, 0);
            temp.Resize(len + 1);
            SendDlgItemMessage(hDlg, IDC_DESC, WM_GETTEXT, len + 1, (LPARAM)temp.data());
            desc = temp;
            EndDialog(hDlg, TRUE);
        }
            return TRUE;
        }
        break;

    case WM_NOTIFY:
        switch (LOWORD(wParam))
        {
        case IDC_URL_LIST:
            NMHDR *pnmh = (LPNMHDR)lParam;
            int code = pnmh->code;
            switch (code)
            {
            case NM_CLICK:
            case NM_SETFOCUS:
                int numsel = ListView_GetSelectedCount(hlist);
                EnableWindow(GetDlgItem(hDlg, IDC_DELETE), numsel > 0);
                for (int i = 0; i < ListView_GetItemCount(hlist); i++)
                {
                    LV_ITEM item;
                    item.mask = LVIF_STATE | LVIF_PARAM;
                    item.iSubItem = 0;
                    item.state = 0;
                    item.stateMask = LVIS_FOCUSED;
                    item.iItem = i;
                    ListView_GetItem(hlist, &item);
                    if (item.state & LVIS_FOCUSED)
                    {
                        SendMessage(GetDlgItem(hDlg, IDC_URL), WM_SETTEXT, 0,
                                    (LPARAM)((Bookmark *)item.lParam)->url.data());
                        SendMessage(GetDlgItem(hDlg, IDC_DESC), WM_SETTEXT, 0,
                                    (LPARAM)((Bookmark *)item.lParam)->desc.data());
                    }
                }
                return TRUE;
            } // pnmh->code switch
            break;
        }
        break;

    } // msg switch
    return FALSE;
}

int
GetBookmarkURL(Interface *ip, TSTR *u, TSTR *c, TSTR *d)
{
    LoadBookmarks(ip, NULL, NULL);

    if (c && c->Length() > 0)
        url = *u + _T("#") + *c;
    else
        url = *u;

    if (d)
        desc = *d;
    else
        desc = _T("");

    if (DialogBoxParam(hInstance, MAKEINTRESOURCE(IDD_URL_BOOKMARKS),
                       GetActiveWindow(), URLBookmarkDlgProc, (LPARAM)ip))
    {
        SplitURL(url, u, c);
        if (d)
            *d = desc;
        SaveBookmarks(ip);
        return TRUE;
    }

    return FALSE;
}

TCHAR *
ExportIniFilename(Interface *ip)
{
    static TCHAR ini_file[MAX_PATH];
    const TCHAR *maxDir = ip->GetDir(APP_PLUGCFG_DIR);
    _tcscpy(ini_file, maxDir);
    _tcscat(ini_file, _T("\\"));
    _tcscat(ini_file, VRMBLIO_INI_FILE);
    return ini_file;
}
