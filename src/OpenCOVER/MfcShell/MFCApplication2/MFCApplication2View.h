
// MFCApplication2View.h : interface of the CMFCApplication2View class
//

#pragma once
#include <cover/OpenCOVER.h>
#include "MFCView.h"

class CMFCApplication2View : public CView
{
protected: // create from serialization only
	CMFCApplication2View();
	DECLARE_DYNCREATE(CMFCApplication2View)

// Attributes
public:
	CMFCApplication2Doc* GetDocument() const;

// Operations
public:
	static UINT ThreadOpenCOVER(LPVOID param);
	opencover::OpenCOVER *Renderer;
	opencover::ui::CMFCView *coverView;
	virtual BOOL OnCommand(WPARAM wParam, LPARAM lParam);
	

// Overrides
public:
	virtual void OnDraw(CDC* pDC);  // overridden to draw this view
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);

// Implementation
public:
	virtual ~CMFCApplication2View();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnStartOpencover();
};

#ifndef _DEBUG  // debug version in MFCApplication2View.cpp
inline CMFCApplication2Doc* CMFCApplication2View::GetDocument() const
   { return reinterpret_cast<CMFCApplication2Doc*>(m_pDocument); }
#endif

