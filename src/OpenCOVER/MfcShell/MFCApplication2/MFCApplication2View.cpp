
// MFCApplication2View.cpp : implementation of the CMFCApplication2View class
//

#include "stdafx.h"
// SHARED_HANDLERS can be defined in an ATL project implementing preview, thumbnail
// and search filter handlers and allows sharing of document code with that project.
#ifndef SHARED_HANDLERS
#include "MFCApplication2.h"
#endif

#include "MFCApplication2Doc.h"
#include "MFCApplication2View.h"

#include "MainFrm.h"

#include <cover/coVRPluginSupport.h>
#include <config/coConfigConstants.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CMFCApplication2View

IMPLEMENT_DYNCREATE(CMFCApplication2View, CView)

BEGIN_MESSAGE_MAP(CMFCApplication2View, CView)
	// Standard printing commands
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CView::OnFilePrintPreview)
	ON_COMMAND(IDC_START_OPENCOVER, &CMFCApplication2View::OnStartOpencover)

END_MESSAGE_MAP()

// CMFCApplication2View construction/destruction

CMFCApplication2View::CMFCApplication2View()
{
	// TODO: add construction code here

}

CMFCApplication2View::~CMFCApplication2View()
{
}

BOOL CMFCApplication2View::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	return CView::PreCreateWindow(cs);
}

// CMFCApplication2View drawing

void CMFCApplication2View::OnDraw(CDC* /*pDC*/)
{
	CMFCApplication2Doc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	// TODO: add draw code for native data here
}


// CMFCApplication2View printing

BOOL CMFCApplication2View::OnPreparePrinting(CPrintInfo* pInfo)
{
	// default preparation
	return DoPreparePrinting(pInfo);
}

void CMFCApplication2View::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add extra initialization before printing
}

void CMFCApplication2View::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add cleanup after printing
}


// CMFCApplication2View diagnostics

#ifdef _DEBUG
void CMFCApplication2View::AssertValid() const
{
	CView::AssertValid();
}

void CMFCApplication2View::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CMFCApplication2Doc* CMFCApplication2View::GetDocument() const // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CMFCApplication2Doc)));
	return (CMFCApplication2Doc*)m_pDocument;
}
#endif //_DEBUG


// CMFCApplication2View message handlers

UINT CMFCApplication2View::ThreadOpenCOVER(LPVOID param)
{

	CMFCApplication2View *mainView = (CMFCApplication2View*)param;

	mainView->Renderer->run();


	return true;
}
void CMFCApplication2View::OnStartOpencover()
{
	CMainFrame *pFrame = (CMainFrame *)AfxGetMainWnd();

	covise::coConfigConstants::setRank(0);

	coverView = new opencover::ui::CMFCView(pFrame->GetMenu());

	Renderer = new opencover::OpenCOVER(this->m_hWnd);
	Renderer->init();
	opencover::cover->ui->addView(coverView);

	CWinThread * coverThread = AfxBeginThread(ThreadOpenCOVER, this);
}



BOOL CMFCApplication2View::OnCommand(WPARAM wParam, LPARAM lParam)
{
	if (wParam >= 5000 && wParam < 8000 ) {
		CString str;
		str.Format(_T(" %d button event;", wParam - 25000 + 1));
		AfxMessageBox(str);
	}

	return CView::OnCommand(wParam, lParam);
}
