      subroutine slinb
     %                 (ndx, ndxc, ngb, gb, vb, lb,
     %                  xc1, ic1, istep, dl, dl1,
     %                  npmax, delta, nismax,
     %                  np, xp, xcp, icp,
     %                  xc2, ic2, dl2, irtn)

C *** BUGS
C     - ibocc still to do
C     - implemented ndx=ndxc=3 only
C     - missing input check

c *** SPECIFICATIONS
      dimension ngb(ndxc), xc1(ndxc), ic1(ndxc),
     %          xp(ndxc,npmax), xcp(ndxc,npmax),
     %          icp(ndxc,npmax), xc2(ndxc), ic2(ndxc) 
      external gb, vb, lb
      dimension gc(3,8), vc(3,8), ibocc(6)

*//* 28.09.92 DELETED  dimension iclip(3), lposs(3,8), xposs(3,8)


c *** INITIALIZE DEFAULT IBOCC
C     do if=1,2*ndxc
C         ibocc(if) = 0
C     enddo

c *** STREAMLINE INITIALIZATION
      np = 1
      npmax1 = npmax
      nopath = 1
      ni = ngb(1)
      nj = ngb(2)
      nk = ngb(3)

*//* 28.09.92 iposs and nposs not now used
*      nposs = 1
*      iposs = 1

*//* 28.09.92 Initialisation of ii,jj,kk moved out of cell loop

      ii = ic1(1)
      jj = ic1(2)
      kk = ic1(3)

C *** CELL LOOP:
C     GEOMETRY, VELOCITY (AND BOUNDARY CONDITIONS) OF THE CURRENT CELL

 100  continue

      if (np.gt.npmax) goto 990

*//* 28.09.92 Initialisation of ii,jj,kk moved out of cell loop

      do ix=1,ndx
          gc(ix , 1) = gb(ix ,ii  ,jj  ,kk  )
          gc(ix , 2) = gb(ix ,ii+1,jj  ,kk  )
          gc(ix , 3) = gb(ix ,ii  ,jj+1,kk  )
          gc(ix , 4) = gb(ix ,ii+1,jj+1,kk  )
          gc(ix , 5) = gb(ix ,ii  ,jj  ,kk+1)
          gc(ix , 6) = gb(ix ,ii+1,jj  ,kk+1)
          gc(ix , 7) = gb(ix ,ii  ,jj+1,kk+1)
          gc(ix , 8) = gb(ix ,ii+1,jj+1,kk+1)
          vc(ix , 1) = vb(ix ,ii  ,jj  ,kk  )
          vc(ix , 2) = vb(ix ,ii+1,jj  ,kk  )
          vc(ix , 3) = vb(ix ,ii  ,jj+1,kk  )
          vc(ix , 4) = vb(ix ,ii+1,jj+1,kk  )
          vc(ix , 5) = vb(ix ,ii  ,jj  ,kk+1)
          vc(ix , 6) = vb(ix ,ii+1,jj  ,kk+1)
          vc(ix , 7) = vb(ix ,ii,jj+1  ,kk+1)
          vc(ix , 8) = vb(ix ,ii+1,jj+1,kk+1)
      enddo

C *** Loads boundary conditions
      iok = lblc( ndxc, lb, ic1, ibocc )

C *** TRACE THE PARTICLE PATH THROUGH THE CURRENT CELL
      call slinc(ndx,ndxc,gc,vc,ibocc,xc1,istep,dl,dl1,
     %           npmax1,delta,nismax,np1,
     %           xp(1,np),xcp(1,np),xc2,dl2,islinc)
*//* 28.09.92 iclip parameter removed

C *** STORAGE OF PATH POINTS
      if (np1.ne.0) then
c         do l=1,6
c             if ( ibocc(l) .ne. 0 ) call drawcell(gc,ibocc)
c         end do
          do ip = np,np+np1-1
              icp(1,ip) = ii
              icp(2,ip) = jj
              icp(3,ip) = kk
          enddo
          np = np+np1
          npmax1 = npmax-np1
      endif

C *** INTERFACE POINT:

*//* 28.09.92 Section up to *** ERROR FROM SLINC changed completely

C FIND NEXT CELL IF WITHIN THIS BLOCK

      if (islinc .le. 1) then

	 if(islinc .eq. 0) nopath = 0
	 ic1(1) = ii	   
	 ic1(2) = jj	   
	 ic1(3) = kk	

	 p = xc2(1)   
	 q = xc2(2)   
	 r = xc2(3) 
  
	 if(p .eq. 0) then
            if (ii .eq. 1) goto 990
            ii = ii - 1 
	     p = 1 
	 elseif(p .eq. 1) then
            if (ii .eq. ni-1) goto 990
            ii = ii + 1 
       	     p = 0 
	 endif

	 if(q .eq. 0) then
            if (jj .eq. 1) goto 990
            jj = jj - 1 
	     q = 1 
	 elseif(q .eq. 1) then
            if (jj .eq. nj-1) goto 990
            jj = jj + 1 
       	     q = 0 
	 endif

	 if(r .eq. 0) then
            if (kk .eq. 1) goto 990
            kk = kk - 1 
	     r = 1 
	 elseif(r .eq. 1) then
            if (kk .eq. nk-1) goto 990
            kk = kk + 1 
       	     r = 0 
	 endif

	 xc1(1) = p
	 xc1(2) = q
	 xc1(3) = r

	 dl1 = dl2

	 goto 100

      endif

C *** ERROR FROM SLINC
C     IRTN = 2 : NISMAX EXCEEDED
C          = 3 : NPMAX EXCEEDED
C          = 4 : DELTA TOLERANCE NOT ACHIEVED
C          = 5 : POINT OF ZERO VELOCITY REACHED 
      np = np-1
      irtn = islinc
      return

C *** IRTN = 0 : PATH CALCULATED SUCCESSFULLY
C            1 : ZERO LENGTH PATH
 990  continue
      do idxc = 1,ndxc 
          ic2(idxc) = ic1(idxc)
      enddo
      np = np-1
      irtn = nopath
      return
      end
