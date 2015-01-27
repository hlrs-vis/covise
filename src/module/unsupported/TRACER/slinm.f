      integer function slinm
     %      (ndx, ndxc, istart,xp, xcp, icp, ibp,ivel,iboc,
     %       istep, dl, npmax, delta, nismax, np, irtn)

C *** BUGS
C     - ibocc still to do
C     - implemented ndx=ndxc=3 only
C     - missing input check

C *** MEMO
C     ISTART : input type of given coord. for the starting point:
C                 1  computational coord.
C                 2  cartesian coord.
C         otherwise  computational and cartesian coord.

C *** SPECIFICATIONS
        dimension xp(ndx,npmax),xcp(ndxc,npmax),
     %            icp(ndxc,npmax),ibp(npmax),ivel(ndx)
        external gb, vb, lb
        dimension ngb(3),xc1(3),ic1(3),ic2(3),xc2(3),x2(3)

C *** STARTING POINT INITIALIZATION
      np = 1

C *** STARTING POINT GIVEN IN COMPUTATIONAL COORDINATES ONLY
      if (istart.eq.1) then
          call geob(ibp(1),ndx1,ndxc1,ngb,gb,iok)
          if (iok.ne.0) goto 996

*//* 28.09.92 Check for out-of-range start point included

C CHECK FOR OUT-OF-RANGE START POINT (PERHAPS CHECK SHOULD BE BEFORE SLINM CALLED) 

          if (icp(1,1).lt.1 .or. icp(1,1).ge.ngb(1) .or. 
     &        icp(2,1).lt.1 .or. icp(2,1).ge.ngb(2) .or.  
     &        icp(3,1).lt.1 .or. icp(3,1).ge.ngb(3) ) goto 996

*//* End of 28.09.92 addition

          call intevb (ndx,ndxc,xcp(1,1),icp(1,1),gb,xp(1,1),iok)
          if (iok.ne.0) goto 997

C *** STARTING POINT GIVEN IN CARTESIAN COORDINATES ONLY
      else if (istart.eq.2) then
          ithru = ndxc+1
          ibp(1)=1
          do idxc=1,ndxc
              icp(idxc,1)=1
          enddo
          call x2xm(ndx,ndxc,ithru,xp(1,1),ibp(1),icp(1,1),xcp(1,1),iok)
          if (iok.ne.0) goto 996
      endif

C *** STREAMLINE INITIALIZATION
      np = 2
      npmax1 = npmax-1
      nopath = 1
      ithru = ndxc

*//* 17.09.92  ex defined and used as in comment below

C If the start point is on a cell boundary we may have problems if
C the flow direction is out of the cell.  To avoid these move the
C start point away from the cell boundary.

      ex = 1.e-6
      do idxc=1,ndxc

*//* 17.09.92 Modified to move point slightly away from cell boundary
          if (xcp(idxc,1) .le. 0.) xcp(idxc,1) = ex 
          if (xcp(idxc,1) .ge. 1.) xcp(idxc,1) = 1-ex 
*//* ***
          xc1(idxc)=xcp(idxc,1)
          ic1(idxc)=icp(idxc,1)
      enddo

      ib=ibp(1)
      dl1 = 0.

C *** BLOCK LOOP:
C     GEOMETRY, VELOCITY (AND BOUNDARY CONDITIONS) OF THE CURRENT BLOCK
 100  continue
      if (np.gt.npmax) goto 990
      call geob (ib,ndx,ndxc,ngb,gb,iok)
      if (iok.ne.0) goto 997
      call limb( ib, ndxc, lb, iok )
      if (iok.ne.0) goto 997
      call varb (ib,ndx,ivel,ngb,vb,iok)
      if (iok.ne.0) goto 997

C *** TRACE THE PARTICLE PATH THROUGH THE CURRENT BLOCK
      call slinb (ndx, ndxc, ngb, gb, vb, lb,
     %            xc1, ic1, istep, dl, dl1,
     %            npmax1, delta, nismax,
     %            np1, xp(1,np), xcp(1,np), icp(1,np),
     %            xc2, ic2, dl2, islinb)

C *** STORAGE OF PATH POINTS
      if (np1.ne.0) then
          do ip = np,np+np1-1
              ibp(ip) = ib
          enddo
          np = np+np1
          npmax1 = npmax-np1
      endif

      np = np - 1
      return

C *** INTERFACE POINT
      if (islinb.eq.0) then
          nopath = 0
          ithru = ndxc

*//* 17.09.92   Extend the path slightly so that it will enter 
*               another block

          ex=1.e-4
          do idxc=1,ndxc
             if(xc2(idxc) .eq. 0.) then
                xc2(idxc) = -ex
             elseif (xc2(idxc) .eq. 1.) then
                xc2(idxc) = 1. + ex
             endif
          enddo 
*//*
         call intevb (ndx,ndxc,xc2,ic2,gb,x2,iok)
          if (iok.ne.0) goto 997
          dl1=dl2
      endif

C *** NEXT BLOCK
      if (islinb.le.1) then
          if (ithru.gt.0) then
              ib=1
              do idxc=1,ndxc
                  ic1(idxc)=1
              enddo
          endif
          call x2xm(ndx,ndxc,ithru,x2,ib,ic1,xc1,iok)
          ithru = -ndxc
          if (iok.eq.0) then
              goto 100
          else
              goto 990
          endif
      endif

C *** ERROR FROM SLINB
C     IRTN = 2 : NISMAX EXCEEDED
C          = 3 : NPMAX EXCEEDED
C          = 4 : DELTA TOLERANCE NOT ACHIEVED
C          = 5 : POINT OF ZERO VELOCITY REACHED
      np = np - 1
      irtn = islinb
      return

C *** EXIT
C     IRTN = 0 : PATH CALCULATED SUCCESSFULLY
C            1 : ZERO LENGTH PATH
 990  continue
      np = np - 1
      irtn = nopath
      return

C *** EXIT
C     IRTN = 6 : STARTING POINT NOT FOUND
 996  continue
      np = np - 1
      irtn = 6
      return

C *** EXIT
C     IRTN = 7 : ERROR IN GEOB, VARB, INTEVB
 997  continue
      np = np - 1
      irtn = 7
      return

      end

