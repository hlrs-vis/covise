      subroutine x2xm(ndx,ndxc,ithru,x,ib,ic,xc,irtn)
      dimension x(ndx),ic(ndxc),xc(ndxc)

C *** BUGS
C     - implemented ndx=ndxc=3 with ithru=-4|-3|+3|+4 only

C *** SPECIFICATIONS
      dimension gc(3,8),ngb(3)
      dimension xc2(2)
      external gb
      
C *** INITIALIZE THE CURRENT CELL
      i=ic(1)
      if (ndxc.ge.2) j=ic(2)
      if (ndxc.ge.3) k=ic(3)

C *** FLAG FOR SKIPPING THE CURRENT CELL (-1|+1 = SKIP|EVALUATE)
      iskip = isign (1, ithru)

C *** BLOCK LOOP
 10   continue

C *** LOAD THE GEOMETRY OF THE CURRENT BLOCK
      call geob(ib,ndx1,ndxc1,ngb,gb,iok)

C *** ERROR LOADING BLOCK GEOMETRY
      if(iok.ne.0) then
          irtn = 1
          return
      endif
      if ((ndx1.ne.ndx).or.(ndxc1.ne.ndxc)) then
          ib = ib+1
          do idxc = 1,ndxc
              ic(idxc) = 1
          enddo
          iskip = +1
          goto 10
      endif
        
C *** DIMENSIONS NI,NJ,NK OF THE CURRENT BLOCK
      ni=ngb(1)
      if (ndxc.ge.2) nj=ngb(2)
      if (ndxc.ge.3) nk=ngb(3)

C *** SEARCH THROUGH CELLS IN A VOLUME
      if ((ndx.eq.3).and.(ndxc.eq.3).and.(abs(ithru).eq.4)) then

C     *** SKIP THE CURRENT CELL, IF REQUIRED
          if (iskip.eq.-1) then
              iskip=+1
              goto 190
          endif

C     *** CELL LOOP
 110      continue

C     *** LOAD THE GEOMETRY OF THE CURRENT CELL
          do 120 ix=1,ndx
              gc(ix,1)=gb(ix,i  ,j  ,k)
              gc(ix,2)=gb(ix,i+1,j  ,k)
              gc(ix,3)=gb(ix,i  ,j+1,k)
              gc(ix,4)=gb(ix,i+1,j+1,k)
              gc(ix,5)=gb(ix,i  ,j  ,k+1)
              gc(ix,6)=gb(ix,i+1,j  ,k+1)
              gc(ix,7)=gb(ix,i  ,j+1,k+1)
              gc(ix,8)=gb(ix,i+1,j+1,k+1)
 120      continue

C     *** TRY THE CURRENT CELL
          call x2xc(ndx,ndxc,x,gc,xc,iok)

C     *** SUCCESSFUL EXIT
          if(iok.eq.0) then
              ic(1)=i
              if (ndxc.ge.2) ic(2)=j
              if (ndxc.ge.3) ic(3)=k
              irtn=0
              return
          endif

C     *** SET THE NEXT CELL
 190      continue
          if (i.ne.(ni-1)) then
              i=i+1
              goto 110       
          else if (j.ne.(nj-1)) then
              i=1
              j=j+1
              goto 110
          else if (k.ne.(nk-1)) then
              i=1
              j=1
              k=k+1
              goto 110
          else
              i=1
              j=1
              k=1
              ib=ib+1
              goto 10
          endif
      endif

C *** SEARCH THROUGH BOUNDARY CELLS IN A VOLUME
      if ((ndx.eq.3).and.(ndxc.eq.3).and.(abs(ithru).eq.3)) then

C     *** SKIP THE CURRENT CELL, IF REQUIRED
          if (iskip.eq.-1) then
              iskip=+1
              goto 290
          endif

C     *** BOUNDARY CELL LOOP
 210      continue 

C     *** LOAD THE GEOMETRY OF THE CURRENT CELL
          do 220 ix=1,ndx
              gc(ix,1)=gb(ix,i  ,j  ,k)
              gc(ix,2)=gb(ix,i+1,j  ,k)
              gc(ix,3)=gb(ix,i  ,j+1,k)
              gc(ix,4)=gb(ix,i+1,j+1,k)
              gc(ix,5)=gb(ix,i  ,j  ,k+1)
              gc(ix,6)=gb(ix,i+1,j  ,k+1)
              gc(ix,7)=gb(ix,i  ,j+1,k+1)
              gc(ix,8)=gb(ix,i+1,j+1,k+1)
 220          continue

C     *** TRY THE CURRENT CELL
          call x2xc(ndx,ndxc,x,gc,xc,iok)

C     *** SUCCESSFUL EXIT
          if(iok.eq.0) then
              ic(1)=i
              if (ndxc.ge.2) ic(2)=j
              if (ndxc.ge.3) ic(3)=k
              irtn=0
              return
          endif

C     *** SET THE NEXT BOUNDARY CELL
 290      continue
          if ((k.eq.1).or.(k.eq.(nk-1))) then
              if (i.ne.(ni-1)) then
                  i=i+1
                  goto 210
              else if (j.ne.(nj-1)) then
                  i=1
                  j=j+1
                  goto 210
              else if (k.ne.(nk-1)) then
                  i=1
                  j=1
                  k=k+1
                  goto 210
              else
                  i=1
                  j=1
                  k=1
                  ib=ib+1
                  goto 10
              endif
          else if ((j.eq.1).or.(j.eq.(nj-1))) then
              if (i.ne.(ni-1)) then
                  i=i+1
                  goto 210
              else if (j.ne.(nj-1)) then
                  i=1
                  j=j+1
                  goto 210
              else
                  i=1
                  j=1
                  k=k+1
                  goto 210
              endif
          else
              if (i.ne.(ni-1)) then
                  i=ni-1
                  goto 210
              else
                  i=1
                  j=j+1
                  goto 210
              endif
          endif
      endif
      end
