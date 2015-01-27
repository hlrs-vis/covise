      SUBROUTINE LBLC (NDXC, LB, IC, LC)

      integer ndxc, ic(ndxc), lc(2*ndxc), lblf, iv(6)
      external lb
      integer d(6,6)	! (num.lato/faccia, imin/imax/jmin/jmax/kmin/kmax)
      data d /0,0,0,1,0,1,
     .        1,1,0,1,0,1,
     .        0,1,0,0,0,1,
     .        0,1,1,1,0,1,
     .        0,1,0,1,0,0,
     .        0,1,0,1,1,1/

      do il = 1, ndxc*2				! lato/faccia
          do jl = 1, ndxc			! ijk
              l2 = jl * 2
              l1 = l2 - 1
              iv(l1) = ic(jl) + d(l1,il)
              iv(l2) = ic(jl) + d(l2,il)
          end do
          lc(il) = lblf( ndxc, lb, iv )
      end do

      return
      end



      INTEGER FUNCTION LBLF (NDXC, LB, IV)
CC
CC Loading of boundary condition for face (edge) searching
CC through the block boundary conditions
CC
      integer ndxc, iv(2*ndxc)
      integer lb
CC    LB: integer function (patchIndex,boudaryCondition,ijkMinMaxIndices)
      external lb
      integer lv(6), val
      logical iok, jok, kok, case2d, case3d
      iok = lv(1) .le. iv(1) .and. iv(2) .le. lv(2)
      jok = lv(3) .le. iv(3) .and. iv(4) .le. lv(4)
      kok = lv(5) .le. iv(5) .and. iv(6) .le. lv(6)
      case2d = ndxc .eq. 2
      case3d = ndxc .eq. 3

      lblf = 0	! default: no special boundary condition
      ip = 0
10    ip = ip + 1
      if ( lb(ip,val,lv) .ne. 0 ) then
          if ( case3d ) then
              if ( iok .and. jok .and. kok ) then
                  lblf = val
                  return
              endif
          else if ( case2d ) then
              if ( iok .and. jok ) then
                  lblf = val
                  return
              endif
          else
              if ( iok ) then
                  lblf = val
                  return
              endif
          endif
          goto 10
      endif

      return
      end
