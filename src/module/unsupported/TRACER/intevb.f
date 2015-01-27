      subroutine intevb (nv, ndxc, xc, ic, vb, vx, irtn)

C *** BUGS
C     - implemented ndxc=3 only
C     - WARNING: declare VC(NV,2**NDXC)

C *** SPECIFICATIONS
      dimension xc(ndxc), ic(ndxc), vx(nv)
      dimension vc(3,8)
      external vb

C *** VARIABLES AT CELL NODES
      ii = ic(1)
      jj = ic(2)
      kk = ic(3)
      do iv=1,nv
          vc(iv , 1) = vb(iv ,ii  ,jj  ,kk  )
          vc(iv , 2) = vb(iv ,ii+1,jj  ,kk  )
          vc(iv , 3) = vb(iv ,ii  ,jj+1,kk  )
          vc(iv , 4) = vb(iv ,ii+1,jj+1,kk  )
          vc(iv , 5) = vb(iv ,ii  ,jj  ,kk+1)
          vc(iv , 6) = vb(iv ,ii+1,jj  ,kk+1)
          vc(iv , 7) = vb(iv ,ii  ,jj+1,kk+1)
          vc(iv , 8) = vb(iv ,ii+1,jj+1,kk+1)
      enddo

C *** CELL INTERPOLATION
      call intevc (nv,ndxc,xc,vc,vx,irtn)

      return
      end
