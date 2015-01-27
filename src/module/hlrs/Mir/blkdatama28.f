      blockdata block1

      double precision eps, rmin, resid, tol, themax, big, dxmax,  &
      errmax, dres, cgce
      logical lblock, grow, abort1, abort2, lbig
      common /ma28ed/ lp, mp, lblock, grow
      common /ma28fd/ eps, rmin, resid, irncp, icncp, minirn, minicn,  &
      irank, abort1, abort2
!     common /ma28gd/ idisp(2)
      common /ma28hd/ tol, themax, big, dxmax, errmax, dres, cgce,  &
      ndrop, maxit, noiter, nsrch, istart, lbig
      data eps /1.0d-4/, tol /0.0d0/, cgce /0.5d0/
      data maxit /16/
      data lp /6/, mp /6/, nsrch /32768/, istart /0/
      data lblock /.true./, grow /.true./, lbig /.false./
      data abort1 /.true./, abort2 /.true./
      end

      blockdata  block2
      double precision eps, rmin, tol, big
      logical abort1, abort2, abort3, lbig
      common /ma30ed/ lp, abort1, abort2, abort3
      common /ma30gd/ eps, rmin
      common /ma30id/ tol, big, ndrop, nsrch, lbig
      data eps /1.0d-4/, tol /0.0d0/, big /0.0d0/
      data lp /6/, nsrch /32768/
      data lbig /.false./
      data abort1 /.true./, abort2 /.true./, abort3 /.false./
      end

      blockdata block3
      logical abort
      common /mc23bd/ lp,numnz,num,large,abort
      data lp/6/,abort/.false./
      end
