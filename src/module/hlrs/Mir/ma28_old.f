      module mod_ma28

      use declaration
      use visco_declaration

      contains

      subroutine ma28c(dim,matrix,t,pivot,info)

      use arithmetic
      use declaration

      integer :: dim,pivot(dim)
      type(sparse_matrix) :: matrix
      real(double),pointer,save :: a(:)
      real(double),pointer :: a2(:)
      real(double) :: t(dim)
      integer,pointer :: irn(:),irn2(:),icn2(:),perminv(:)
      integer,pointer,save :: icn(:)
      integer :: i,j,iflag,info
      real(double),pointer,save :: w(:)
      real(double)         :: u
      integer,pointer :: iw(:,:)
      integer,pointer,save :: ikeep(:,:)
      integer,save :: intflag,k,licn,lirn,nz,mtype,n


      info=0

      if (intflag .ne. 12345) then
        intflag=12345
        write(*,*) 'changed intflag'
        licn=size(matrix%container%element)*10
        lirn=size(matrix%container%element)*10
        allocate(a(licn))
        allocate(irn(lirn))
        allocate(irn2(lirn))
        allocate(icn(licn))
        allocate(w(dim))
        allocate(iw(dim,8))
        allocate(ikeep(dim,5))
        allocate(perminv(matrix%maxrow))

        n=dim
        nz=size(matrix%container%element)
        mtype=1
        iflag=1
        u=1.0
        w=0.0
        ikeep=0
        iw=0
        k=0

        do i=1,size(matrix%container%element)
          a(i)=matrix%container%element(i)
          icn(i)=matrix%container%column_number(i)
        enddo

        k=0
        do i=1,matrix%maxng
          do j=1,matrix%container%ng(i)
            k=k+1
            irn2(k)=j
          enddo
        enddo

        do i=1,dim
          perminv(matrix%container%permutation(i))=i
        enddo

        do i=1,nz
          irn(i)=perminv(irn2(i))
        enddo

        allocate(a2(licn))
        allocate(icn2(licn))
        a2=a
        icn2=icn
        irn2=irn
        iflag=-100

        do while ((iflag==-3) .or. (iflag==-4) .or. (iflag== -100)) 
 
          call ma28ad(n,nz,a,licn,irn,lirn,icn,u,ikeep,iw,w,iflag)
 
          if (iflag == -3) then
            lirn=lirn*2
            deallocate(irn)
            allocate(irn(lirn))
            do i=1,nz 
              irn(i)=irn2(i)
              a(i)=a2(i)
              icn(i)=icn2(i)
            enddo
          endif
          if (iflag == -4) then
            licn=licn*2
            deallocate(icn)
            allocate(icn(licn))
            deallocate(a)
            allocate(a(licn))
            do i=1,nz 
              a(i)=a2(i)
              icn(i)=icn2(i)
              irn(i)=irn2(i)
            enddo
          endif
        end do
      
        deallocate(irn)
        deallocate(iw)
        deallocate(a2)
        deallocate(icn2)
        deallocate(irn2)
      endif

      call ma28cd(n,a,licn,icn,ikeep,t,w,mtype)
      
      end subroutine ma28c


      subroutine ma28b(dim,matrix,T,pivot,info)

      use arithmetic
      use declaration

      integer                            :: dim,pivot(dim),info 
      type(sparse_matrix)                ::matrix 
      real(double),dimension(dim)        :: T
      integer, dimension(matrix%maxrow**2*10) :: icn,irn,irn3
      real(double), dimension(matrix%maxrow**2*10):: a,a2
      real(double) :: u,w(matrix%maxrow),rhs(matrix%maxrow)
      integer :: n,nz,licn,lirn,iflag,mtype
      integer :: ikeep(matrix%maxrow,5),iw(matrix%maxrow,8)
      integer :: i,j,k,perminv(dim)
      real(double), dimension(dim)   :: T2


      info=0
      n=matrix%maxrow
      nz=size(matrix%container%element)
      licn=matrix%maxrow**2
      lirn=matrix%maxrow**2
      iflag=1
      u=1.0
      w=0.0
      ikeep=0
      iw=0
      mytype=1

      do i=1,size(matrix%container%element)
       a(i)=dble(matrix%container%element(i))
       icn(i)=matrix%container%column_number(i)
      enddo

      k=0
      do i=1,matrix%maxng
       do j=1,matrix%container%ng(i)
        k=k+1
        irn(k)=j
       enddo
      enddo

      do i=1,dim
       perminv(matrix%container%permutation(i))=i
       rhs(i)=T(i)
      enddo
      
      do i=1,nz
       irn3(i)=perminv(irn(i))
      enddo

      do i=1,nz
       print*,a(i),irn3(i),icn(i),'ma28b'
      enddo
      do i=1,dim
       print*,rhs(i),'rhs'
      enddo


      call ma28ad(n,nz,a,licn,irn3,lirn,icn,u,ikeep,iw,w,iflag)
      call ma28cd(n,a,licn,icn,ikeep,rhs,w,mtype)


      do i=1,dim
       t(i)=rhs(i)
      enddo

      end subroutine ma28b








      subroutine ma28(dim,AC,T,pivot,info)

      integer :: dim,pivot(dim)
      real(double) :: AC(dim*dim)
      real(double),pointer :: AC2(:)
      real(double) :: T(dim)
      integer,pointer :: irn(:),irn2(:),icn2(:)
      integer,pointer,save :: icn(:)
      integer :: i,j,iflag,info
      real(double),pointer,save :: w(:)
      real(double)         :: u
      integer,pointer :: iw(:,:)
      integer,pointer,save :: ikeep(:,:)
      integer,save :: intflag,k,licn,lirn,nz,mtype,n


      info=0

      if (intflag .ne. 12345) then
       intflag=12345

       licn=dim*dim
       lirn=dim*dim
       allocate(irn(lirn))
       allocate(icn(licn))
       allocate(w(dim))
       allocate(iw(dim,8))
       allocate(ikeep(dim,5))

       n=dim
       mtype=1
       iflag=1
       u=1.0
       w=0.0
       ikeep=0
       iw=0
       k=0

       do j=1,dim
        do i=1,dim
         if (AC(i+(j-1)*dim).ne.0.0) then
          k=k+1
          AC(k)=AC(i+(j-1)*dim)
          irn(k)=i
          icn(k)=j
         endif
        enddo
       enddo

       nz=k
       
       allocate(AC2(licn))
       allocate(icn2(licn))
       allocate(irn2(lirn))
       AC2=AC
       icn2=icn
       irn2=irn
       iflag=-4

       do while ((iflag==-3) .or. (iflag==-4)) 
 
      do i=1,nz
       print*,AC(i),irn(i),icn(i),'ma28'
      enddo
      do i=1,dim
       print*,t(i)
      enddo

        call ma28ad(n,nz,AC,licn,irn,lirn,icn,u,ikeep,iw,w,iflag)
 
        if (iflag == -3) then
         lirn=lirn*2
         deallocate(irn)
         allocate(irn(licn))
         do i=1,nz 
          irn(i)=irn2(i)
          AC(i)=AC2(i)
          icn(i)=icn2(i)
         enddo
        endif
        if (iflag == -4) then
         licn=licn*2
!         deallocate(AC)
         deallocate(icn)
!         allocate(AC(licn))
         allocate(icn(licn))
         do i=1,nz 
          AC(i)=AC2(i)
          icn(i)=icn2(i)
          irn(i)=irn2(i)
         enddo
        endif

       end do
      
       deallocate(irn)
       deallocate(iw)
       deallocate(AC2)
       deallocate(icn2)
       deallocate(irn2)
      endif

      call ma28cd(n,AC,licn,icn,ikeep,T,w,mtype)

             
      end subroutine ma28


!######date   01 jan 1984     copyright ukaea, harwell.
!######alias ma28ad ma28bd ma28cd
!###### calls   ma30    mc20    mc22    mc23    mc24
      subroutine ma28ad(n, nz, a, licn, irn, lirn, icn, u, ikeep, iw, w,  &
      iflag)
! this subroutine performs the lu factorization of a.
!
! the parameters are as follows.....
! n     order of matrix  not altered by subroutine.
! nz    number of non-zeros in input matrix  not altered by subroutine.
! a is a  real array  length licn.  holds non-zeros of matrix on entry
!     and non-zeros of factors on exit.  reordered by mc20a/ad and
!     mc23a/ad and altered by ma30a/ad.
! licn  integer  length of arrays a and icn.  not altered by subroutine.
! irn   integer array of length lirn.  holds row indices on input.
!     used as workspace by ma30a/ad to hold column orientation of
!     matrix.
! lirn  integer  length of array irn. not altered by the subroutine.
! icn   integer array of length licn.  holds column indices on entry
!     and column indices of decomposed matrix on exit. reordered by
!     mc20a/ad and mc23a/ad and altered by ma30a/ad.
! u     real variable  set by user to control bias towards numeri! or
!     sparsity pivoting.  u=1.0 gives partial pivoting while u=0. does
!     not check multipliers at all.  values of u greater than one are
!     treated as one while negative values are treated as zero.  not
!     altered by subroutine.
! ikeep  integer array of length 5*n  used as workspace by ma28a/ad
!     (see later comments).  it is not required to be set on entry
!     and, on exit, it contains information about the decomposition.
!     it should be preserved between this call and subsequent calls
!     to ma28b/bd or ma28c/cd.
!     ikeep(i,1),i=1,n  holds the total length of the part of row i
!     in the diagonal block.
!     row ikeep(i,2),i=1,n  of the input matrix is the ith row in
!     pivot order.
!     column ikeep(i,3),i=1,n  of the input matrix is the ith column
!     in pivot order.
!     ikeep(i,4),i=1,n  holds the length of the part of row i in
!     the l part of the l/u decomposition.
!     ikeep(i,5),i=1,n  holds the length of the part of row i in the
!     off-diagonal blocks.  if there is only one diagonal block,
!     ikeep(1,5) will be set to -1.
! iw    integer array of length 8*n.  if the option nsrch.le.n is
!     used, then the length of array iw can be reduced to 7*n.
! w real array  length n.  used by mc24a/ad both as workspace and to
!     return growth estimate in w(1).  the use of this array by ma28a/ad
!     is thus optional depending on common block logical variable grow.
! iflag  integer variable  used as error flag by routine.  a positive
!     or zero value on exit indicates success.  possible negative
!     values are -1 through -14.
!
      integer n, nz, licn, lirn, iflag
      integer irn(lirn), icn(licn), ikeep(n,5), iw(n,8)
      real(double) a(licn), u, w(n)

! common and private variables.
!     common block ma28f/fd is used merely
!     to communicate with common block ma30f/fd  so that the user
!     need not declare this common block in his main program.
! the common block variables are as follows ...
! lp,mp  integer  default value 6 (line printer).  unit number
!     for error messages and duplicate element warning resp.
! nlp,mlp  integer  unit number for messages from ma30a/ad and
!     mc23a/ad resp.  set by ma28a/ad to value of lp.
! lblock  logical  default value true.  if true mc23a/ad is used
!     to first permute the matrix to block lower triangular form.
! grow    logical  default value true.  if true then an estimate
!     of the increase in size of matrix elements during l/u
!     decomposition is given by mc24a/ad.
! eps,rmin,resid  real/real(double) variables not referenced
!     by ma28a/ad.
! irncp,icncp  integer  set to number of compresses on arrays irn and
!     icn/a respectively.
! minirn,minicn  integer  minimum length of arrays irn and icn/a
!     respectively, for success on future runs.
! irank  integer   estimated rank of matrix.
! mirncp,micncp,mirank,mirn,micn integer variables.  used to
!     communicate between ma30f/fd and ma28f/fd values of abovenamed
!     variables with somewhat similar names.
! abort1,abort2  logical variables with default value true.  if false
!     then decomposition will be performed even if the matrix is
!     structurally or numerically singular respectively.
! aborta,abortb  logical variables used to communicate values of
!     abort1 and abort2 to ma30a/ad.
! abort  logical  used to communicate value of abort1 to mc23a/ad.
! abort3  logical variable not referenced by ma28a/ad.
! idisp   integer array  length 2.  used to communicate information
!     on decomposition between this call to ma28a/ad and subsequent
!     calls to ma28b/bd and ma28c/cd.  on exit, idisp(1) and
!     idisp(2) indicate position in arrays a and icn of the
!     first and last elements in the l/u decomposition of the
!     diagonal blocks, respectively.
! numnz  integer  structural rank of matrix.
! num    integer  number of diagonal blocks.
! large  integer  size of largest diagonal block.

! see block data for further comments on common block variables.
! see code for comments on private variables.

      real(double) tol, themax, big, dxmax, errmax, dres, cgce,  &
      tol1, big1, upriv, rmin, eps, resid, zero
      integer idisp(2)
      logical grow, lblock, abort, abort1, abort2, abort3, aborta,  &
      abortb, lbig, lbig1
      common /ma28ed/ lp, mp, lblock, grow
      common /ma28fd/ eps, rmin, resid, irncp, icncp, minirn, minicn,  &
      irank, abort1, abort2
      common /ma28gd/ idisp
      common /ma28hd/ tol, themax, big, dxmax, errmax, dres, cgce,  &
      ndrop, maxit, noiter, nsrch, istart, lbig
      common /ma30id/ tol1, big1, ndrop1, nsrch1, lbig1
      common /ma30ed/ nlp, aborta, abortb, abort3
      common /ma30fd/ mirncp, micncp, mirank, mirn, micn
      common /mc23bd/ mlp, numnz, num, large, abort
      common /lpivot/ lpiv(10),lnpiv(10),mapiv,manpiv,iavpiv,  &
                     ianpiv,kountl

! some  initialization and transfer of information between
!     common blocks (see earlier comments).
      data zero /0.0d0/
      iflag = 0
      aborta = abort1
      abortb = abort2
      abort = abort1
      mlp = lp
      nlp = lp
      tol1 = tol
      lbig1 = lbig
      nsrch1 = nsrch
! upriv private copy of u is used in case it is outside
!     range  zero to one  and  is thus altered by ma30a/ad.
      upriv = u
! simple data check on input variables and array dimensions.
      if (n.gt.0) go to 10
      iflag = -8
      if (lp.ne.0) write (lp,99999) n
      go to 210
   10 if (nz.gt.0) go to 20
      iflag = -9
      if (lp.ne.0) write (lp,99998) nz
      go to 210
   20 if (licn.ge.nz) go to 30
      iflag = -10
      if (lp.ne.0) write (lp,99997) licn
      go to 210
   30 if (lirn.ge.nz) go to 40
      iflag = -11
      if (lp.ne.0) write (lp,99996) lirn
      go to 210

! data check to see if all indices lie between 1 and n.
   40 do 50 i=1,nz
        if (irn(i).gt.0 .and. irn(i).le.n .and. icn(i).gt.0 .and.  &
        icn(i).le.n) go to 50
        if (iflag.eq.0 .and. lp.ne.0) write (lp,99995)
        iflag = -12
        if (lp.ne.0) write (lp,99994) i, a(i), irn(i), icn(i)
   50 continue
      if (iflag.lt.0) go to 220

! sort matrix into row order.
      call mc20ad(n, nz, a, icn, iw, irn, 0)
! part of ikeep is used here as a work-array.  ikeep(i,2) is
!     the last row to have a non-zero in column i.  ikeep(i,3)
!     is the off-set of column i from the start of the row.
      do 60 i=1,n
        ikeep(i,2) = 0
        ikeep(i,1) = 0
   60 continue

! check for duplicate elements .. summing any such entries and
!     printing a warning message on unit mp.
! move is equal to the number of duplicate elements found.
      move = 0
! the loop also calculates the largest element in the matrix, themax.
      themax = zero
! j1 is position in arrays of first non-zero in row.
      j1 = iw(1,1)
      do 130 i=1,n
        iend = nz + 1
        if (i.ne.n) iend = iw(i+1,1)
        length = iend - j1
        if (length.eq.0) go to 130
        j2 = iend - 1
        newj1 = j1 - move
        do 120 jj=j1,j2
          j = icn(jj)
          themax = dmax1(themax,dabs(a(jj)))
          if (ikeep(j,2).eq.i) go to 110
! first time column has ocurred in current row.
          ikeep(j,2) = i
          ikeep(j,3) = jj - move - newj1
          if (move.eq.0) go to 120
! shift necessary because of  previous duplicate element.
          newpos = jj - move
          a(newpos) = a(jj)
          icn(newpos) = icn(jj)
          go to 120
! duplicate element.
  110     move = move + 1
          length = length - 1
          jay = ikeep(j,3) + newj1
          if (mp.ne.0) write (mp,99993) i, j, a(jj)
          a(jay) = a(jay) + a(jj)
          themax = dmax1(themax,dabs(a(jay)))
  120   continue
        ikeep(i,1) = length
        j1 = iend
  130 continue

! knum is actual number of non-zeros in matrix with any multiple
!     entries counted only once.
      knum = nz - move
      if (.not.lblock) go to 140

! perform block triangularisation.
      call mc23ad(n, icn, a, licn, ikeep, idisp, ikeep(1,2),  &
     ikeep(1,3), ikeep(1,5), iw(1,3), iw)
      if (idisp(1).gt.0) go to 170
      iflag = -7
      if (idisp(1).eq.-1) iflag = -1
      if (lp.ne.0) write (lp,99992)
      go to 210

! block triangularization not requested.
! move structure to end of data arrays in preparation for
!     ma30a/ad.
! also set lenoff(1) to -1 and set permutation arrays.
  140 do 150 i=1,knum
        ii = knum - i + 1
        newpos = licn - i + 1
        icn(newpos) = icn(ii)
        a(newpos) = a(ii)
  150 continue
      idisp(1) = 1
      idisp(2) = licn - knum + 1
      do 160 i=1,n
        ikeep(i,2) = i
        ikeep(i,3) = i
  160 continue
      ikeep(1,5) = -1
  170 if (lbig) big1 = themax
      if (nsrch.le.n) go to 180

! perform l/u decomosition on diagonal blocks.
      call ma30ad(n, icn, a, licn, ikeep, ikeep(1,4), idisp,  &
     ikeep(1,2), ikeep(1,3), irn, lirn, iw(1,2), iw(1,3), iw(1,4),  &
     iw(1,5), iw(1,6), iw(1,7), iw(1,8), iw, upriv, iflag)
      go to 190
! this call if used if nsrch has been set less than or equal n.
!     in this case, two integer work arrays of length can be saved.
  180 call ma30ad(n, icn, a, licn, ikeep, ikeep(1,4), idisp,  &
      ikeep(1,2), ikeep(1,3), irn, lirn, iw(1,2), iw(1,3), iw(1,4),  &
      iw(1,5), iw, iw, iw(1,6), iw, upriv, iflag)

! transfer common block information.
  190 minirn = max0(mirn,nz)
      minicn = max0(micn,nz)
      irncp = mirncp
      icncp = micncp
      irank = mirank
      ndrop = ndrop1
      if (lbig) big = big1
      if (iflag.ge.0) go to 200
      if (lp.ne.0) write (lp,99991)
      go to 210

! reorder off-diagonal blocks according to pivot permutation.
  200 i1 = idisp(1) - 1
      if (i1.ne.0) call mc22ad(n, icn, a, i1, ikeep(1,5), ikeep(1,2),  &
      ikeep(1,3), iw, irn)
      i1 = idisp(1)
      iend = licn - i1 + 1

! optionally calculate element growth estimate.
      if (grow) call mc24ad(n, icn, a(i1), iend, ikeep, ikeep(1,4), w)
! increment growth estimate by original maximum element.
      if (grow) w(1) = w(1) + themax
      if (grow .and. n.gt.1) w(2) = themax
! set flag if the only error is due to duplicate elements.
      if (iflag.ge.0 .and. move.ne.0) iflag = -14
      go to 220
  210 if (lp.ne.0) write (lp,99990)
  220 return
99999 format (36x, 17hn out of range = , i10)
99998 format (36x, 18hnz non positive = , i10)
99997 format (36x, 17hlicn too small = , i10)
99996 format (36x, 17hlirn too small = , i10)
99995 format (54h error return from ma28a/ad because indices found out ,  &
      8hof range)
99994 format (1x, i6, 22hth element with value , 1pd22.14, 9h is out o,  &
      21hf range with indices , i8, 2h ,, i8)
99993 format (31h duplicate element in position , i8, 2h ,, i8,  &
      12h with value , 1pd22.14)
99992 format (36x, 26herror return from mc23a/ad)
99991 format (36x, 26herror return from ma30a/ad)
99990 format (36h+error return from ma28a/ad because )
      end subroutine


!######date   01 jan 1984     copyright ukaea, harwell.
!######alias ma30ad
      subroutine ma30ad(nn, icn, a, licn, lenr, lenrl, idisp, ip, iq,  &
      irn, lirn, lenc, ifirst, lastr, nextr, lastc, nextc, iptr, ipc,  &
      u, iflag)
! if  the user requires a more convenient data interface then the ma28
!     package should be used.  the ma28 subroutines call the ma30
!     subroutines after checking the user's input data and optionally
!     using mc23a/ad to permute the matrix to block triangular form.
! this package of subroutines (ma30a/ad, ma30b/bd, ma30c/cd and
!     ma30d/dd) performs operations pertinent to the solution of a
!     general sparse n by n system of linear equations (i.e. solve
!     ax=b). structually singular matrices are permitted including
!     those with row or columns consisting entirely of zeros (i.e.
!     including rectangular matrices).  it is assumed that the
!     non-zeros of the matrix a do not differ widely in size.  if
!     necessary a prior call of the scaling subroutine mc19a/ad may be
!     made.
! a discussion of the design of these subroutines is given by duff and
!     reid (acm trans math software 5 pp 18-35,1979 (css 48)) while
!     fuller details of the implementation are given in duff (harwell
!     report aere-r 8730,1977).  the additional pivoting option in
!     ma30a/ad and the use of drop tolerances (see common block
!     ma30i/id) were added to the package after joint work with reid,
!     schaumburg, wasniewski and zlatev (duff, reid, schaumburg,
!     wasniewski and zlatev, harwell report css 135, 1983).

! ma30a/ad performs the lu decomposition of the diagonal blocks of the
!     permutation paq of a sparse matrix a, where input permutations
!     p1 and q1 are used to define the diagonal blocks.  there may be
!     non-zeros in the off-diagonal blocks but they are unaffected by
!     ma30a/ad. p and p1 differ only within blocks as do q and q1. the
!     permutations p1 and q1 may be found by calling mc23a/ad or the
!     matrix may be treated as a single block by using p1=q1=i. the
!     matrix non-zeros should be held compactly by rows, although it
!     should be noted that the user can supply the matrix by columns
!     to get the lu decomposition of a transpose.

! the parameters are...
! this description should also be consulted for further information on
!     most of the parameters of ma30b/bd and ma30c/cd.

! n  is an integer variable which must be set by the user to the order
!     of the matrix.  it is not altered by ma30a/ad.
! icn is an integer array of length licn. positions idisp(2) to
!     licn must be set by the user to contain the column indices of
!     the non-zeros in the diagonal blocks of p1*a*q1. those belonging
!     to a single row must be contiguous but the ordering of column
!     indices with each row is unimportant. the non-zeros of row i
!     precede those of row i+1,i=1,...,n-1 and no wasted space is
!     allowed between the rows.  on output the column indices of the
!     lu decomposition of paq are held in positions idisp(1) to
!     idisp(2), the rows are in pivotal order, and the column indices
!     of the l part of each row are in pivotal order and precede those
!     of u. again there is no wasted space either within a row or
!     between the rows. icn(1) to icn(idisp(1)-1), are neither
!     required nor altered. if mc23a/ad been called, these will hold
!     information about the off-diagonal blocks.
! a is a real/real(double) array of length licn whose entries
!     idisp(2) to licn must be set by the user to the  values of the
!     non-zero entries of the matrix in the order indicated by  icn.
!     on output a will hold the lu factors of the matrix where again
!     the position in the matrix is determined by the corresponding
!     values in icn. a(1) to a(idisp(1)-1) are neither required nor
!     altered.
! licn  is an integer variable which must be set by the user to the
!     length of arrays icn and a. it must be big enough for a and icn
!     to hold all the non-zeros of l and u and leave some "elbow
!     room".  it is possible to calculate a minimum value for licn by
!     a preliminary run of ma30a/ad. the adequacy of the elbow room
!     can be judged by the size of the common block variable icncp. it
!     is not altered by ma30a/ad.
! lenr  is an integer array of length n.  on input, lenr(i) should
!     equal the number of non-zeros in row i, i=1,...,n of the
!     diagonal blocks of p1*a*q1. on output, lenr(i) will equal the
!     total number of non-zeros in row i of l and row i of u.
! lenrl  is an integer array of length n. on output from ma30a/ad,
!     lenrl(i) will hold the number of non-zeros in row i of l.
! idisp  is an integer array of length 2. the user should set idisp(1)
!     to be the first available position in a/icn for the lu
!     decomposition while idisp(2) is set to the position in a/icn of
!     the first non-zero in the diagonal blocks of p1*a*q1. on output,
!     idisp(1) will be unaltered while idisp(2) will be set to the
!     position in a/icn of the last non-zero of the lu decomposition.
! ip  is an integer array of length n which holds a permutation of
!     the integers 1 to n.  on input to ma30a/ad, the absolute value of
!     ip(i) must be set to the row of a which is row i of p1*a*q1. a
!     negative value for ip(i) indicates that row i is at the end of a
!     diagonal block.  on output from ma30a/ad, ip(i) indicates the row
!     of a which is the i th row in paq. ip(i) will still be negative
!     for the last row of each block (except the last).
! iq is an integer array of length n which again holds a
!     permutation of the integers 1 to n.  on input to ma30a/ad, iq(j)
!     must be set to the column of a which is column j of p1*a*q1. on
!     output from ma30a/ad, the absolute value of iq(j) indicates the
!     column of a which is the j th in paq.  for rows, i say, in which
!     structural or numerical singularity is detected iq(i) is
!     negated.
! irn  is an integer array of length lirn used as workspace by
!     ma30a/ad.
! lirn  is an integer variable. it should be greater than the
!     largest number of non-zeros in a diagonal block of p1*a*q1 but
!     need not be as large as licn. it is the length of array irn and
!     should be large enough to hold the active part of any block,
!     plus some "elbow room", the  a posteriori  adequacy of which can
!     be estimated by examining the size of common block variable
!     irncp.
! lenc,ifirst,lastr,nextr,lastc,nextc are all integer arrays of
!     length n which are used as workspace by ma30a/ad.  if nsrch is
!     set to a value less than or equal to n, then arrays lastc and
!     nextc are not referenced by ma30a/ad and so can be dummied in
!     the call to ma30a/ad.
! iptr,ipc are integer arrays of length n which are used as workspace
!     by ma30a/ad.
! u  is a real/real(double) variable which should be set by the
!     user to a value between 0. and 1.0. if less than zero it is
!     reset to zero and if its value is 1.0 or greater it is reset to
!     0.9999 (0.999999999 in d version).  it determines the balance
!     between pivoting for sparsity and for stability, values near
!     zero emphasizing sparsity and values near one emphasizing
!     stability. we recommend u=0.1 as a posible first trial value.
!     the stability can be judged by a later call to mc24a/ad or by
!     setting lbig to .true.
! iflag  is an integer variable. it will have a non-negative value if
!     ma30a/ad is successful. negative values indicate error
!     conditions while positive values indicate that the matrix has
!     been successfully decomposed but is singular. for each non-zero
!     value, an appropriate message is output on unit lp.  possible
!     non-zero values for iflag are ...

! -1  the matrix is structually singular with rank given by irank in
!     common block ma30f/fd.
! +1  if, however, the user wants the lu decomposition of a
!     structurally singular matrix and sets the common block variable
!     abort1 to .false., then, in the event of singularity and a
!     successful decomposition, iflag is returned with the value +1
!     and no message is output.
! -2  the matrix is numerically singular (it may also be structually
!     singular) with estimated rank given by irank in common block
!     ma30f/fd.
! +2  the  user can choose to continue the decomposition even when a
!     zero pivot is encountered by setting common block variable
!     abort2 to .false.  if a singularity is encountered, iflag will
!     then return with a value of +2, and no message is output if the
!     decomposition has been completed successfully.
! -3  lirn has not been large enough to continue with the
!     decomposition.  if the stage was zero then common block variable
!     minirn gives the length sufficient to start the decomposition on
!     this block.  for a successful decomposition on this block the
!     user should make lirn slightly (say about n/2) greater than this
!     value.
! -4  licn not large enough to continue with the decomposition.
! -5  the decomposition has been completed but some of the lu factors
!     have been discarded to create enough room in a/icn to continue
!     the decomposition. the variable minicn in common block ma30f/fd
!     then gives the size that licn should be to enable the
!     factorization to be successful.  if the user sets common block
!     variable abort3 to .true., then the subroutine will exit
!     immediately instead of destroying any factors and continuing.
! -6  both licn and lirn are too small. termination has been caused by
!     lack of space in irn (see error iflag= -3), but already some of
!     the lu factors in a/icn have been lost (see error iflag= -5).
!     minicn gives the minimum amount of space required in a/icn for
!     decomposition up to this point.

      real(double) a(licn), u, au, umax, amax, zero, pivrat, pivr,  &
      tol, big, anew, aanew, scale
      integer iptr(nn), pivot, pivend, dispc, oldpiv, oldend, pivrow,  &
      rowi, ipc(nn), idisp(2), colupd
      integer icn(licn), lenr(nn), lenrl(nn), ip(nn), iq(nn),  &
      lenc(nn), irn(lirn), ifirst(nn), lastr(nn), nextr(nn),  &
      lastc(nn), nextc(nn)
      logical abort1, abort2, abort3, lbig
! for comments of common block variables see block data subprogram.
      common /ma30ed/ lp, abort1, abort2, abort3
      common /ma30fd/ irncp, icncp, irank, minirn, minicn
      common /ma30id/ tol, big, ndrop, nsrch, lbig
      common /lpivot/ lpiv(10),lnpiv(10),mapiv,manpiv,iavpiv,  &
                     ianpiv,kountl

      data umax/.999999999d0/
      data zero /0.0d0/
      msrch = nsrch
      ndrop = 0
      do 1272 kk=1,10
        lnpiv(kk)=0
        lpiv(kk)=0
 1272 continue
      mapiv = 0
      manpiv = 0
      iavpiv = 0
      ianpiv = 0
      kountl = 0
      minirn = 0
      minicn = idisp(1) - 1
      morei = 0
      irank = nn
      irncp = 0
      icncp = 0
      iflag = 0
! reset u if necessary.
      u = dmin1(u,umax)
! ibeg is the position of the next pivot row after elimination step
!     using it.
      u = dmax1(u,zero)
      ibeg = idisp(1)
! iactiv is the position of the first entry in the active part of a/icn.
      iactiv = idisp(2)
! nzrow is current number of non-zeros in active and unprocessed part
!     of row file icn.
      nzrow = licn - iactiv + 1
      minicn = nzrow + minicn

! count the number of diagonal blocks and set up pointers to the
!     beginnings of the rows.
! num is the number of diagonal blocks.
      num = 1
      iptr(1) = iactiv
      if (nn.eq.1) go to 20
      nnm1 = nn - 1
      do 10 i=1,nnm1
        if (ip(i).lt.0) num = num + 1
        iptr(i+1) = iptr(i) + lenr(i)
   10 continue
! ilast is the last row in the previous block.
   20 ilast = 0

! ***********************************************
! ****    lu decomposition of block nblock   ****
! ***********************************************

! each pass through this loop performs lu decomposition on one
!     of the diagonal blocks.
      do 1000 nblock=1,num
        istart = ilast + 1
        do 30 irows=istart,nn
          if (ip(irows).lt.0) go to 40
   30   continue
        irows = nn
   40   ilast = irows
! n is the number of rows in the current block.
! istart is the index of the first row in the current block.
! ilast is the index of the last row in the current block.
! iactiv is the position of the first entry in the block.
! itop is the position of the last entry in the block.
        n = ilast - istart + 1
        if (n.ne.1) go to 90

! code for dealing with 1x1 block.
        lenrl(ilast) = 0
        ising = istart
        if (lenr(ilast).ne.0) go to 50
! block is structurally singular.
        irank = irank - 1
        ising = -ising
        if (iflag.ne.2 .and. iflag.ne.-5) iflag = 1
        if (.not.abort1) go to 80
        idisp(2) = iactiv
        iflag = -1
        if (lp.ne.0) write (lp,99999)
!     return
        go to 1120
   50   scale = dabs(a(iactiv))
        if (scale.eq.zero) go to 60
        if (lbig) big = dmax1(big,scale)
        go to 70
   60   ising = -ising
        irank = irank - 1
        iptr(ilast) = 0
        if (iflag.ne.-5) iflag = 2
        if (.not.abort2) go to 70
        idisp(2) = iactiv
        iflag = -2
        if (lp.ne.0) write (lp,99998)
        go to 1120
   70   a(ibeg) = a(iactiv)
        icn(ibeg) = icn(iactiv)
        iactiv = iactiv + 1
        iptr(istart) = 0
        ibeg = ibeg + 1
        nzrow = nzrow - 1
   80   lastr(istart) = istart
        ipc(istart) = -ising
        go to 1000

! non-trivial block.
   90   itop = licn
        if (ilast.ne.nn) itop = iptr(ilast+1) - 1

! set up column oriented storage.
        do 100 i=istart,ilast
          lenrl(i) = 0
          lenc(i) = 0
  100   continue
        if (itop-iactiv.lt.lirn) go to 110
        minirn = itop - iactiv + 1
        pivot = istart - 1
        go to 1100

! calculate column counts.
  110   do 120 ii=iactiv,itop
          i = icn(ii)
          lenc(i) = lenc(i) + 1
  120   continue
! set up column pointers so that ipc(j) points to position after end
!     of column j in column file.
        ipc(ilast) = lirn + 1
        j1 = istart + 1
        do 130 jj=j1,ilast
          j = ilast - jj + j1 - 1
          ipc(j) = ipc(j+1) - lenc(j+1)
  130   continue
        do 150 indrow=istart,ilast
          j1 = iptr(indrow)
          j2 = j1 + lenr(indrow) - 1
          if (j1.gt.j2) go to 150
          do 140 jj=j1,j2
            j = icn(jj)
            ipos = ipc(j) - 1
            irn(ipos) = indrow
            ipc(j) = ipos
  140     continue
  150   continue
! dispc is the lowest indexed active location in the column file.
        dispc = ipc(istart)
        nzcol = lirn - dispc + 1
        minirn = max0(nzcol,minirn)
        nzmin = 1

! initialize array ifirst.  ifirst(i) = +/- k indicates that row/col
!     k has i non-zeros.  if ifirst(i) = 0, there is no row or column
!     with i non zeros.
        do 160 i=1,n
          ifirst(i) = 0
  160   continue

! compute ordering of row and column counts.
! first run through columns (from column n to column 1).
        do 180 jj=istart,ilast
          j = ilast - jj + istart
          nz = lenc(j)
          if (nz.ne.0) go to 170
          ipc(j) = 0
          go to 180
  170     if (nsrch.le.nn) go to 180
          isw = ifirst(nz)
          ifirst(nz) = -j
          lastc(j) = 0
          nextc(j) = -isw
          isw1 = iabs(isw)
          if (isw.ne.0) lastc(isw1) = j
  180   continue
! now run through rows (again from n to 1).
        do 210 ii=istart,ilast
          i = ilast - ii + istart
          nz = lenr(i)
          if (nz.ne.0) go to 190
          iptr(i) = 0
          lastr(i) = 0
          go to 210
  190     isw = ifirst(nz)
          ifirst(nz) = i
          if (isw.gt.0) go to 200
          nextr(i) = 0
          lastr(i) = isw
          go to 210
  200     nextr(i) = isw
          lastr(i) = lastr(isw)
          lastr(isw) = i
  210   continue

! **********************************************
! ****    start of main elimination loop    ****
! **********************************************
        do 980 pivot=istart,ilast

! first find the pivot using markowitz criterion with stability
!     control.
! jcost is the markowitz cost of the best pivot so far,.. this
!     pivot is in row ipiv and column jpiv.
          nz2 = nzmin
          jcost = n*n

! examine rows/columns in order of ascending count.
          do 340 l=1,2
            pivrat = zero
            isrch = 1
            ll = l
! a pass with l equal to 2 is only performed in the case of singularity.
            do 330 nz=nz2,n
              if (jcost.le.(nz-1)**2) go to 420
              ijfir = ifirst(nz)
              if (ijfir) 230, 220, 240
  220         if (ll.eq.1) nzmin = nz + 1
              go to 330
  230         ll = 2
              ijfir = -ijfir
              go to 290
  240         ll = 2
! scan rows with nz non-zeros.
              do 270 idummy=1,n
                if (jcost.le.(nz-1)**2) go to 420
                if (isrch.gt.msrch) go to 420
                if (ijfir.eq.0) go to 280
! row ijfir is now examined.
                i = ijfir
                ijfir = nextr(i)
! first calculate multiplier threshold level.
                amax = zero
                j1 = iptr(i) + lenrl(i)
                j2 = iptr(i) + lenr(i) - 1
                do 250 jj=j1,j2
                  amax = dmax1(amax,dabs(a(jj)))
  250           continue
                au = amax*u
                isrch = isrch + 1
! scan row for possible pivots
                do 260 jj=j1,j2
                  if (dabs(a(jj)).le.au .and. l.eq.1) go to 260
                  j = icn(jj)
                  kcost = (nz-1)*(lenc(j)-1)
                  if (kcost.gt.jcost) go to 260
                  pivr = zero
                  if (amax.ne.zero) pivr = dabs(a(jj))/amax
                  if (kcost.eq.jcost .and. (pivr.le.pivrat .or.  &
                  nsrch.gt.nn+1)) go to 260
! best pivot so far is found.
                  jcost = kcost
                  ijpos = jj
                  ipiv = i
                  jpiv = j
                  if (msrch.gt.nn+1 .and. jcost.le.(nz-1)**2) go to 420
                  pivrat = pivr
  260           continue
  270         continue

! columns with nz non-zeros now examined.
  280         ijfir = ifirst(nz)
              ijfir = -lastr(ijfir)
  290         if (jcost.le.nz*(nz-1)) go to 420
              if (msrch.le.nn) go to 330
              do 320 idummy=1,n
                if (ijfir.eq.0) go to 330
                j = ijfir
                ijfir = nextc(ijfir)
                i1 = ipc(j)
                i2 = i1 + nz - 1
! scan column j.
                do 310 ii=i1,i2
                  i = irn(ii)
                  kcost = (nz-1)*(lenr(i)-lenrl(i)-1)
                  if (kcost.ge.jcost) go to 310
! pivot has best markowitz count so far ... now check its
!     suitability on numeric grounds by examining the other non-zeros
!     in its row.
                  j1 = iptr(i) + lenrl(i)
                  j2 = iptr(i) + lenr(i) - 1
! we need a stability check on singleton columns because of possible
!     problems with underdetermined systems.
                  amax = zero
                  do 300 jj=j1,j2
                    amax = dmax1(amax,dabs(a(jj)))
                    if (icn(jj).eq.j) jpos = jj
  300             continue
                  if (dabs(a(jpos)).le.amax*u .and. l.eq.1) go to 310
                  jcost = kcost
                  ipiv = i
                  jpiv = j
                  ijpos = jpos
                  if (amax.ne.zero) pivrat = dabs(a(jpos))/amax
                  if (jcost.le.nz*(nz-1)) go to 420
  310           continue

  320         continue

  330       continue
! in the event of singularity, we must make sure all rows and columns
! are tested.
            msrch = n

! matrix is numerically or structurally singular  ... which it is will
!     be diagnosed later.
            irank = irank - 1
  340     continue
! assign rest of rows and columns to ordering array.
! matrix is structurally singular.
          if (iflag.ne.2 .and. iflag.ne.-5) iflag = 1
          irank = irank - ilast + pivot + 1
          if (.not.abort1) go to 350
          idisp(2) = iactiv
          iflag = -1
          if (lp.ne.0) write (lp,99999)
          go to 1120
  350     k = pivot - 1
          do 390 i=istart,ilast
            if (lastr(i).ne.0) go to 390
            k = k + 1
            lastr(i) = k
            if (lenrl(i).eq.0) go to 380
            minicn = max0(minicn,nzrow+ibeg-1+morei+lenrl(i))
            if (iactiv-ibeg.ge.lenrl(i)) go to 360
            call ma30dd(a, icn, iptr(istart), n, iactiv, itop, .true.)
! check now to see if ma30d/dd has created enough available space.
            if (iactiv-ibeg.ge.lenrl(i)) go to 360
! create more space by destroying previously created lu factors.
            morei = morei + ibeg - idisp(1)
            ibeg = idisp(1)
            if (lp.ne.0) write (lp,99997)
            iflag = -5
            if (abort3) go to 1090
  360       j1 = iptr(i)
            j2 = j1 + lenrl(i) - 1
            iptr(i) = 0
            do 370 jj=j1,j2
              a(ibeg) = a(jj)
              icn(ibeg) = icn(jj)
              icn(jj) = 0
              ibeg = ibeg + 1
  370       continue
            nzrow = nzrow - lenrl(i)
  380       if (k.eq.ilast) go to 400
  390     continue
  400     k = pivot - 1
          do 410 i=istart,ilast
            if (ipc(i).ne.0) go to 410
            k = k + 1
            ipc(i) = k
            if (k.eq.ilast) go to 990
  410     continue

! the pivot has now been found in position (ipiv,jpiv) in location
!     ijpos in row file.
! update column and row ordering arrays to correspond with removal
!     of the active part of the matrix.
  420     ising = pivot
          if (a(ijpos).ne.zero) go to 430
! numerical singularity is recorded here.
          ising = -ising
          if (iflag.ne.-5) iflag = 2
          if (.not.abort2) go to 430
          idisp(2) = iactiv
          iflag = -2
          if (lp.ne.0) write (lp,99998)
          go to 1120
  430     oldpiv = iptr(ipiv) + lenrl(ipiv)
          oldend = iptr(ipiv) + lenr(ipiv) - 1
! changes to column ordering.
          if (nsrch.le.nn) go to 460
          colupd = nn + 1
            lenpp = oldend-oldpiv+1
            if (lenpp.lt.4) lpiv(1) = lpiv(1) + 1
            if (lenpp.ge.4 .and. lenpp.le.6) lpiv(2) = lpiv(2) + 1
            if (lenpp.ge.7 .and. lenpp.le.10) lpiv(3) = lpiv(3) + 1
            if (lenpp.ge.11 .and. lenpp.le.15) lpiv(4) = lpiv(4) + 1
            if (lenpp.ge.16 .and. lenpp.le.20) lpiv(5) = lpiv(5) + 1
            if (lenpp.ge.21 .and. lenpp.le.30) lpiv(6) = lpiv(6) + 1
            if (lenpp.ge.31 .and. lenpp.le.50) lpiv(7) = lpiv(7) + 1
            if (lenpp.ge.51 .and. lenpp.le.70) lpiv(8) = lpiv(8) + 1
            if (lenpp.ge.71 .and. lenpp.le.100) lpiv(9) = lpiv(9) + 1
            if (lenpp.ge.101) lpiv(10) = lpiv(10) + 1
            mapiv = max0(mapiv,lenpp)
            iavpiv = iavpiv + lenpp
          do 450 jj=oldpiv,oldend
            j = icn(jj)
            lc = lastc(j)
            nc = nextc(j)
            nextc(j) = -colupd
            if (jj.ne.ijpos) colupd = j
            if (nc.ne.0) lastc(nc) = lc
            if (lc.eq.0) go to 440
            nextc(lc) = nc
            go to 450
  440       nz = lenc(j)
            isw = ifirst(nz)
            if (isw.gt.0) lastr(isw) = -nc
            if (isw.lt.0) ifirst(nz) = -nc
  450     continue
! changes to row ordering.
  460     i1 = ipc(jpiv)
          i2 = i1 + lenc(jpiv) - 1
          do 480 ii=i1,i2
            i = irn(ii)
            lr = lastr(i)
            nr = nextr(i)
            if (nr.ne.0) lastr(nr) = lr
            if (lr.le.0) go to 470
            nextr(lr) = nr
            go to 480
  470       nz = lenr(i) - lenrl(i)
            if (nr.ne.0) ifirst(nz) = nr
            if (nr.eq.0) ifirst(nz) = lr
  480     continue

! move pivot to position lenrl+1 in pivot row and move pivot row
!     to the beginning of the available storage.
! the l part and the pivot in the old copy of the pivot row is
!     nullified while, in the strictly upper triangular part, the
!     column indices, j say, are overwritten by the corresponding
!     entry of iq (iq(j)) and iq(j) is set to the negative of the
!     displacement of the column index from the pivot entry.
          if (oldpiv.eq.ijpos) go to 490
          au = a(oldpiv)
          a(oldpiv) = a(ijpos)
          a(ijpos) = au
          icn(ijpos) = icn(oldpiv)
          icn(oldpiv) = jpiv
! check to see if there is space immediately available in a/icn to
!     hold new copy of pivot row.
  490     minicn = max0(minicn,nzrow+ibeg-1+morei+lenr(ipiv))
          if (iactiv-ibeg.ge.lenr(ipiv)) go to 500
          call ma30dd(a, icn, iptr(istart), n, iactiv, itop, .true.)
          oldpiv = iptr(ipiv) + lenrl(ipiv)
          oldend = iptr(ipiv) + lenr(ipiv) - 1
! check now to see if ma30d/dd has created enough available space.
          if (iactiv-ibeg.ge.lenr(ipiv)) go to 500
! create more space by destroying previously created lu factors.
          morei = morei + ibeg - idisp(1)
          ibeg = idisp(1)
          if (lp.ne.0) write (lp,99997)
          iflag = -5
          if (abort3) go to 1090
          if (iactiv-ibeg.ge.lenr(ipiv)) go to 500
! there is still not enough room in a/icn.
          iflag = -4
          go to 1090
! copy pivot row and set up iq array.
  500     ijpos = 0
          j1 = iptr(ipiv)

          do 530 jj=j1,oldend
            a(ibeg) = a(jj)
            icn(ibeg) = icn(jj)
            if (ijpos.ne.0) go to 510
            if (icn(jj).eq.jpiv) ijpos = ibeg
            icn(jj) = 0
            go to 520
  510       k = ibeg - ijpos
            j = icn(jj)
            icn(jj) = iq(j)
            iq(j) = -k
  520       ibeg = ibeg + 1
  530     continue

          ijp1 = ijpos + 1
          pivend = ibeg - 1
          lenpiv = pivend - ijpos
          nzrow = nzrow - lenrl(ipiv) - 1
          iptr(ipiv) = oldpiv + 1
          if (lenpiv.eq.0) iptr(ipiv) = 0

! remove pivot row (including pivot) from column oriented file.
          do 560 jj=ijpos,pivend
            j = icn(jj)
            i1 = ipc(j)
            lenc(j) = lenc(j) - 1
! i2 is last position in new column.
            i2 = ipc(j) + lenc(j) - 1
            if (i2.lt.i1) go to 550
            do 540 ii=i1,i2
              if (irn(ii).ne.ipiv) go to 540
              irn(ii) = irn(i2+1)
              go to 550
  540       continue
  550       irn(i2+1) = 0
  560     continue
          nzcol = nzcol - lenpiv - 1

! go down the pivot column and for each row with a non-zero add
!     the appropriate multiple of the pivot row to it.
! we loop on the number of non-zeros in the pivot column since
!     ma30d/dd may change its actual position.

          nzpc = lenc(jpiv)
          if (nzpc.eq.0) go to 900
          do 840 iii=1,nzpc
            ii = ipc(jpiv) + iii - 1
            i = irn(ii)
! search row i for non-zero to be eliminated, calculate multiplier,
!     and place it in position lenrl+1 in its row.
!  idrop is the number of non-zero entries dropped from row    i
!        because these fall beneath tolerance level.

            idrop = 0
            j1 = iptr(i) + lenrl(i)
            iend = iptr(i) + lenr(i) - 1
            do 570 jj=j1,iend
              if (icn(jj).ne.jpiv) go to 570
! if pivot is zero, rest of column is and so multiplier is zero.
              au = zero
              if (a(ijpos).ne.zero) au = -a(jj)/a(ijpos)
              if (lbig) big = dmax1(big,dabs(au))
              a(jj) = a(j1)
              a(j1) = au
              icn(jj) = icn(j1)
              icn(j1) = jpiv
              lenrl(i) = lenrl(i) + 1
              go to 580
  570       continue
! jump if pivot row is a singleton.
  580       if (lenpiv.eq.0) go to 840
! now perform necessary operations on rest of non-pivot row i.
            rowi = j1 + 1
            iop = 0
! jump if all the pivot row causes fill-in.
            if (rowi.gt.iend) go to 650
! perform operations on current non-zeros in row i.
! innermost loop.
            lenpp = iend-rowi+1
            if (lenpp.lt.4) lnpiv(1) = lnpiv(1) + 1
            if (lenpp.ge.4 .and. lenpp.le.6) lnpiv(2) = lnpiv(2) + 1
            if (lenpp.ge.7 .and. lenpp.le.10) lnpiv(3) = lnpiv(3) + 1
            if (lenpp.ge.11 .and. lenpp.le.15) lnpiv(4) = lnpiv(4) + 1
            if (lenpp.ge.16 .and. lenpp.le.20) lnpiv(5) = lnpiv(5) + 1
            if (lenpp.ge.21 .and. lenpp.le.30) lnpiv(6) = lnpiv(6) + 1
            if (lenpp.ge.31 .and. lenpp.le.50) lnpiv(7) = lnpiv(7) + 1
            if (lenpp.ge.51 .and. lenpp.le.70) lnpiv(8) = lnpiv(8) + 1
            if (lenpp.ge.71 .and. lenpp.le.100) lnpiv(9) = lnpiv(9) + 1
            if (lenpp.ge.101) lnpiv(10) = lnpiv(10) + 1
            manpiv = max0(manpiv,lenpp)
            ianpiv = ianpiv + lenpp
            kountl = kountl + 1
            do 590 jj=rowi,iend
              j = icn(jj)
              if (iq(j).gt.0) go to 590
              iop = iop + 1
              pivrow = ijpos - iq(j)
              a(jj) = a(jj) + au*a(pivrow)
              if (lbig) big = dmax1(dabs(a(jj)),big)
              icn(pivrow) = -icn(pivrow)
              if (dabs(a(jj)).lt.tol) idrop = idrop + 1
  590       continue

!  jump if no non-zeros in non-pivot row have been removed
!       because these are beneath the drop-tolerance  tol.

            if (idrop.eq.0) go to 650

!  run through non-pivot row compressing row so that only
!      non-zeros greater than   tol   are stored.  all non-zeros
!      less than   tol   are also removed from the column structure.

            jnew = rowi
            do 630 jj=rowi,iend
              if (dabs(a(jj)).lt.tol) go to 600
              a(jnew) = a(jj)
              icn(jnew) = icn(jj)
              jnew = jnew + 1
              go to 630

!  remove non-zero entry from column structure.

  600         j = icn(jj)
              i1 = ipc(j)
              i2 = i1 + lenc(j) - 1
              do 610 ii=i1,i2
                if (irn(ii).eq.i) go to 620
  610         continue
  620         irn(ii) = irn(i2)
              irn(i2) = 0
              lenc(j) = lenc(j) - 1
              if (nsrch.le.nn) go to 630
! remove column from column chain and place in update chain.
              if (nextc(j).lt.0) go to 630
! jump if column already in update chain.
              lc = lastc(j)
              nc = nextc(j)
              nextc(j) = -colupd
              colupd = j
              if (nc.ne.0) lastc(nc) = lc
              if (lc.eq.0) go to 622
              nextc(lc) = nc
              go to 630
  622         nz = lenc(j) + 1
              isw = ifirst(nz)
              if (isw.gt.0) lastr(isw) = -nc
              if (isw.lt.0) ifirst(nz) = -nc
  630       continue
            do 640 jj=jnew,iend
              icn(jj) = 0
  640       continue
! the value of idrop might be different from that calculated earlier
!     because, we may now have dropped some non-zeros which were not
!     modified by the pivot row.
            idrop = iend + 1 - jnew
            iend = jnew - 1
            lenr(i) = lenr(i) - idrop
            nzrow = nzrow - idrop
            nzcol = nzcol - idrop
            ndrop = ndrop + idrop
  650       ifill = lenpiv - iop
! jump is if there is no fill-in.
            if (ifill.eq.0) go to 750
! now for the fill-in.
            minicn = max0(minicn,morei+ibeg-1+nzrow+ifill+lenr(i))
! see if there is room for fill-in.
! get maximum space for row i in situ.
            do 660 jdiff=1,ifill
              jnpos = iend + jdiff
              if (jnpos.gt.licn) go to 670
              if (icn(jnpos).ne.0) go to 670
  660       continue
! there is room for all the fill-in after the end of the row so it
!     can be left in situ.
! next available space for fill-in.
            iend = iend + 1
            go to 750
! jmore spaces for fill-in are required in front of row.
  670       jmore = ifill - jdiff + 1
            i1 = iptr(i)
! we now look in front of the row to see if there is space for
!     the rest of the fill-in.
            do 680 jdiff=1,jmore
              jnpos = i1 - jdiff
              if (jnpos.lt.iactiv) go to 690
              if (icn(jnpos).ne.0) go to 700
  680       continue
  690       jnpos = i1 - jmore
            go to 710
! whole row must be moved to the beginning of available storage.
  700       jnpos = iactiv - lenr(i) - ifill
! jump if there is space immediately available for the shifted row.
  710       if (jnpos.ge.ibeg) go to 730
            call ma30dd(a, icn, iptr(istart), n, iactiv, itop, .true.)
            i1 = iptr(i)
            iend = i1 + lenr(i) - 1
            jnpos = iactiv - lenr(i) - ifill
            if (jnpos.ge.ibeg) go to 730
! no space available so try to create some by throwing away previous
!     lu decomposition.
            morei = morei + ibeg - idisp(1) - lenpiv - 1
            if (lp.ne.0) write (lp,99997)
            iflag = -5
            if (abort3) go to 1090
! keep record of current pivot row.
            ibeg = idisp(1)
            icn(ibeg) = jpiv
            a(ibeg) = a(ijpos)
            ijpos = ibeg
            do 720 jj=ijp1,pivend
              ibeg = ibeg + 1
              a(ibeg) = a(jj)
              icn(ibeg) = icn(jj)
  720       continue
            ijp1 = ijpos + 1
            pivend = ibeg
            ibeg = ibeg + 1
            if (jnpos.ge.ibeg) go to 730
! this still does not give enough room.
            iflag = -4
            go to 1090
  730       iactiv = min0(iactiv,jnpos)
! move non-pivot row i.
            iptr(i) = jnpos
            do 740 jj=i1,iend
              a(jnpos) = a(jj)
              icn(jnpos) = icn(jj)
              jnpos = jnpos + 1
              icn(jj) = 0
  740       continue
! first new available space.
            iend = jnpos
  750       nzrow = nzrow + ifill
! innermost fill-in loop which also resets icn.
            idrop = 0
            do 830 jj=ijp1,pivend
              j = icn(jj)
              if (j.lt.0) go to 820
              anew = au*a(jj)
              aanew = dabs(anew)
              if (aanew.ge.tol) go to 760
              idrop = idrop + 1
              ndrop = ndrop + 1
              nzrow = nzrow - 1
              minicn = minicn - 1
              ifill = ifill - 1
              go to 830
  760         if (lbig) big = dmax1(aanew,big)
              a(iend) = anew
              icn(iend) = j
              iend = iend + 1

! put new entry in column file.
              minirn = max0(minirn,nzcol+lenc(j)+1)
              jend = ipc(j) + lenc(j)
              jroom = nzpc - iii + 1 + lenc(j)
              if (jend.gt.lirn) go to 770
              if (irn(jend).eq.0) go to 810
  770         if (jroom.lt.dispc) go to 780
! compress column file to obtain space for new copy of column.
              call ma30dd(a, irn, ipc(istart), n, dispc, lirn, .false.)
              if (jroom.lt.dispc) go to 780
              jroom = dispc - 1
              if (jroom.ge.lenc(j)+1) go to 780
! column file is not large enough.
              go to 1100
! copy column to beginning of file.
  780         jbeg = ipc(j)
              jend = ipc(j) + lenc(j) - 1
              jzero = dispc - 1
              dispc = dispc - jroom
              idispc = dispc
              do 790 ii=jbeg,jend
                irn(idispc) = irn(ii)
                irn(ii) = 0
                idispc = idispc + 1
  790         continue
              ipc(j) = dispc
              jend = idispc
              do 800 ii=jend,jzero
                irn(ii) = 0
  800         continue
  810         irn(jend) = i
              nzcol = nzcol + 1
              lenc(j) = lenc(j) + 1
! end of adjustment to column file.
              go to 830

  820         icn(jj) = -j
  830       continue
            if (idrop.eq.0) go to 834
            do 832 kdrop=1,idrop
            icn(iend) = 0
            iend = iend + 1
  832       continue
  834       lenr(i) = lenr(i) + ifill
! end of scan of pivot column.
  840     continue

! remove pivot column from column oriented storage and update row
!     ordering arrays.
          i1 = ipc(jpiv)
          i2 = ipc(jpiv) + lenc(jpiv) - 1
          nzcol = nzcol - lenc(jpiv)
          do 890 ii=i1,i2
            i = irn(ii)
            irn(ii) = 0
            nz = lenr(i) - lenrl(i)
            if (nz.ne.0) go to 850
            lastr(i) = 0
            go to 890
  850       ifir = ifirst(nz)
            ifirst(nz) = i
            if (ifir) 860, 880, 870
  860       lastr(i) = ifir
            nextr(i) = 0
            go to 890
  870       lastr(i) = lastr(ifir)
            nextr(i) = ifir
            lastr(ifir) = i
            go to 890
  880       lastr(i) = 0
            nextr(i) = 0
            nzmin = min0(nzmin,nz)
  890     continue
! restore iq and nullify u part of old pivot row.
!    record the column permutation in lastc(jpiv) and the row
!    permutation in lastr(ipiv).
  900     ipc(jpiv) = -ising
          lastr(ipiv) = pivot
          if (lenpiv.eq.0) go to 980
          nzrow = nzrow - lenpiv
          jval = ijp1
          jzer = iptr(ipiv)
          iptr(ipiv) = 0
          do 910 jcount=1,lenpiv
            j = icn(jval)
            iq(j) = icn(jzer)
            icn(jzer) = 0
            jval = jval + 1
            jzer = jzer + 1
  910     continue
! adjust column ordering arrays.
          if (nsrch.gt.nn) go to 920
          do 916 jj=ijp1,pivend
            j = icn(jj)
            nz = lenc(j)
            if (nz.ne.0) go to 914
            ipc(j) = 0
            go to 916
  914       nzmin = min0(nzmin,nz)
  916     continue
          go to 980
  920     jj = colupd
          do 970 jdummy=1,nn
            j = jj
            if (j.eq.nn+1) go to 980
            jj = -nextc(j)
            nz = lenc(j)
            if (nz.ne.0) go to 924
            ipc(j) = 0
            go to 970
  924       ifir = ifirst(nz)
            lastc(j) = 0
            if (ifir) 930, 940, 950
  930       ifirst(nz) = -j
            ifir = -ifir
            lastc(ifir) = j
            nextc(j) = ifir
            go to 970
  940       ifirst(nz) = -j
            nextc(j) = 0
            go to 960
  950       lc = -lastr(ifir)
            lastr(ifir) = -j
            nextc(j) = lc
            if (lc.ne.0) lastc(lc) = j
  960       nzmin = min0(nzmin,nz)
  970     continue
  980   continue
! ********************************************
! ****    end of main elimination loop    ****
! ********************************************

! reset iactiv to point to the beginning of the next block.
  990   if (ilast.ne.nn) iactiv = iptr(ilast+1)
 1000 continue

! ********************************************
! ****    end of deomposition of block    ****
! ********************************************

! record singularity (if any) in iq array.
      if (irank.eq.nn) go to 1020
      do 1010 i=1,nn
        if (ipc(i).lt.0) go to 1010
        ising = ipc(i)
        iq(ising) = -iq(ising)
        ipc(i) = -ising
 1010 continue

! run through lu decomposition changing column indices to that of new
!     order and permuting lenr and lenrl arrays according to pivot
!     permutations.
 1020 istart = idisp(1)
      iend = ibeg - 1
      if (iend.lt.istart) go to 1040
      do 1030 jj=istart,iend
        jold = icn(jj)
        icn(jj) = -ipc(jold)
 1030 continue
 1040 do 1050 ii=1,nn
        i = lastr(ii)
        nextr(i) = lenr(ii)
        iptr(i) = lenrl(ii)
 1050 continue
      do 1060 i=1,nn
        lenrl(i) = iptr(i)
        lenr(i) = nextr(i)
 1060 continue

! update permutation arrays ip and iq.
      do 1070 ii=1,nn
        i = lastr(ii)
        j = -ipc(ii)
        nextr(i) = iabs(ip(ii)+0)
        iptr(j) = iabs(iq(ii)+0)
 1070 continue
      do 1080 i=1,nn
        if (ip(i).lt.0) nextr(i) = -nextr(i)
        ip(i) = nextr(i)
        if (iq(i).lt.0) iptr(i) = -iptr(i)
        iq(i) = iptr(i)
 1080 continue
      ip(nn) = iabs(ip(nn)+0)
      idisp(2) = iend
      go to 1120

!   ***    error returns    ***
 1090 idisp(2) = iactiv
      if (lp.eq.0) go to 1120
      write (lp,99996)
      go to 1110
 1100 if (iflag.eq.-5) iflag = -6
      if (iflag.ne.-6) iflag = -3
      idisp(2) = iactiv
      if (lp.eq.0) go to 1120
      if (iflag.eq.-3) write (lp,99995)
      if (iflag.eq.-6) write (lp,99994)
 1110 pivot = pivot - istart + 1
      write (lp,99993) pivot, nblock, istart, ilast
      if (pivot.eq.0) write (lp,99992) minirn

 1120 return
99999 format (54h error return from ma30a/ad because matrix is structur,  &
      13hally singular)
99998 format (54h error return from ma30a/ad because matrix is numerica,  &
      12hlly singular)
99997 format (48h lu decomposition destroyed to create more space)
99996 format (54h error return from ma30a/ad because licn not big enoug,  &
      1hh)
99995 format (54h error return from ma30a/ad because lirn not big enoug,  &
      1hh)
99994 format (51h error return from ma30a/ad lirn and licn too small)
99993 format (10h at stage , i5, 10h in block , i5, 16h with first row ,  &
      i5, 14h and last row , i5)
99992 format (34h to continue set lirn to at least , i8)
      end subroutine

      subroutine ma30dd(a, icn, iptr, n, iactiv, itop, reals)
! this subroutine performs garbage collection operations on the
!     arrays a, icn and irn.
! iactiv is the first position in arrays a/icn from which the compress
!     starts.  on exit, iactiv equals the position of the first entry
!     in the compressed part of a/icn

      real(double) a(itop)
      logical reals
      integer iptr(n)
      integer icn(itop)
! see block data for comments on variables in common.
      common /ma30fd/ irncp, icncp, irank, minirn, minicn

      if (reals) icncp = icncp + 1
      if (.not.reals) irncp = irncp + 1
! set the first non-zero entry in each row to the negative of the
!     row/col number and hold this row/col index in the row/col
!     pointer.  this is so that the beginning of each row/col can
!     be recognized in the subsequent scan.
      do 10 j=1,n
        k = iptr(j)
        if (k.lt.iactiv) go to 10
        iptr(j) = icn(k)
        icn(k) = -j
   10 continue
      kn = itop + 1
      kl = itop - iactiv + 1
! go through arrays in reverse order compressing to the back so
!     that there are no zeros held in positions iactiv to itop in icn.
!     reset first entry of each row/col and pointer array iptr.
      do 30 k=1,kl
        jpos = itop - k + 1
        if (icn(jpos).eq.0) go to 30
        kn = kn - 1
        if (reals) a(kn) = a(jpos)
        if (icn(jpos).ge.0) go to 20
! first non-zero of row/col has been located
        j = -icn(jpos)
        icn(jpos) = iptr(j)
        iptr(j) = kn
   20   icn(kn) = icn(jpos)
   30 continue
      iactiv = kn
      return
      end subroutine

!######date   01 jan 1984     copyright ukaea, harwell.
!######alias mc13d
      subroutine mc13d(n,icn,licn,ip,lenr,ior,ib,num,iw)
      integer ip(n)
      integer icn(licn),lenr(n),ior(n),ib(n),iw(n,3)
      call mc13e(n,icn,licn,ip,lenr,ior,ib,num,iw(1,1),iw(1,2),iw(1,3))
      return
      end subroutine

      subroutine mc13e(n,icn,licn,ip,lenr,arp,ib,num,lowl,numb,prev)
      integer stp,dummy
      integer ip(n)

! arp(i) is one less than the number of unsearched edges leaving
!     node i.  at the end of the algorithm it is set to a
!     permutation which puts the matrix in block lower
!     triangular form.
! ib(i) is the position in the ordering of the start of the ith
!     block.  ib(n+1-i) holds the node number of the ith node
!     on the stack.
! lowl(i) is the smallest stack position of any node to which a path
!     from node i has been found.  it is set to n+1 when node i
!     is removed from the stack.
! numb(i) is the position of node i in the stack if it is on
!     it, is the permuted order of node i for those nodes
!     whose final position has been found and is otherwise zero.
! prev(i) is the node at the end of the path when node i was
!     placed on the stack.
      integer icn(licn),lenr(n),arp(n),ib(n),lowl(n),numb(n), &
     prev(n)

!   icnt is the number of nodes whose positions in final ordering have
!     been found.
      icnt=0
! num is the number of blocks that have been found.
      num=0
      nnm1=n+n-1

! initialization of arrays.
      do 20 j=1,n
      numb(j)=0
      arp(j)=lenr(j)-1
   20 continue


      do 120 isn=1,n
! look for a starting node
      if (numb(isn).ne.0) go to 120
      iv=isn
! ist is the number of nodes on the stack ... it is the stack pointer.
      ist=1
! put node iv at beginning of stack.
      lowl(iv)=1
      numb(iv)=1
      ib(n)=iv

! the body of this loop puts a new node on the stack or backtracks.
      do 110 dummy=1,nnm1
      i1=arp(iv)
! have all edges leaving node iv been searched.
      if (i1.lt.0) go to 60
      i2=ip(iv)+lenr(iv)-1
      i1=i2-i1

! look at edges leaving node iv until one enters a new node or
!     all edges are exhausted.
      do 50 ii=i1,i2
      iw=icn(ii)
! has node iw been on stack already.
      if (numb(iw).eq.0) go to 100
! update value of lowl(iv) if necessary.
  50  lowl(iv)=min0(lowl(iv),lowl(iw))

! there are no more edges leaving node iv.
      arp(iv)=-1
! is node iv the root of a block.
   60 if (lowl(iv).lt.numb(iv)) go to 90

! order nodes in a block.
      num=num+1
      ist1=n+1-ist
      lcnt=icnt+1
! peel block off the top of the stack starting at the top and
!     working down to the root of the block.
      do 70 stp=ist1,n
      iw=ib(stp)
      lowl(iw)=n+1
      icnt=icnt+1
      numb(iw)=icnt
      if (iw.eq.iv) go to 80
   70 continue
   80 ist=n-stp
      ib(num)=lcnt
! are there any nodes left on the stack.
      if (ist.ne.0) go to 90
! have all the nodes been ordered.
      if (icnt.lt.n) go to 120
      go to 130

! backtrack to previous node on path.
   90 iw=iv
      iv=prev(iv)
! update value of lowl(iv) if necessary.
      lowl(iv)=min0(lowl(iv),lowl(iw))
      go to 110

! put new node on the stack.
 100  arp(iv)=i2-ii-1
      prev(iw)=iv
      iv=iw
      ist=ist+1
      lowl(iv)=ist
      numb(iv)=ist
      k=n+1-ist
      ib(k)=iv
  110 continue

  120 continue

! put permutation in the required form.
  130 do 140 i=1,n
      ii=numb(i)
 140  arp(ii)=i
      return
      end subroutine

!######date   01 jan 1984     copyright ukaea, harwell.
!######alias mc20ad mc20bd
      subroutine mc20ad(nc,maxa,a,inum,jptr,jnum,jdisp)

      integer   inum(maxa),jnum(maxa)
      real(double) a(maxa),ace,acep
      dimension jptr(nc)

!     ******************************************************************
      null=-jdisp
!**      clear jptr
      do 60 j=1,nc
   60 jptr(j)=0
!**      count the number of elements in each column.
      do 120 k=1,maxa
      j=jnum(k)+jdisp
      jptr(j)=jptr(j)+1
  120 continue
!**      set the jptr array
      k=1
      do 150 j=1,nc
      kr=k+jptr(j)
      jptr(j)=k
  150 k=kr

!**      reorder the elements into column order.  the algorithm is an
!        in-place sort and is of order maxa.
      do 230 i=1,maxa
!        establish the current entry.
      jce=jnum(i)+jdisp
      if(jce.eq.0) go to 230
      ace=a(i)
      ice=inum(i)
!        clear the location vacated.
      jnum(i)=null
!        chain from current entry to store items.
      do 200 j=1,maxa
!        current entry not in correct position.  determine correct
!        position to store entry.
      loc=jptr(jce)
      jptr(jce)=jptr(jce)+1
!        save contents of that location.
      acep=a(loc)
      icep=inum(loc)
      jcep=jnum(loc)
!        store current entry.
      a(loc)=ace
      inum(loc)=ice
      jnum(loc)=null
!        check if next current entry needs to be processed.
      if(jcep.eq.null) go to 230
!        it does.  copy into current entry.
      ace=acep
      ice=icep
  200 jce=jcep+jdisp

  230 continue

!**      reset jptr vector.
      ja=1
      do 250 j=1,nc
      jb=jptr(j)
      jptr(j)=ja
  250 ja=jb
      return
      end subroutine

!######date   01 jan 1984     copyright ukaea, harwell.
!######alias mc21a
      subroutine mc21a(n,icn,licn,ip,lenr,iperm,numnz,iw)
      integer ip(n)
      integer icn(licn),lenr(n),iperm(n),iw(n,4)
      call mc21b(n,icn,licn,ip,lenr,iperm,numnz,iw(1,1),iw(1,2),iw(1,3), &
      iw(1,4))
      return
      end subroutine

      subroutine mc21b(n,icn,licn,ip,lenr,iperm,numnz,pr,arp,cv,out)
      integer ip(n)
!   pr(i) is the previous row to i in the depth first search.
! it is used as a work array in the sorting algorithm.
!   elements (iperm(i),i) i=1, ... n  are non-zero at the end of the
! algorithm unless n assignments have not been made.  in which case
! (iperm(i),i) will be zero for n-numnz entries.
!   cv(i) is the most recent row extension at which column i
! was visited.
!   arp(i) is one less than the number of non-zeros in row i
! which have not been scanned when looking for a cheap assignment.
!   out(i) is one less than the number of non-zeros in row i
! which have not been scanned during one pass through the main loop.
      integer icn(licn),lenr(n),iperm(n),pr(n),cv(n), &
      arp(n),out(n)

!   initialization of arrays.
      do 10 i=1,n
      arp(i)=lenr(i)-1
      cv(i)=0
   10 iperm(i)=0
      numnz=0

!   main loop.
!   each pass round this loop either results in a new assignment
! or gives a row with no assignment.
      do 130 jord=1,n
      j=jord
      pr(j)=-1
      do 100 k=1,jord
! look for a cheap assignment
      in1=arp(j)
      if (in1.lt.0) go to 60
      in2=ip(j)+lenr(j)-1
      in1=in2-in1
      do 50 ii=in1,in2
      i=icn(ii)
      if (iperm(i).eq.0) go to 110
   50 continue
!   no cheap assignment in row.
      arp(j)=-1
!   begin looking for assignment chain starting with row j.
   60 out(j)=lenr(j)-1
! inner loop.  extends chain by one or backtracks.
      do 90 kk=1,jord
      in1=out(j)
      if (in1.lt.0) go to 80
      in2=ip(j)+lenr(j)-1
      in1=in2-in1
! forward scan.
      do 70 ii=in1,in2
      i=icn(ii)
      if (cv(i).eq.jord) go to 70
!   column i has not yet been accessed during this pass.
      j1=j
      j=iperm(i)
      cv(i)=jord
      pr(j)=j1
      out(j1)=in2-ii-1
      go to 100
   70 continue

!   backtracking step.
   80 j=pr(j)
      if (j.eq.-1) go to 130
   90 continue

  100 continue

!   new assignment is made.
  110 iperm(i)=j
      arp(j)=in2-ii-1
      numnz=numnz+1
      do 120 k=1,jord
      j=pr(j)
      if (j.eq.-1) go to 130
      ii=ip(j)+lenr(j)-out(j)-2
      i=icn(ii)
      iperm(i)=j
  120 continue

  130 continue

!   if matrix is structurally singular, we now complete the
! permutation iperm.
      if (numnz.eq.n) return
      do 140 i=1,n
  140 arp(i)=0
      k=0
      do 160 i=1,n
      if (iperm(i).ne.0) go to 150
      k=k+1
      out(k)=i
      go to 160
  150 j=iperm(i)
      arp(j)=i
  160 continue
      k=0
      do 170 i=1,n
      if (arp(i).ne.0) go to 170
      k=k+1
      ioutk=out(k)
      iperm(ioutk)=i
  170 continue
      return
      end subroutine

!######date   01 jan 1984     copyright ukaea, harwell.
!######alias mc22ad
      subroutine mc22ad(n,icn,a,nz,lenrow,ip,iq,iw,iw1)
      real(double) a(nz),aval
      integer iw(n,2)
      integer   icn(nz),lenrow(n),ip(n),iq(n),iw1(nz)
      if (nz.le.0) go to 1000
      if (n.le.0) go to 1000
! set start of row i in iw(i,1) and lenrow(i) in iw(i,2)
      iw(1,1)=1
      iw(1,2)=lenrow(1)
      do 10 i=2,n
      iw(i,1)=iw(i-1,1)+lenrow(i-1)
 10   iw(i,2)=lenrow(i)
! permute lenrow according to ip.  set off-sets for new position
!     of row iold in iw(iold,1) and put old row indices in iw1 in
!     positions corresponding to the new position of this row in a/icn.
      jj=1
      do 20 i=1,n
      iold=ip(i)
      iold=iabs(iold)
      length=iw(iold,2)
      lenrow(i)=length
      if (length.eq.0) go to 20
      iw(iold,1)=iw(iold,1)-jj
      j2=jj+length-1
      do 15 j=jj,j2
 15   iw1(j)=iold
      jj=j2+1
 20   continue
! set inverse permutation to iq in iw(.,2).
      do 30 i=1,n
      iold=iq(i)
      iold=iabs(iold)
 30   iw(iold,2)=i
! permute a and icn in place, changing to new column numbers.

! ***   main loop   ***
! each pass through this loop places a closed chain of column indices
!     in their new (and final) positions ... this is recorded by
!     setting the iw1 entry to zero so that any which are subsequently
!     encountered during this major scan can be bypassed.
      do 200 i=1,nz
      iold=iw1(i)
      if (iold.eq.0) go to 200
      ipos=i
      jval=icn(i)
! if row iold is in same positions after permutation go to 150.
      if (iw(iold,1).eq.0) go to 150
      aval=a(i)
! **  chain loop  **
! each pass through this loop places one (permuted) column index
!     in its final position  .. viz. ipos.
      do 100 ichain=1,nz
! newpos is the original position in a/icn of the element to be placed
! in position ipos.  it is also the position of the next element in
!     the chain.
      newpos=ipos+iw(iold,1)
! is chain complete ?
      if (newpos.eq.i) go to 130
      a(ipos)=a(newpos)
      jnum=icn(newpos)
      icn(ipos)=iw(jnum,2)
      ipos=newpos
      iold=iw1(ipos)
      iw1(ipos)=0
! **  end of chain loop  **
 100  continue
 130  a(ipos)=aval
 150  icn(ipos)=iw(jval,2)
! ***   end of main loop   ***
 200  continue

 1000 return
      end subroutine

!######date   01 jan 1984     copyright ukaea, harwell.
!######alias mc23ad
!###### calls   mc13    mc21
      subroutine mc23ad(n,icn,a,licn,lenr,idisp,ip,iq,lenoff,iw,iw1)
      real(double) a(licn)
      integer idisp(2),iw1(n,2)
      logical abort
      integer   icn(licn),lenr(n),ip(n),iq(n),lenoff(n),iw(n,5)
      common /mc23bd/ lp,numnz,num,large,abort
! input ... n,icn .. a,icn,lenr ....

! set up pointers iw(.,1) to the beginning of the rows and set lenoff
!     equal to lenr.
      iw1(1,1)=1
      lenoff(1)=lenr(1)
      if (n.eq.1) go to 20
      do 10 i=2,n
      lenoff(i)=lenr(i)
   10 iw1(i,1)=iw1(i-1,1)+lenr(i-1)
! idisp(1) points to the first position in a/icn after the
!     off-diagonal blocks and untreated rows.
   20 idisp(1)=iw1(n,1)+lenr(n)

! find row permutation ip to make diagonal zero-free.
      call mc21a(n,icn,licn,iw1,lenr,ip,numnz,iw)

! possible error return for structurally singular matrices.
      if (numnz.ne.n.and.abort) go to 170

! iw1(.,2) and lenr are permutations of iw1(.,1) and lenr/lenoff
!     suitable for entry
!     to mc13d since matrix with these row pointer and length arrays
!     has maximum number of non-zeros on the diagonal.
      do 30 ii=1,n
      i=ip(ii)
      iw1(ii,2)=iw1(i,1)
   30 lenr(ii)=lenoff(i)

! find symmetri! permutation iq to block lower triangular form.
      call mc13d(n,icn,licn,iw1(1,2),lenr,iq,iw(1,4),num,iw)

      if (num.ne.1) go to 60

! action taken if matrix is irreducible.
! whole matrix is just moved to the end of the storage.
      do 40 i=1,n
      lenr(i)=lenoff(i)
      ip(i)=i
   40 iq(i)=i
      lenoff(1)=-1
! idisp(1) is the first position after the last element in the
!     off-diagonal blocks and untreated rows.
      nz=idisp(1)-1
      idisp(1)=1
! idisp(2) is the position in a/icn of the first element in the
!     diagonal blocks.
      idisp(2)=licn-nz+1
      large=n
      if (nz.eq.licn) go to 230
      do 50 k=1,nz
      j=nz-k+1
      jj=licn-k+1
      a(jj)=a(j)
   50 icn(jj)=icn(j)
! 230 = return
      go to 230

! data structure reordered.

! form composite row permutation ... ip(i) = ip(iq(i)).
   60 do 70 ii=1,n
      i=iq(ii)
   70 iw(ii,1)=ip(i)
      do 80 i=1,n
   80 ip(i)=iw(i,1)

! run through blocks in reverse order separating diagonal blocks
!     which are moved to the end of the storage.  elements in
!     off-diagonal blocks are left in place unless a compress is
!     necessary.

! ibeg indicates the lowest value of j for which icn(j) has been
!     set to zero when element in position j was moved to the
!     diagonal block part of storage.
      ibeg=licn+1
! iend is the position of the first element of those treated rows
!     which are in diagonal blocks.
      iend=licn+1
! large is the dimension of the largest block encountered so far.
      large=0

! num is the number of diagonal blocks.
      do 150 k=1,num
      iblock=num-k+1
! i1 is first row (in permuted form) of block iblock.
! i2 is last row (in permuted form) of block iblock.
      i1=iw(iblock,4)
      i2=n
      if (k.ne.1) i2=iw(iblock+1,4)-1
      large=max0(large,i2-i1+1)
! go through the rows of block iblock in the reverse order.
      do 140 ii=i1,i2
      inew=i2-ii+i1
! we now deal with row inew in permuted form (row iold in original
!     matrix).
      iold=ip(inew)
! if there is space to move up diagonal block portion of row go to 110
      if (iend-idisp(1).ge.lenoff(iold)) go to 110

! in-line compress.
! moves separated off-diagonal elements and untreated rows to
!     front of storage.
      jnpos=ibeg
      ilend=idisp(1)-1
      if (ilend.lt.ibeg) go to 190
      do 90 j=ibeg,ilend
      if (icn(j).eq.0) go to 90
      icn(jnpos)=icn(j)
      a(jnpos)=a(j)
      jnpos=jnpos+1
   90 continue
      idisp(1)=jnpos
      if (iend-jnpos.lt.lenoff(iold)) go to 190
      ibeg=licn+1
! reset pointers to the beginning of the rows.
      do 100 i=2,n
  100 iw1(i,1)=iw1(i-1,1)+lenoff(i-1)

! row iold is now split into diag. and off-diag. parts.
  110 irowb=iw1(iold,1)
      leni=0
      irowe=irowb+lenoff(iold)-1
! backward scan of whole of row iold (in original matrix).
      if (irowe.lt.irowb) go to 130
      do 120 jj=irowb,irowe
      j=irowe-jj+irowb
      jold=icn(j)
! iw(.,2) holds the inverse permutation to iq.
!     ..... it was set to this in mc13d.
      jnew=iw(jold,2)
! if (jnew.lt.i1) then ....
! element is in off-diagonal block and so is left in situ.
      if (jnew.lt.i1) go to 120
! element is in diagonal block and is moved to the end of the storage.
      iend=iend-1
      a(iend)=a(j)
      icn(iend)=jnew
      ibeg=min0(ibeg,j)
      icn(j)=0
      leni=leni+1
  120 continue

      lenoff(iold)=lenoff(iold)-leni
  130 lenr(inew)=leni
  140 continue

      ip(i2)=-ip(i2)
  150 continue
! resets ip(n) to positive value.
      ip(n)=-ip(n)
! idisp(2) is position of first element in diagonal blocks.
      idisp(2)=iend

! this compress is used to move all off-diagonal elements to the
!     front of the storage.
      if (ibeg.gt.licn) go to 230
      jnpos=ibeg
      ilend=idisp(1)-1
      do 160 j=ibeg,ilend
      if (icn(j).eq.0) go to 160
      icn(jnpos)=icn(j)
      a(jnpos)=a(j)
      jnpos=jnpos+1
  160 continue
! idisp(1) is first position after last element of off-diagonal blocks.
      idisp(1)=jnpos
      go to 230

! error return
  170 if (lp.ne.0) write(lp,180) numnz
  180 format(33x,41h matrix is structurally singular, rank = ,i6)
      idisp(1)=-1
      go to 210
  190 if (lp.ne.0) write(lp,200) n
  200 format(33x,33h licn not big enough increase by ,i6)
      idisp(1)=-2
  210 if (lp.ne.0) write(lp,220)
  220 format(33h+error return from mc23ad because)

  230 return
      end subroutine

!######date   01 jan 1984     copyright ukaea, harwell.
!######alias mc24ad
      subroutine mc24ad(n,icn,a,licn,lenr,lenrl,w)
      real(double) a(licn),w(n),amaxl,wrowl,amaxu,zero
      integer   icn(licn),lenr(n),lenrl(n)
      data zero/0.0d0/
      amaxl=zero
      do 10 i=1,n
 10   w(i)=zero
      j0=1
      do 100 i=1,n
      if (lenr(i).eq.0) go to 100
      j2=j0+lenr(i)-1
      if (lenrl(i).eq.0) go to 50
! calculation of 1-norm of l.
      j1=j0+lenrl(i)-1
      wrowl=zero
      do 30 jj=j0,j1
 30   wrowl=wrowl+dabs(a(jj))
! amaxl is the maximum norm of columns of l so far found.
      amaxl=dmax1(amaxl,wrowl)
      j0=j1+1
! calculation of norms of columns of u (max-norms).
 50   j0=j0+1
      if (j0.gt.j2) go to 90
      do 80 jj=j0,j2
      j=icn(jj)
 80   w(j)=dmax1(dabs(a(jj)),w(j))
 90   j0=j2+1
 100  continue
! amaxu is set to maximum max-norm of columns of u.
      amaxu=zero
      do 200 i=1,n
 200  amaxu=dmax1(amaxu,w(i))
! grofac is max u max-norm times max l 1-norm.
      w(1)=amaxl*amaxu
      return
      end subroutine

      subroutine ma28cd(n, a, licn, icn, ikeep, rhs, w, mtype)

! this subroutine uses the factors from ma28a/ad or ma28b/bd to
!     solve a system of equations without iterative refinement.
! the parameters are ...
! n   integer  order of matrix  not altered by subroutine.
! a   real/real(double) array  length licn.  the same array as
!     was used in the most recent call to ma28a/ad or ma28b/bd.
! licn  integer  length of arrays a and icn.  not altered by
!     subroutine.
! icn    integer array of length licn.  same array as output from
!     ma28a/ad.  unchanged by ma28c/cd.
! ikeep  integer array of length 5*n.  same array as output from
!     ma28a/ad.  unchanged by ma28c/cd.
! rhs    real/real(double) array  length n.  on entry, it holds the
!     right hand side.  on exit, the solution vector.
! w      real/real(double) array  length n. used as workspace by
!     ma30c/cd.
! mtype  integer  used to tell ma30c/cd to solve the direct equation
!     (mtype=1) or its transpose (mtype.ne.1).
      
      real(double) a(licn), rhs(n), w(n), resid, mresid, eps, rmin
      integer idisp(2)
      integer icn(licn), ikeep(n,5)
      logical abort1, abort2
! common block variables.
! unless otherwise stated common block variables are as in ma28a/ad.
!     those variables referenced by ma28c/cd are mentioned below.
! resid  real/real(double)  variable returns maximum residual of
!     equations where pivot was zero.
! mresid  real/real(double) variable used by ma28c/cd to
!     communicate between ma28f/fd and ma30h/hd.
! idisp  integer array  length 2  the same as that used by ma28a/ad.
!     it is unchanged by ma28b/bd.

! further information on common block variables can be found in block
!     data or ma28a/ad.
      common /ma28fd/ eps, rmin, resid, irncp, icncp, minirn, minicn,  &
      irank, abort1, abort2
      common /ma28gd/ idisp
      common /ma30hd/ mresid

! this call performs the solution of the set of equations.
      call ma30cd(n, icn, a, licn, ikeep, ikeep(1,4), ikeep(1,5),  &
      idisp, ikeep(1,2), ikeep(1,3), rhs, w, mtype)
! transfer common block information.
      resid = mresid
      return
      end subroutine

      subroutine ma30cd(n, icn, a, licn, lenr, lenrl, lenoff, idisp, ip,  &
      iq, x, w, mtype)
! ma30c/cd uses the factors produced by ma30a/ad or ma30b/bd to solve
!     ax=b or a transpose x=b when the matrix p1*a*q1 (paq) is block
!     lower triangular (including the case of only one diagonal
!     block).

! we now describe the argument list for ma30c/cd.
! n  is an integer variable set to the order of the matrix. it is not
!     altered by the subroutine.
! icn is an integer array of length licn. entries idisp(1) to
!     idisp(2) should be unchanged since the last call to ma30a/ad. if
!     the matrix has more than one diagonal block, then column indices
!     corresponding to non-zeros in sub-diagonal blocks of paq must
!     appear in positions 1 to idisp(1)-1. for the same row those
!     entries must be contiguous, with those in row i preceding those
!     in row i+1 (i=1,...,n-1) and no wasted space between rows.
!     entries may be in any order within each row. it is not altered
!     by ma30c/cd.
! a  is a real/real(double) array of length licn.  entries
!     idisp(1) to idisp(2) should be unchanged since the last call to
!     ma30a/ad or ma30b/bd.  if the matrix has more than one diagonal
!     block, then the values of the non-zeros in sub-diagonal blocks
!     must be in positions 1 to idisp(1)-1 in the order given by icn.
!     it is not altered by ma30c/cd.
! licn  is an integer variable set to the size of arrays icn and a.
!     it is not altered by ma30c/cd.
! lenr,lenrl are integer arrays of length n which should be
!     unchanged since the last call to ma30a/ad. they are not altered
!     by ma30c/cd.
! lenoff  is an integer array of length n. if the matrix paq (or
!     p1*a*q1) has more than one diagonal block, then lenoff(i),
!     i=1,...,n should be set to the number of non-zeros in row i of
!     the matrix paq which are in sub-diagonal blocks.  if there is
!     only one diagonal block then lenoff(1) may be set to -1, in
!     which case the other entries of lenoff are never accessed. it is
!     not altered by ma30c/cd.
! idisp  is an integer array of length 2 which should be unchanged
!     since the last call to ma30a/ad. it is not altered by ma30c/cd.
! ip,iq are integer arrays of length n which should be unchanged
!     since the last call to ma30a/ad. they are not altered by
!     ma30c/cd.
! x is a real/real(double) array of length n. it must be set by
!     the user to the values of the right hand side vector b for the
!     equations being solved.  on exit from ma30c/cd it will be equal
!     to the solution x required.
! w  is a real/real(double) array of length n which is used as
!     workspace by ma30c/cd.
! mtype is an integer variable which must be set by the user. if
!     mtype=1, then the solution to the system ax=b is returned; any
!     other value for mtype will return the solution to the system a
!     transpose x=b. it is not altered by ma30c/cd.

      real(double) a(licn), x(n), w(n), wii, wi, resid, zero
      logical neg, nobloc
      integer idisp(2)
      integer icn(licn), lenr(n), lenrl(n), lenoff(n), ip(n), iq(n)
! see block data for comments on variables in common.
      common /ma30hd/ resid
      data zero /0.0d0/

! the final value of resid is the maximum residual for an inconsistent
!     set of equations.
      resid = zero
! nobloc is .true. if subroutine block has been used previously and
!     is .false. otherwise.  the value .false. means that lenoff
!     will not be subsequently accessed.
      nobloc = lenoff(1).lt.0
      if (mtype.ne.1) go to 140

! we now solve   a * x = b.
! neg is used to indicate when the last row in a block has been
!     reached.  it is then set to true whereafter backsubstitution is
!     performed on the block.
      neg = .false.
! ip(n) is negated so that the last row of the last block can be
!     recognised.  it is reset to its positive value on exit.
      ip(n) = -ip(n)
! preorder vector ... w(i) = x(ip(i))
      do 10 ii=1,n
        i = ip(ii)
        i = iabs(i)
        w(ii) = x(i)
   10 continue
! lt holds the position of the first non-zero in the current row of the
!     off-diagonal blocks.
      lt = 1
! ifirst holds the index of the first row in the current block.
      ifirst = 1
! iblock holds the position of the first non-zero in the current row
!     of the lu decomposition of the diagonal blocks.
      iblock = idisp(1)
! if i is not the last row of a block, then a pass through this loop
!     adds the inner product of row i of the off-diagonal blocks and w
!     to w and performs forward elimination using row i of the lu
!     decomposition.   if i is the last row of a block then, after
!     performing these aforementioned operations, backsubstitution is
!     performed using the rows of the block.
      do 120 i=1,n
        wi = w(i)
        if (nobloc) go to 30
        if (lenoff(i).eq.0) go to 30
! operations using lower triangular blocks.
! ltend is the end of row i in the off-diagonal blocks.
        ltend = lt + lenoff(i) - 1
        do 20 jj=lt,ltend
          j = icn(jj)
          wi = wi - a(jj)*w(j)
   20   continue
! lt is set the beginning of the next off-diagonal row.
        lt = ltend + 1
! set neg to .true. if we are on the last row of the block.
   30   if (ip(i).lt.0) neg = .true.
        if (lenrl(i).eq.0) go to 50
! forward elimination phase.
! iend is the end of the l part of row i in the lu decomposition.
        iend = iblock + lenrl(i) - 1
        do 40 jj=iblock,iend
          j = icn(jj)
          wi = wi + a(jj)*w(j)
   40   continue
! iblock is adjusted to point to the start of the next row.
   50   iblock = iblock + lenr(i)
        w(i) = wi
        if (.not.neg) go to 120
! back substitution phase.
! j1 is position in a/icn after end of block beginning in row ifirst
!     and ending in row i.
        j1 = iblock
! are there any singularities in this block?  if not, continue with
!     the backsubstitution.
        ib = i
        if (iq(i).gt.0) go to 70
        do 60 iii=ifirst,i
          ib = i - iii + ifirst
          if (iq(ib).gt.0) go to 70
          j1 = j1 - lenr(ib)
          resid = dmax1(resid,dabs(w(ib)))
          w(ib) = zero
   60   continue
! entire block is singular.
        go to 110
! each pass through this loop performs the back-substitution
!     operations for a single row, starting at the end of the block and
!     working through it in reverse order.
   70   do 100 iii=ifirst,ib
          ii = ib - iii + ifirst
! j2 is end of row ii.
          j2 = j1 - 1
! j1 is beginning of row ii.
          j1 = j1 - lenr(ii)
! jpiv is the position of the pivot in row ii.
          jpiv = j1 + lenrl(ii)
          jpivp1 = jpiv + 1
! jump if row  ii of u has no non-zeros.
          if (j2.lt.jpivp1) go to 90
          wii = w(ii)
          do 80 jj=jpivp1,j2
            j = icn(jj)
            wii = wii - a(jj)*w(j)
   80     continue
          w(ii) = wii
   90     w(ii) = w(ii)/a(jpiv)
  100   continue
  110   ifirst = i + 1
        neg = .false.
  120 continue

! reorder solution vector ... x(i) = w(iqinverse(i))
      do 130 ii=1,n
        i = iq(ii)
        i = iabs(i)
        x(i) = w(ii)
  130 continue
      ip(n) = -ip(n)
      go to 320

! we now solve   atranspose * x = b.
! preorder vector ... w(i)=x(iq(i))
  140 do 150 ii=1,n
        i = iq(ii)
        i = iabs(i)
        w(ii) = x(i)
  150 continue
! lj1 points to the beginning the current row in the off-diagonal
!     blocks.
      lj1 = idisp(1)
! iblock is initialized to point to the beginning of the block after
!     the last one ]
      iblock = idisp(2) + 1
! ilast is the last row in the current block.
      ilast = n
! iblend points to the position after the last non-zero in the
!     current block.
      iblend = iblock
! each pass through this loop operates with one diagonal block and
!     the off-diagonal part of the matrix corresponding to the rows
!     of this block.  the blocks are taken in reverse order and the
!     number of times the loop is entered is min(n,no. blocks+1).
      do 290 numblk=1,n
        if (ilast.eq.0) go to 300
        iblock = iblock - lenr(ilast)
! this loop finds the index of the first row in the current block..
!     it is first and iblock is set to the position of the beginning
!     of this first row.
        do 160 k=1,n
          ii = ilast - k
          if (ii.eq.0) go to 170
          if (ip(ii).lt.0) go to 170
          iblock = iblock - lenr(ii)
  160   continue
  170   ifirst = ii + 1
! j1 points to the position of the beginning of row i (lt part) or pivot
        j1 = iblock
! forward elimination.
! each pass through this loop performs the operations for one row of the
!     block.  if the corresponding entry of w is zero then the
!     operations can be avoided.
        do 210 i=ifirst,ilast
          if (w(i).eq.zero) go to 200
! jump if row i singular.
          if (iq(i).lt.0) go to 220
! j2 first points to the pivot in row i and then is made to point to the
!     first non-zero in the u transpose part of the row.
          j2 = j1 + lenrl(i)
          wi = w(i)/a(j2)
          if (lenr(i)-lenrl(i).eq.1) go to 190
          j2 = j2 + 1
! j3 points to the end of row i.
          j3 = j1 + lenr(i) - 1
          do 180 jj=j2,j3
            j = icn(jj)
            w(j) = w(j) - a(jj)*wi
  180     continue
  190     w(i) = wi
  200     j1 = j1 + lenr(i)
  210   continue
        go to 240
! deals with rest of block which is singular.
  220   do 230 ii=i,ilast
          resid = dmax1(resid,dabs(w(ii)))
          w(ii) = zero
  230   continue
! back substitution.
! this loop does the back substitution on the rows of the block in
!     the reverse order doing it simultaneously on the l transpose part
!     of the diagonal blocks and the off-diagonal blocks.
  240   j1 = iblend
        do 280 iback=ifirst,ilast
          i = ilast - iback + ifirst
! j1 points to the beginning of row i.
          j1 = j1 - lenr(i)
          if (lenrl(i).eq.0) go to 260
! j2 points to the end of the l transpose part of row i.
          j2 = j1 + lenrl(i) - 1
          do 250 jj=j1,j2
            j = icn(jj)
            w(j) = w(j) + a(jj)*w(i)
  250     continue
  260     if (nobloc) go to 280
! operations using lower triangular blocks.
          if (lenoff(i).eq.0) go to 280
! lj2 points to the end of row i of the off-diagonal blocks.
          lj2 = lj1 - 1
! lj1 points to the beginning of row i of the off-diagonal blocks.
          lj1 = lj1 - lenoff(i)
          do 270 jj=lj1,lj2
            j = icn(jj)
            w(j) = w(j) - a(jj)*w(i)
  270     continue
  280   continue
        iblend = j1
        ilast = ifirst - 1
  290 continue
! reorder solution vector ... x(i)=w(ipinverse(i))
  300 do 310 ii=1,n
        i = ip(ii)
        i = iabs(i)
        x(i) = w(ii)
  310 continue

  320 return
      end subroutine

      end module mod_ma28

