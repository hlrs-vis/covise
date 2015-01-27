!=====================================================================
!
!	=============== 12.1.1995 ===================================
!
!	MAKE SURE THAT ALL VECTORS ARE INITIALISED
!	CORRECTLY
!
!	=============== 12.1.1995 ===================================
!
!	This programme is intended to calculate the flow of
!	a viscoelastic fluid in a symmetric 2D channel.
!	The mesh is made up of rectangular "superelements".
!
!	The equation that has to be solved
!
!	 n+1          -1    n                    -1 n+1
!	q   = [I-C(AC)  A][q - dtB + dtR] + C(AC)  r   
!
!	where
!
!	C	Gradient Matrix
!	A	Divergence Matrix
!	q	Vector of all unknowns (u,v,sigma,tau,gamma)
!	R	Right hand side of stress equations
!	r	Right hand side of boundary calculation
!
!	Define two auxiliary vectors:
!
!	      n     n
!	S := q + dtR  - dtB
!
!		    n+1
!	T := A S - r
!
!	Now the calculation can be done in two steps
!
!	Step 1:
!
!		AC U = T
!
!	Step 2:
!
!		 n+1 
!		q   = S - C U 
!
!
!	Original author: Michael Resch (MR)
!	First released: 5. January 1995
!
!=====================================================================
	subroutine visco_test(fname,num_timesteps)
        use visco_declaration
	use visco_init
        use visco_init_1
	use visco_kernel
        use visco_bq
	use visco_control
        use arithmetic
        use mod_ma28
        use visco_smooth

	implicit none
        character *(*) :: fname
        integer :: num_timesteps

        interface getrf
          subroutine sgetrf(m,n,a,lda,ipiv,info)
            integer     :: m,n,lda,ipiv(m)
            real        :: a(lda,n)
          end subroutine
          subroutine dgetrf(m,n,a,lda,ipiv,info)
            use declaration
            integer     :: m,n,lda,ipiv(m)
            real(double) :: a(lda,n)
          end subroutine
        end interface

        interface getrs
          subroutine sgetrs(trans,n,nrhs,a,lda,ipiv,b,ldb,info)
            character*1 :: trans
            integer     :: n,nrhs,lda,ipiv(n),ldb,info
            real        :: b(*)
            real        :: a(lda,n)
          end subroutine
          subroutine dgetrs(trans,n,nrhs,a,lda,ipiv,b,ldb,info)
            use declaration
            character*1 :: trans
            integer     :: n,nrhs,lda,ipiv(n),ldb,info
            real(double)        :: b(*)
            real(double)        :: a(lda,n)
          end subroutine
        end interface

	integer		:: time_step,i
	integer, dimension(8) :: times_1,times_2


          print*,'FILENAME',fname,'FILENAME'
	open(70,file='/vobs/covise/src/application/rus/MIR/test.mesh')
!=====================================================================
!	read mesh information and calculation parameters
!=====================================================================
        call visco_read_data()
        call date_and_time(date=today,values=times_1)
!        call visco_get_mesh()
       call visco_get_neighbours()
!=====================================================================
!	allocate memory for the matrices  and vectors needed
!=====================================================================
        call visco_init_A_1()
        call visco_init_C_1        
        AC2=A2*C2
!=====================================================================
!	correct q if it is not divergence free!!
!=====================================================================
 	call visco_correct_q()
 	call visco_calc_Aq()

!=====================================================================
!	initialise the smoothing operator
!=====================================================================
	call visco_init_smoother()
!=====================================================================
!	find an estimate for initial values of velocities
!=====================================================================
	timeloop: do time_step = 1,num_timesteps
          print*,time_step,'TS'
!=====================================================================
!	B depends on q and thus has to be recalculated for each
!	time step.
!=====================================================================
          call visco_calc_Bq()
!=====================================================================
!	Calculate R (the right hand side of stress equations)
!=====================================================================
          call visco_calc_R()
!=====================================================================
!	Step 1: Solve for AC U = T
!=====================================================================
!=====================================================================
!	Calculate S := q + dtR - dtB
!=====================================================================
          call visco_calc_S()
!=====================================================================
!	Calculate T := A S - r
!=====================================================================
          call visco_calc_T()
!=====================================================================
!       This LAPACK routine solves for factorized AC
!	to get the vector U
!=====================================================================
          call ma28c(dim,AC2,T,pivot,info)
          if (info /= 0) then
            write(*,*) 'info from solve ma28c ',info
            STOP
          end if
          U = T
          U2 = U
!=====================================================================
!	before calcultaing pressure smooth results!!!
!=====================================================================
          call visco_smoother()
          U = T
          U2 = U
          p = U/delta_t
          p2 = p
!=====================================================================
!	calculate the new field of velocities and stresses
!	q    = S - C U
!=====================================================================
          call visco_calc_q()
          call visco_distribute_results()
!=====================================================================
!       update A,C and AC-matrix according to
!       velocity field!!
!=====================================================================
         !! if (mod(time_step,1000) == 0 .and.  &
	 !!  time_step > 100)   then
!          if ((mod(time_step,50) == 0) .and. (time_step > 200))  & 
!		   then
!            call visco_check_alpha()
!            if (check == 1) then
!              call visco_init_A_1()
!              call visco_init_C_1
!              AC2=A2*C2
!            end if
!          end if
           call covise_update(time_step,dim)
        end do timeloop

	call visco_write_results()
        call visco_write_vectors()

        call date_and_time(values=times_2)
        write(*,*) ((times_2(i) - times_1(i)),i=5,8)
        close(70)
	end subroutine visco_test

!=====================================================================
	subroutine visco_get_vectors(x_coord,y_coord,vel_u,vel_v,vl,num_c,num_conn)
!=====================================================================
!	This subroutine writes all velocity vectors
!=====================================================================
	use visco_declaration
	implicit none
        real(double), dimension(dim) :: x_coord,y_coord,vel_u,vel_v
        integer, dimension(dim) :: vl
        integer      :: num_c,num_conn


	integer		:: i,n1,n2,n3
	num_conn = 0
	do i = 1,dim
          x_coord(i)=cell_data(i)%centre(1)
          y_coord(i)=cell_data(i)%centre(2)
          vel_u(i)=cell_data(i)%unknowns%velocities(1)
          vel_v(i)=cell_data(i)%unknowns%velocities(2)
          n1=cell_data(i)%neighbours(1)
          if (n1 > 0) then
             n2=cell_data(n1)%neighbours(2)
             if (n2 > 0) then
                n3=cell_data(n2)%neighbours(3)
                if (n3 > 0) then
                   if (cell_data(n3)%neighbours(4) == i) then
                       num_conn = num_conn+1
                       vl(num_conn)=i-1
                       num_conn = num_conn+1
                       vl(num_conn)=n1-1
                       num_conn = num_conn+1
                       vl(num_conn)=n2-1
                       num_conn = num_conn+1
                       vl(num_conn)=n3-1
                   end if
                end if
             end if
          end if
	end do
        num_c=dim;
	
	end subroutine visco_get_vectors
	  
