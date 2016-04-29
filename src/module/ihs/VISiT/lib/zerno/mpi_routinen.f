      SUBROUTINE MPI_INIT(ierr)
      integer ierr
      ierr=0        
      return
      end
      
      
      SUBROUTINE MPI_COMM_RANK(MPI_COMM_WORLD,myid,ierr)
      integer ierr,myid
      ierr=0       
      myid=0       
      return
      end
      
      SUBROUTINE MPI_COMM_SIZE(MPI_COMM_WORLD,numprocs,ierr)
      integer ierr,numprocs
      ierr=0
      numprocs=1
      return
      end
      
      SUBROUTINE MPI_FINALIZE(ierr)
      integer ierr
c     ierr=-1000
c     write(6,*)'Aufruf von MPI_FINALIZE '
c     write(6,*)'                   '
c     write(6,*)'Programm-Abbruch   '
c     if (ierr.eq.-1000) stop
      return
      end


      SUBROUTINE MPI_ABORT(MPI_COMM_WORLD,ierrcode,ierr)
      integer ierr,ierrcode
      ierr=0         
      ierrcode=0         
      stop
      return
      end

      
      SUBROUTINE MPI_SEND(rerr)
      real rerr
      integer ierr
      rerr=-1000
      ierr=-1000
      write(6,*)'Aufruf von MPI_SEND '
      write(6,*)'                   '
      write(6,*)'Programm-Abbruch   '
      if (ierr.eq.-1000) stop
      return
      end
      
      SUBROUTINE MPI_ISEND(rerr)
      real rerr
      integer ierr
      rerr=-1000
      ierr=-1000
      write(6,*)'Aufruf von MPI_ISEND '
      write(6,*)'                   '
      write(6,*)'Programm-Abbruch   '
      if (ierr.eq.-1000) stop
      return
      end
      
      SUBROUTINE MPI_RECV(rerr)
      real rerr
      integer ierr
      rerr=-1000
      ierr=-1000
      write(6,*)'Aufruf von MPI_RECV '
      write(6,*)'                   '
      write(6,*)'Programm-Abbruch   '
      if (ierr.eq.-1000) stop
      return
      end
      
      SUBROUTINE MPI_IRECV(rerr)
      real rerr
      integer ierr
      rerr=-1000
      ierr=-1000
      write(6,*)'Aufruf von MPI_IRECV '
      write(6,*)'                   '
      write(6,*)'Programm-Abbruch   '
      if (ierr.eq.-1000) stop
      return
      end
      
      SUBROUTINE MPI_REDUCE(rerr)
      real rerr
      integer ierr
      rerr=-1000
      ierr=-1000
      write(6,*)'Aufruf von MPI_REDUCE'
      write(6,*)'                   '
      write(6,*)'Programm-Abbruch   '
      if (ierr.eq.-1000) stop
      return
      end

      SUBROUTINE MPI_ALLREDUCE(rerr)
      real rerr
      integer ierr
      rerr=-1000
      ierr=-1000
      write(6,*)'Aufruf von MPI_REDUCE'
      write(6,*)'                   '
      write(6,*)'Programm-Abbruch   '
      if (ierr.eq.-1000) stop
      return
      end

      
      SUBROUTINE MPI_BCAST(rerr)
      real rerr
      integer ierr
      rerr=-1000
      ierr=-1000
      write(6,*)'Aufruf von MPI_BCAST'
      write(6,*)'                   '
      write(6,*)'Programm-Abbruch   '
      if (ierr.eq.-1000) stop
      return
      end
      
      SUBROUTINE MPI_GATHER(rerr)
      real rerr
      integer ierr
      rerr=-1000
      ierr=-1000
      write(6,*)'Aufruf von MPI_GATHER'
      write(6,*)'                   '
      write(6,*)'Programm-Abbruch   '
      if (ierr.eq.-1000) stop
      return
      end

      SUBROUTINE MPI_SCATTER(rerr)
      real rerr
      integer ierr
      rerr=-1000
      ierr=-1000
      write(6,*)'Aufruf von MPI_SCATTER'
      write(6,*)'                   '
      write(6,*)'Programm-Abbruch   '
      if (ierr.eq.-1000) stop
      return
      end

      SUBROUTINE MPI_BARRIER(rerr)
      real rerr
      integer ierr
      rerr=-1000
      ierr=-1000
      write(6,*)'Aufruf von MPI_BARRIER'
      write(6,*)'                   '
      write(6,*)'Programm-Abbruch   '
      if (ierr.eq.-1000) stop
      return
      end

      SUBROUTINE MPI_WAIT(rerr)
      real rerr
      integer ierr
      rerr=-1000
      ierr=-1000
      write(6,*)'Aufruf von MPI_WAIT'
      write(6,*)'                   '
      write(6,*)'Programm-Abbruch   '
      if (ierr.eq.-1000) stop
      return
      end
      
      function MPI_WTIME()
      double precision rerr,MPI_WTIME
      rerr=-1000.0
c     write(6,*)'Aufruf von MPI_WTIME'
c     write(6,*)'                   '
c     write(6,*)'Programm-Abbruch   '
      MPI_WTIME=rerr
      return
      end
      
      function MPI_WTICK()
      integer ierr,MPI_WTICK
      ierr=-1000
c     write(6,*)'Aufruf von MPI_WTICK'
c     write(6,*)'                   '
c     write(6,*)'Programm-Abbruch   '
      MPI_WTICK=ierr
      return
      end
      
      SUBROUTINE MPI_NULL_COPY_FN(ierr)
      integer ierr
      ierr=-1000
      write(6,*)'Aufruf von MPI_NULL_COPY_FN'
      write(6,*)'                   '
      write(6,*)'Programm-Abbruch   '
      if (ierr.eq.-1000) stop
      return
      end
      
      SUBROUTINE MPI_NULL_DELETE_FN(ierr)
      integer ierr
      ierr=-1000
      write(6,*)'Aufruf von MPI_NULL_DELETE_FN'
      write(6,*)'                   '
      write(6,*)'Programm-Abbruch   '
      if (ierr.eq.-1000) stop
      return
      end
      
      SUBROUTINE MPI_DUP_FN(ierr)
      integer ierr
      ierr=-1000
      write(6,*)'Aufruf von MPI_DUP_FN'
      write(6,*)'                   '
      write(6,*)'Programm-Abbruch   '
      if (ierr.eq.-1000) stop
      return
      end
