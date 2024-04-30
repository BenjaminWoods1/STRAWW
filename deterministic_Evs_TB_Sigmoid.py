"""
    Calculating the deterministic valley splitting (i.e. using the virtual crystal approximation)
    for the sp3d5s* tight binding model as we change lambda, n_Ge_well_max, and shear strain
"""

### Importing files and create file directory
if True:
    import sys
    import os
    import SiGe_Quantum_Well_sp3d5s2 as SQWTB
    import numpy as np
    import scipy.linalg as linalg
    import scipy.sparse as Spar
    import scipy.sparse.linalg as SparLinalg
    import matplotlib.pyplot as plt
    import parameters as par
    from time import time

    dir1 = "%s/deterministic_Evs_TB_Sigmoid_DATA" % (dir)
    dirS = "%s/deterministic_Evs_TB_Sigmoid_DATA/data" % (dir)
    dirF = "%s/deterministic_Evs_TB_Sigmoid_DATA/figs" % (dir)
    if not os.path.isdir(dir1):
        os.mkdir(dir1)
    if not os.path.isdir(dirS):
        os.mkdir(dirS)
    if not os.path.isdir(dirF):
        os.mkdir(dirF)

    def convert_mins_to_readable_time(minutes):
        if minutes < 2.:
            string = "%d minutes and %d seconds" % (minutes//1.,minutes % 1. * 60.)
        elif minutes < 60.:
            string = "%d minutes" % (minutes)
        else:
            string = "%d hours and %d minutes" % (minutes//60., minutes % 60.)
        return string

### Parameters
if True:
    x_Ge_substrate = 0.3 # Ge concentration in the substrate that defines the in-plane lattice constant
    system = SQWTB.SiGe_Quantum_Well_sp3d5s2(x_Ge_substrate)
    N_bar = 40
    N_well = 140
    N_interface = int(sys.argv[1]) # width of the interface
    xGe_bar = 1.*x_Ge_substrate
    Fz = float(sys.argv[2]) # electric field in (mV/nm)

### Plot the wave function envelope of the ground valley for the input parameters in the absence of
### shear strain and wiggles to makes sure the barrier and well regions are large enough
if str(sys.argv[3]) == 'wavefunction_plot':

    ### Generate the Ge profile
    interface_width = (0.543/4)*N_interface
    print("Interface width = %.2f nm" % (interface_width))
    Ge_arr, z_arr_cont, Ge_arr_cont = SQWTB.Sigmoid_barrier_WiggleWell_Ge_profile(N_bar,N_well,N_interface,10,xGe_bar,0.0,z0 = 0,PLOT = True)
    z_arr = SQWTB.z_arr_gen(system.a_sub,Ge_arr); z_arr = z_arr[:] - z_arr[N_bar+N_interface]

    ### Calculate the perpendicular lattice constant (along the growth direction) for pure Si layers
    a_perp_Si = SQWTB.calc_m_perp_lat_constant(system.a_sub,0.)

    ### Generate the Hamiltonian, and calculate the valley states and the valley splitting
    kx = 0; ky = 0; sigma = 1.; num = 4; epsilon_xy = 0.00
    Ham, z_arr = system.Ham_spinless_gen(kx,ky,Fz,Ge_arr,epsilon_xy)
    print("Ham shape", Ham.shape)
    eigs, U = system.solve_Ham(Ham,num,sigma,Return_vecs = True)
    E_VS = 1.e6*(eigs[1] - eigs[0])
    print("E_VS = %.5f micro eV" % (E_VS))

    ### Plot the envelope of the valley state
    envelope = system.project_state_onto_valley(U[:,0])
    N_orb = 10
    A = np.square(np.absolute(envelope))
    A_proj = np.zeros(int(A.size/N_orb))
    for i in range(N_orb):
        A_proj[:] += A[i::N_orb]
    z_cont = .1*z_arr_cont*(a_perp_Si/5.430)
    z_wavefunction = (np.arange(A_proj.size) - N_bar-N_interface)*(0.1*a_perp_Si/4)
    norm_factor = np.max(A_proj)/np.max(Ge_arr_cont)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1); ax1t = ax1.twinx()
    #ax.plot(z_cont,Ge_arr_cont*norm_factor*.8,c = 'r')
    ax1t.plot(z_cont,Ge_arr_cont,c = 'r')
    ax1.fill_between(z_wavefunction,0,A_proj,color = 'darkblue',alpha = 0.5)
    ax1.set_xlabel(r"$z$ (nm)")
    ax1.set_ylabel(r"$|\psi_{+z}|^{2}$",color = 'darkblue')
    ax1t.set_ylabel(r"$n_{Ge}$",color='r')
    ax1t.tick_params(axis='y', labelcolor='r')
    ax1.set_yticklabels([])
    ax1t.set_yticks([0,0.15,0.3])
    ax1.set_ylim(ymin = 0)
    ax1t.set_ylim(ymin = 0,ymax = 0.38)
    ax1.set_xlim(np.min(z_wavefunction),np.max(z_wavefunction))
    plt.subplots_adjust(left = 0.1, bottom=0.16, right=0.85, top=0.96)
    fig.savefig("%s/EnvelopeWavefunction_Nint=%d_Fz=%.2f.png" % (dirF,N_interface,Fz),dpi = 200)
    plt.show()

### Calculate the valley splitting as a function of Wiggle period and shear strain for fixed xGe_well_max
if str(sys.argv[3]) == 'calc_1':

    xGe_well_max = float(sys.argv[4])
    N_wiggles = np.linspace(3.5,23,101)
    epsilon_xy = np.linspace(0.,0.005,51)

    ### Loop through values of N_wiggles and epsilon_xy
    num = 2
    eig_arr = np.zeros((N_wiggles.size,epsilon_xy.size,num))
    Evs_arr = np.zeros((N_wiggles.size,epsilon_xy.size))
    ell_arr = np.zeros((N_wiggles.size))
    for i in range(N_wiggles.size):
        start_time = time()

        ### Create Ge concentration profile
        Ge_arr, z_arr_cont, Ge_arr_cont = SQWTB.Sigmoid_barrier_WiggleWell_Ge_profile(N_bar,N_well,N_interface,N_wiggles[i],xGe_bar,xGe_well_max,z0 = 0,PLOT = False)
        z_arr = SQWTB.z_arr_gen(system.a_sub,Ge_arr)

        ### Calculate the wiggle well period
        a_perp_Wiggles_ave = SQWTB.calc_m_perp_lat_constant(system.a_sub,0.5*xGe_well_max)
        ell_arr[i] = N_wiggles[i]*(a_perp_Wiggles_ave/4.)

        ### Loop through the shear strain values
        for j in range(epsilon_xy.size):
            #print(epsilon_xy.size - j)

            ### Construct the Hamiltonian and solve for the low-energy states
            kx = 0; ky = 0; sigma = 1.
            Ham, z_arr = system.Ham_spinless_gen(kx,ky,Fz,Ge_arr,epsilon_xy[j])
            if (i== 0) and (j == 0):
                print("Ham shape", Ham.shape)
            eig_arr[i,j,:] = system.solve_Ham(Ham,num,sigma,Return_vecs = False)
            Evs_arr[i,j] = 1.e6*(eig_arr[i,j,1] - eig_arr[i,j,0]) # Valley splitting

        ### save data
        if i > 5:
            np.save("%s/eigArr_Nint=%d_Fz=%.2f_xGeWellMax=%.4f.npy" % (dirS,N_interface,Fz,xGe_well_max),eig_arr)
            np.save("%s/EvsArr_Nint=%d_Fz=%.2f_xGeWellMax=%.4f.npy" % (dirS,N_interface,Fz,xGe_well_max),Evs_arr)
            np.save("%s/ellArr_Nint=%d_Fz=%.2f_xGeWellMax=%.4f.npy" % (dirS,N_interface,Fz,xGe_well_max),ell_arr)
            np.save("%s/NwigglesArr_Nint=%d_Fz=%.2f_xGeWellMax=%.4f.npy" % (dirS,N_interface,Fz,xGe_well_max),N_wiggles)
            np.save("%s/epxyArr_Nint=%d_Fz=%.2f_xGeWellMax=%.4f.npy" % (dirS,N_interface,Fz,xGe_well_max),epsilon_xy)

        ### Estimate time remaining
        end_time = time()
        Delta_t = end_time - start_time
        time_remaining = Delta_t * (N_wiggles.size - i -1) / 60.
        readable_time = convert_mins_to_readable_time(time_remaining)
        if i % 1 == 0:
            print ("Delta_t = %.1f seconds, %d values remain -> etr = %s" % (Delta_t,N_wiggles.size-i-1,readable_time))

### Calculate the valley splitting as a function of Wiggle period and xGe_well_max for fixed shear strain
if str(sys.argv[3]) == 'calc_2':

    epsilon_xy = float(sys.argv[4])
    N_wiggles = np.linspace(3.5,23,101)
    xGe_well_max = np.linspace(0.,0.1,51)
    #xGe_well_max = np.linspace(0.,0.1,5)

    ### Loop through values of N_wiggles and xGe_well_max
    num = 2
    eig_arr = np.zeros((N_wiggles.size,xGe_well_max.size,num))
    Evs_arr = np.zeros((N_wiggles.size,xGe_well_max.size))
    ell_arr = np.zeros((N_wiggles.size,xGe_well_max.size))
    for i in range(N_wiggles.size):
        start_time = time()

        ### Loop through the shear strain values
        for j in range(xGe_well_max.size):
            #print(xGe_well_max.size - j)

            ### Create Ge concentration profile
            Ge_arr, z_arr_cont, Ge_arr_cont = SQWTB.Sigmoid_barrier_WiggleWell_Ge_profile(N_bar,N_well,N_interface,N_wiggles[i],xGe_bar,xGe_well_max[j],z0 = 0,PLOT = False)
            z_arr = SQWTB.z_arr_gen(system.a_sub,Ge_arr)

            ### Calculate the wiggle well period
            a_perp_Wiggles_ave = SQWTB.calc_m_perp_lat_constant(system.a_sub,0.5*xGe_well_max[j])
            ell_arr[i,j] = N_wiggles[i]*(a_perp_Wiggles_ave/4.)

            ### Construct the Hamiltonian and solve for the low-energy states
            kx = 0; ky = 0; sigma = 1.
            Ham, z_arr = system.Ham_spinless_gen(kx,ky,Fz,Ge_arr,epsilon_xy)
            if (i== 0) and (j == 0):
                print("Ham shape", Ham.shape)
            eig_arr[i,j,:] = system.solve_Ham(Ham,num,sigma,Return_vecs = False)
            Evs_arr[i,j] = 1.e6*(eig_arr[i,j,1] - eig_arr[i,j,0]) # Valley splitting

        ### save data
        if i > 5:
            np.save("%s/eigArr_Nint=%d_Fz=%.2f_epxy=%.4f.npy" % (dirS,N_interface,Fz,epsilon_xy),eig_arr)
            np.save("%s/EvsArr_Nint=%d_Fz=%.2f_epxy=%.4f.npy" % (dirS,N_interface,Fz,epsilon_xy),Evs_arr)
            np.save("%s/ellArr_Nint=%d_Fz=%.2f_epxy=%.4f.npy" % (dirS,N_interface,Fz,epsilon_xy),ell_arr)
            np.save("%s/NwigglesArr_Nint=%d_Fz=%.2f_epxy=%.4f.npy" % (dirS,N_interface,Fz,epsilon_xy),N_wiggles)
            np.save("%s/xGeWellMaxArr_Nint=%d_Fz=%.2f_epxy=%.4f.npy" % (dirS,N_interface,Fz,epsilon_xy),xGe_well_max)

        ### Estimate time remaining
        end_time = time()
        Delta_t = end_time - start_time
        time_remaining = Delta_t * (N_wiggles.size - i -1) / 60.
        readable_time = convert_mins_to_readable_time(time_remaining)
        if i % 1 == 0:
            print ("Delta_t = %.1f seconds, %d values remain -> etr = %s" % (Delta_t,N_wiggles.size-i-1,readable_time))

### Calculate the valley splitting as a function of shear straind and xGe_well_max for fixed Wiggle Well period
if str(sys.argv[3]) == 'calc_3':

    N_wiggles = float(sys.argv[4])
    xGe_well_max = np.linspace(0.,0.1,51)
    epsilon_xy = np.linspace(0.,0.005,51)

    ### Loop through values of N_wiggles and xGe_well_max
    num = 2
    eig_arr = np.zeros((xGe_well_max.size,epsilon_xy.size,num))
    Evs_arr = np.zeros((xGe_well_max.size,epsilon_xy.size))
    ell_arr = np.zeros((xGe_well_max.size))
    for i in range(xGe_well_max.size):
        start_time = time()

        ### Create Ge concentration profile
        Ge_arr, z_arr_cont, Ge_arr_cont = SQWTB.Sigmoid_barrier_WiggleWell_Ge_profile(N_bar,N_well,N_interface,N_wiggles,xGe_bar,xGe_well_max[i],z0 = 0,PLOT = False)
        z_arr = SQWTB.z_arr_gen(system.a_sub,Ge_arr)

        ### Calculate the wiggle well period
        a_perp_Wiggles_ave = SQWTB.calc_m_perp_lat_constant(system.a_sub,0.5*xGe_well_max[i])
        ell_arr[i] = N_wiggles*(a_perp_Wiggles_ave/4.)

        ### Loop through the shear strain values
        for j in range(epsilon_xy.size):
            #print(epsilon_xy.size - j)

            ### Construct the Hamiltonian and solve for the low-energy states
            kx = 0; ky = 0; sigma = 1.
            Ham, z_arr = system.Ham_spinless_gen(kx,ky,Fz,Ge_arr,epsilon_xy[j])
            if (i== 0) and (j == 0):
                print("Ham shape", Ham.shape)
            eig_arr[i,j,:] = system.solve_Ham(Ham,num,sigma,Return_vecs = False)
            Evs_arr[i,j] = 1.e6*(eig_arr[i,j,1] - eig_arr[i,j,0]) # Valley splitting

        ### save data
        if i > 5:
            np.save("%s/eigArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles),eig_arr)
            np.save("%s/EvsArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles),Evs_arr)
            np.save("%s/ellArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles),ell_arr)
            np.save("%s/epxyArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles),epsilon_xy)
            np.save("%s/xGeWellMaxArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles),xGe_well_max)

        ### Estimate time remaining
        end_time = time()
        Delta_t = end_time - start_time
        time_remaining = Delta_t * (xGe_well_max.size - i -1) / 60.
        readable_time = convert_mins_to_readable_time(time_remaining)
        if i % 1 == 0:
            print ("Delta_t = %.1f seconds, %d values remain -> etr = %s" % (Delta_t,xGe_well_max.size-i-1,readable_time))





### Plot the valley splitting as a function of Wiggle period and shear strain for fixed xGe_well_max
if str(sys.argv[3]) == 'plot_1':

    xGe_well_max = float(sys.argv[4])

    ### Calculate the average lattice constant in the well area
    a_perp_Wiggles_ave = SQWTB.calc_m_perp_lat_constant(system.a_sub,0.5*xGe_well_max)

    ### Load data
    eig_arr = np.load("%s/eigArr_Nint=%d_Fz=%.2f_xGeWellMax=%.4f.npy" % (dirS,N_interface,Fz,xGe_well_max))
    Evs_arr = np.load("%s/EvsArr_Nint=%d_Fz=%.2f_xGeWellMax=%.4f.npy" % (dirS,N_interface,Fz,xGe_well_max))
    ell_arr = np.load("%s/ellArr_Nint=%d_Fz=%.2f_xGeWellMax=%.4f.npy" % (dirS,N_interface,Fz,xGe_well_max))
    N_wiggles = np.load("%s/NwigglesArr_Nint=%d_Fz=%.2f_xGeWellMax=%.4f.npy" % (dirS,N_interface,Fz,xGe_well_max))
    epsilon_xy = np.load("%s/epxyArr_Nint=%d_Fz=%.2f_xGeWellMax=%.4f.npy" % (dirS,N_interface,Fz,xGe_well_max))

    X = []; Y = []; Z = []
    for i in range(N_wiggles.size):
        for j in range(epsilon_xy.size):
            #X.append(ell_arr[i]/10.)
            X.append(N_wiggles[i]*(.1*a_perp_Wiggles_ave/4.))
            Y.append(epsilon_xy[j]*100)
            Z.append(Evs_arr[i,j])

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    TC1 = ax1.tricontourf(X,Y,Z,201,cmap = 'hot')
    #cbar = fig.colorbar(TC1, ax=ax1,ticks = ticks)
    cbar = fig.colorbar(TC1, ax=ax1)
    cbar.set_label('$E_{vs}$ ($\mu$eV)')
    ax1.set_xlabel(r"$\lambda$ (nm)")
    ax1.set_ylabel(r"$\epsilon_{xy}$ (%)")
    plt.subplots_adjust(left = 0.16, bottom=0.2, right=0.93, top=0.96)
    plt.show()

    plt.plot(epsilon_xy,Evs_arr[45,:],c = 'r')
    plt.show()

### Plot the valley splitting as a function of Wiggle period and xGe_well_max for fixed shear strain
if str(sys.argv[3]) == 'plot_2':

    epsilon_xy = float(sys.argv[4])

    ### Calculate the average lattice constant for pure Si
    a_perp_Si = SQWTB.calc_m_perp_lat_constant(system.a_sub,0.)

    ### Load data
    eig_arr = np.load("%s/eigArr_Nint=%d_Fz=%.2f_epxy=%.4f.npy" % (dirS,N_interface,Fz,epsilon_xy))
    Evs_arr = np.load("%s/EvsArr_Nint=%d_Fz=%.2f_epxy=%.4f.npy" % (dirS,N_interface,Fz,epsilon_xy))
    ell_arr = np.load("%s/ellArr_Nint=%d_Fz=%.2f_epxy=%.4f.npy" % (dirS,N_interface,Fz,epsilon_xy))
    N_wiggles = np.load("%s/NwigglesArr_Nint=%d_Fz=%.2f_epxy=%.4f.npy" % (dirS,N_interface,Fz,epsilon_xy))
    xGe_well_max = np.load("%s/xGeWellMaxArr_Nint=%d_Fz=%.2f_epxy=%.4f.npy" % (dirS,N_interface,Fz,epsilon_xy))

    X = []; Y = []; Z = []
    for i in range(N_wiggles.size):
        for j in range(xGe_well_max.size):
            #X.append(ell_arr[i]/10.)
            X.append(N_wiggles[i]*(.1*a_perp_Si/4.))
            Y.append(xGe_well_max[j])
            Z.append(Evs_arr[i,j])

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    TC1 = ax1.tricontourf(X,Y,Z,201,cmap = 'hot')
    #cbar = fig.colorbar(TC1, ax=ax1,ticks = ticks)
    cbar = fig.colorbar(TC1, ax=ax1)
    cbar.set_label('$E_{vs}$ ($\mu$eV)')
    ax1.set_xlabel(r"$\lambda$ (nm)")
    ax1.set_ylabel(r"$n_{Ge}$")
    plt.subplots_adjust(left = 0.16, bottom=0.2, right=0.9, top=0.96)
    plt.show()

### Plot the valley splitting as a function of shear straind and xGe_well_max for fixed Wiggle Well period
if str(sys.argv[3]) == 'plot_3':

    N_wiggles = float(sys.argv[4])

    ### Calculate the average lattice constant for pure Si
    a_perp_Si = SQWTB.calc_m_perp_lat_constant(system.a_sub,0.)
    ell = N_wiggles * (.1*a_perp_Si/4.); print("lambda ~ %.2f nm" % (ell))

    ### Load data
    eig_arr = np.load("%s/eigArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles))
    Evs_arr = np.load("%s/EvsArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles))
    ell_arr = np.load("%s/ellArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles))
    epsilon_xy = np.load("%s/epxyArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles))
    xGe_well_max = np.load("%s/xGeWellMaxArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles))

    X = []; Y = []; Z = []
    for i in range(epsilon_xy.size):
        for j in range(xGe_well_max.size):
            X.append(epsilon_xy[i]*1.e2)
            Y.append(xGe_well_max[j])
            Z.append(Evs_arr[j,i])

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    TC1 = ax1.tricontourf(X,Y,Z,201,cmap = 'hot')
    #cbar = fig.colorbar(TC1, ax=ax1,ticks = ticks)
    cbar = fig.colorbar(TC1, ax=ax1)
    cbar.set_label('$E_{vs}$ ($\mu$eV)')
    ax1.set_xlabel(r"$\epsilon_{xy}$ (%)")
    ax1.set_ylabel(r"$n_{Ge}$")
    plt.subplots_adjust(left = 0.16, bottom=0.2, right=0.9, top=0.96)
    plt.show()

### Plot the valley splitting for all three situations considered above
if str(sys.argv[3]) == 'plot_Evs_allThree':

    epsilon_xy = float(sys.argv[4])
    xGe_well_max = float(sys.argv[5])
    N_wiggles = float(sys.argv[6])
    ### Calculate the average lattice constant for pure Si
    a_perp_Si = SQWTB.calc_m_perp_lat_constant(system.a_sub,0.)
    ell_approx = N_wiggles*.1*a_perp_Si/4.

    ### bounds for plots
    epsilon_xy_max = 0.0042
    ell_min = 0.58; ell_max = 2.85
    ell_min = 0.85; ell_max = 2.65
    xGe_max = 0.085/2
    Evs_max = 249.99

    width = 3.847; height = .8 * width
    plt.rcParams["figure.figsize"] = (width,height)
    fig = plt.figure()
    ys = .12
    x3o = 0.14; y3o = .11
    dx3 = .65; dy3 = .23
    y2o = y3o + ys + dy3
    y1o = y2o + .4*ys + dy3
    x4o = x3o + dx3 + .03
    dx4 = .02; dy4 = y1o + dy3 - y3o
    ax1 = fig.add_axes([x3o,y1o,dx3,dy3])
    ax2 = fig.add_axes([x3o,y2o,dx3,dy3])
    ax3 = fig.add_axes([x3o,y3o,dx3,dy3])
    ax4 = fig.add_axes([x4o,y3o,dx4,dy4])

    ### 1st plot (fixed epsilon_xy)
    eig_arr = np.load("%s/eigArr_Nint=%d_Fz=%.2f_epxy=%.4f.npy" % (dirS,N_interface,Fz,epsilon_xy))
    Evs_arr = np.load("%s/EvsArr_Nint=%d_Fz=%.2f_epxy=%.4f.npy" % (dirS,N_interface,Fz,epsilon_xy))
    ell_arr = np.load("%s/ellArr_Nint=%d_Fz=%.2f_epxy=%.4f.npy" % (dirS,N_interface,Fz,epsilon_xy))
    N_wiggles_arr = np.load("%s/NwigglesArr_Nint=%d_Fz=%.2f_epxy=%.4f.npy" % (dirS,N_interface,Fz,epsilon_xy))
    xGe_well_max_arr = np.load("%s/xGeWellMaxArr_Nint=%d_Fz=%.2f_epxy=%.4f.npy" % (dirS,N_interface,Fz,epsilon_xy))

    X = []; Y = []; Z = []
    for i in range(N_wiggles_arr.size):
        for j in range(xGe_well_max_arr.size):
            X.append(ell_arr[i,j]/10.)
            #X.append(N_wiggles_arr[i]*(.1*a_perp_Si/4.))
            Y.append(xGe_well_max_arr[j]*1.e2/2.)
            Z.append(min(Evs_arr[i,j],Evs_max))

    ax1.set_facecolor('k')
    #levels = np.linspace(0.,500,251)
    levels = np.linspace(0.,250,251)
    TC1 = ax1.tricontourf(X,Y,Z,200,cmap = 'tab20c',levels = levels)
    #cbar = fig.colorbar(TC1, cax=ax4,ticks = [0,100,200,300,400,500])
    cbar = fig.colorbar(TC1, cax=ax4,ticks = [0,50,100,150,200,250])
    cbar.ax.set_yticklabels(['0', '50', '100', '150', '200', r'$\geq 250$'])
    cbar.set_label('$E_{vs}$ ($\mu$eV)',labelpad = -4)
    #ax1.set_xlabel(r"$\lambda$ (nm)",labelpad = -1)
    ax1.set_ylabel(r"$\bar{n}_{Ge}$ (%)",labelpad = 14)
    ax1.set_xlim(xmin = ell_min,xmax = ell_max)
    ax1.set_ylim(ymin = 0,ymax = xGe_max*1.e2)
    ax1.set_xticks([1,1.5,2,2.5])
    ax1.set_yticks([0,2,4])
    ax1.set_xticklabels([])


    ### 2nd plot (fixed xGe_well_max)
    eig_arr = np.load("%s/eigArr_Nint=%d_Fz=%.2f_xGeWellMax=%.4f.npy" % (dirS,N_interface,Fz,xGe_well_max))
    Evs_arr = np.load("%s/EvsArr_Nint=%d_Fz=%.2f_xGeWellMax=%.4f.npy" % (dirS,N_interface,Fz,xGe_well_max))
    ell_arr = np.load("%s/ellArr_Nint=%d_Fz=%.2f_xGeWellMax=%.4f.npy" % (dirS,N_interface,Fz,xGe_well_max))
    N_wiggles_arr = np.load("%s/NwigglesArr_Nint=%d_Fz=%.2f_xGeWellMax=%.4f.npy" % (dirS,N_interface,Fz,xGe_well_max))
    epsilon_xy_arr = np.load("%s/epxyArr_Nint=%d_Fz=%.2f_xGeWellMax=%.4f.npy" % (dirS,N_interface,Fz,xGe_well_max))

    X = []; Y = []; Z = []
    for i in range(N_wiggles_arr.size):
        for j in range(epsilon_xy_arr.size):
            X.append(ell_arr[i]/10.)
            Y.append(epsilon_xy_arr[j]*100)
            Z.append(min(Evs_arr[i,j],Evs_max))

    TC2 = ax2.tricontourf(X,Y,Z,200,cmap = 'tab20c',levels = levels)
    ax2.set_xlabel(r"$\lambda$ (nm)",labelpad = -1)
    ax2.set_ylabel(r"$\epsilon_{xy}$ (%)")
    ax2.set_xlim(xmin = ell_min,xmax = ell_max)
    ax2.set_ylim(ymin = 0, ymax = epsilon_xy_max*1.e2)
    ax2.set_yticks([0,0.2,0.4])


    ### 3rd plot (fixed N_Ge)
    eig_arr = np.load("%s/eigArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles))
    Evs_arr = np.load("%s/EvsArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles))
    ell_arr = np.load("%s/ellArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles))
    epsilon_xy_arr = np.load("%s/epxyArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles))
    xGe_well_max_arr = np.load("%s/xGeWellMaxArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles))

    X = []; Y = []; Z = []
    for i in range(epsilon_xy_arr.size):
        for j in range(xGe_well_max_arr.size):
            X.append(epsilon_xy_arr[i]*1.e2)
            Y.append(xGe_well_max_arr[j]*1.e2/2.)
            Z.append(min(Evs_arr[j,i],Evs_max))

    TC3 = ax3.tricontourf(X,Y,Z,200,cmap = 'tab20c',levels = levels)
    ax3.set_xlabel(r"$\epsilon_{xy}$ (%)",labelpad = -1)
    ax3.set_ylabel(r"$\bar{n}_{Ge}$ (%)",labelpad = 14)
    ax3.set_xlim(xmin = 0, xmax = epsilon_xy_max*1.e2)
    ax3.set_ylim(ymin = 0,ymax = xGe_max*1.e2)
    ax3.set_yticks([0,2,4])

    ax1.scatter([ell_approx],[xGe_well_max*1.e2/2.],c = 'k',marker = '*',s = 50)
    ax2.scatter([ell_approx],[epsilon_xy*1.e2],c = 'k',marker = '*',s = 50)
    ax3.scatter([epsilon_xy*1.e2],[xGe_well_max*1.e2/2.],c = 'k',marker = '*',s = 50)

    ax1.text(.04, .92, "(a)", ha="left", va="top", transform=ax1.transAxes,fontsize = 10.5,weight = 'bold', c= 'k')
    ax1.text(.05, .08, r"$\epsilon_{xy} = %.2f$%% " % (epsilon_xy*1.e2), ha="left", va="bottom", transform=ax1.transAxes,fontsize = 10., c= 'k')
    ax2.text(.04, .92, "(b)", ha="left", va="top", transform=ax2.transAxes,fontsize = 10.5,weight = 'bold', c= 'k')
    ax2.text(.05, .08, r"$\bar{n}_{Ge} = %.1f$%%" % (xGe_well_max*1.e2/2), ha="left", va="bottom", transform=ax2.transAxes,fontsize = 10., c= 'k')
    ax3.text(.04, .92, "(c)", ha="left", va="top", transform=ax3.transAxes,fontsize = 10.5,weight = 'bold', c= 'k')
    ax3.text(.05, .08, r"$\lambda = %.2f$ nm" % (ell_approx), ha="left", va="bottom", transform=ax3.transAxes,fontsize = 10., c= 'k')

    fig.savefig("%s/EvsMaps_Nint=%d_Fz=%.2f_epxy=%.4f_xGeWellMax=%.4f_Nwiggles=%.2f.png" % (dirF,N_interface,Fz,epsilon_xy,xGe_well_max,N_wiggles),dpi = 300)
    plt.show()

### Plot the valley splitting for all three situations considered above
if str(sys.argv[3]) == 'plot_Evs_allThree_NatureVersion':

    epsilon_xy = float(sys.argv[4])
    xGe_well_max = float(sys.argv[5])
    N_wiggles = float(sys.argv[6])
    ### Calculate the average lattice constant for pure Si
    a_perp_Si = 5.430
    ell_approx = N_wiggles*.1*a_perp_Si/4.

    ### bounds for plots
    epsilon_xy_max = 0.0042
    ell_min = 0.58; ell_max = 2.85
    ell_min = 0.85; ell_max = 2.65
    xGe_max = 0.085/2
    Evs_max = 500.

    width = 3.847; height = .8 * width
    plt.rcParams["figure.figsize"] = (width,height)
    fig = plt.figure()
    ys = .1
    x3o = 0.14; y3o = .05
    dx3 = .65; dy3 = .23
    y2o = y3o + ys + dy3
    y1o = y2o + ys + dy3
    x4o = x3o + dx3 + .03
    dx4 = .02; dy4 = y1o + dy3 - y3o
    ax1 = fig.add_axes([x3o,y1o,dx3,dy3])
    ax2 = fig.add_axes([x3o,y2o,dx3,dy3])
    ax3 = fig.add_axes([x3o,y3o,dx3,dy3])
    ax4 = fig.add_axes([x4o,y3o,dx4,dy4])

    ### 1st plot (fixed epsilon_xy)
    eig_arr = np.load("%s/eigArr_Nint=%d_Fz=%.2f_epxy=%.4f.npy" % (dirS,N_interface,Fz,epsilon_xy))
    Evs_arr = np.load("%s/EvsArr_Nint=%d_Fz=%.2f_epxy=%.4f.npy" % (dirS,N_interface,Fz,epsilon_xy))
    ell_arr = np.load("%s/ellArr_Nint=%d_Fz=%.2f_epxy=%.4f.npy" % (dirS,N_interface,Fz,epsilon_xy))
    N_wiggles_arr = np.load("%s/NwigglesArr_Nint=%d_Fz=%.2f_epxy=%.4f.npy" % (dirS,N_interface,Fz,epsilon_xy))
    xGe_well_max_arr = np.load("%s/xGeWellMaxArr_Nint=%d_Fz=%.2f_epxy=%.4f.npy" % (dirS,N_interface,Fz,epsilon_xy))

    X = []; Y = []; Z = []
    for i in range(N_wiggles_arr.size):
        for j in range(xGe_well_max_arr.size):
            X.append(ell_arr[i,j]/10.)
            #X.append(N_wiggles_arr[i]*(.1*a_perp_Si/4.))
            Y.append(xGe_well_max_arr[j]*1.e2/2.)
            Z.append(min(Evs_arr[i,j],Evs_max*0.999))

    ax1.set_facecolor('k')
    #levels = np.linspace(0.,500,251)
    levels = np.linspace(0.,Evs_max,601)
    TC1 = ax1.tricontourf(X,Y,Z,601,cmap = 'tab20c',levels = levels)
    #ax1.set_xlabel(r"$\lambda$ (nm)",labelpad = -1)
    ax1.set_ylabel(r"$\bar{n}_{Ge}$ (%)",labelpad = 14)
    ax1.set_xlim(xmin = ell_min,xmax = ell_max)
    ax1.set_ylim(ymin = 0,ymax = xGe_max*1.e2)
    ax1.set_xticks([1,1.5,2,2.5])
    ax1.set_yticks([0,2,4])
    ax1.set_xticklabels([])

    #cbar = fig.colorbar(TC1, cax=ax4,ticks = [0,100,200,300,400,500])
    #cbar = fig.colorbar(TC1, cax=ax4,ticks = [0,50,100,150,200,250])
    #cbar.ax.set_yticklabels(['0', '50', '100', '150', '200', r'$\geq 250$'])
    cbar = fig.colorbar(TC1, cax=ax4,ticks = [0,int(Evs_max/5),int(2*Evs_max/5),int(3*Evs_max/5),int(4*Evs_max/5),int(Evs_max)])
    cbar.ax.set_yticklabels(['0', '%d' % (int(1*Evs_max/5)), '%d' % (int(2*Evs_max/5)), '%d' % (int(3*Evs_max/5)), '%d' % (int(4*Evs_max/5)), r'$\geq %d$' % (int(5*Evs_max/5))])
    cbar.ax.set_yticklabels(['','','','','',''])
    cbar.set_label('$E_{vs}$ ($\mu$eV)',labelpad = -4)
    cbar.set_label('')


    ### 2nd plot (fixed xGe_well_max)
    eig_arr = np.load("%s/eigArr_Nint=%d_Fz=%.2f_xGeWellMax=%.4f.npy" % (dirS,N_interface,Fz,xGe_well_max))
    Evs_arr = np.load("%s/EvsArr_Nint=%d_Fz=%.2f_xGeWellMax=%.4f.npy" % (dirS,N_interface,Fz,xGe_well_max))
    ell_arr = np.load("%s/ellArr_Nint=%d_Fz=%.2f_xGeWellMax=%.4f.npy" % (dirS,N_interface,Fz,xGe_well_max))
    N_wiggles_arr = np.load("%s/NwigglesArr_Nint=%d_Fz=%.2f_xGeWellMax=%.4f.npy" % (dirS,N_interface,Fz,xGe_well_max))
    epsilon_xy_arr = np.load("%s/epxyArr_Nint=%d_Fz=%.2f_xGeWellMax=%.4f.npy" % (dirS,N_interface,Fz,xGe_well_max))

    X = []; Y = []; Z = []
    for i in range(N_wiggles_arr.size):
        for j in range(epsilon_xy_arr.size):
            X.append(ell_arr[i]/10.)
            Y.append(epsilon_xy_arr[j]*100)
            Z.append(min(Evs_arr[i,j],Evs_max*0.999))

    TC2 = ax2.tricontourf(X,Y,Z,601,cmap = 'tab20c',levels = levels)
    ax2.set_xlabel(r"$\lambda$ (nm)",labelpad = -1)
    ax2.set_ylabel(r"$\epsilon_{xy}$ (%)")
    ax2.set_xlim(xmin = ell_min,xmax = ell_max)
    ax2.set_ylim(ymin = 0, ymax = epsilon_xy_max*1.e2)
    ax2.set_yticks([0,0.2,0.4])


    ### 3rd plot (fixed N_Ge)
    eig_arr = np.load("%s/eigArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles))
    Evs_arr = np.load("%s/EvsArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles))
    ell_arr = np.load("%s/ellArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles))
    epsilon_xy_arr = np.load("%s/epxyArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles))
    xGe_well_max_arr = np.load("%s/xGeWellMaxArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles))

    X = []; Y = []; Z = []
    for i in range(epsilon_xy_arr.size):
        for j in range(xGe_well_max_arr.size):
            X.append(epsilon_xy_arr[i]*1.e2)
            Y.append(xGe_well_max_arr[j]*1.e2/2.)
            Z.append(min(Evs_arr[j,i],Evs_max*0.999))

    TC3 = ax3.tricontourf(X,Y,Z,601,cmap = 'tab20c',levels = levels)
    ax3.set_xlabel(r"$\epsilon_{xy}$ (%)",labelpad = -1)
    ax3.set_ylabel(r"$\bar{n}_{Ge}$ (%)",labelpad = 14)
    ax3.set_xlim(xmin = 0, xmax = epsilon_xy_max*1.e2)
    ax3.set_ylim(ymin = 0,ymax = xGe_max*1.e2)
    ax3.set_yticks([0,2,4])

    ax1.scatter([ell_approx],[xGe_well_max*1.e2/2.],c = 'k',marker = '*',s = 50)
    ax2.scatter([ell_approx],[epsilon_xy*1.e2],c = 'k',marker = '*',s = 50)
    ax3.scatter([epsilon_xy*1.e2],[xGe_well_max*1.e2/2.],c = 'k',marker = '*',s = 50)

    #ax1.text(.04, .92, "(a)", ha="left", va="top", transform=ax1.transAxes,fontsize = 10.5,weight = 'bold', c= 'k')
    #ax1.text(.05, .08, r"$\epsilon_{xy} = %.2f$%% " % (epsilon_xy*1.e2), ha="left", va="bottom", transform=ax1.transAxes,fontsize = 10., c= 'k')
    #ax2.text(.04, .92, "(b)", ha="left", va="top", transform=ax2.transAxes,fontsize = 10.5,weight = 'bold', c= 'k')
    #ax2.text(.05, .08, r"$\bar{n}_{Ge} = %.1f$%%" % (xGe_well_max*1.e2/2), ha="left", va="bottom", transform=ax2.transAxes,fontsize = 10., c= 'k')
    #ax3.text(.04, .92, "(c)", ha="left", va="top", transform=ax3.transAxes,fontsize = 10.5,weight = 'bold', c= 'k')
    #ax3.text(.05, .08, r"$\lambda = %.2f$ nm" % (ell_approx), ha="left", va="bottom", transform=ax3.transAxes,fontsize = 10., c= 'k')

    ax1.set_ylabel('')
    ax2.set_ylabel('')
    ax3.set_ylabel('')
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    ax3.set_xlabel('')
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    ax1.set_yticklabels([])
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])


    fig.savefig("%s/EvsMaps_NatureVersion_Nint=%d_Fz=%.2f_epxy=%.4f_xGeWellMax=%.4f_Nwiggles=%.2f.png" % (dirF,N_interface,Fz,epsilon_xy,xGe_well_max,N_wiggles),dpi = 300)
    plt.show()

### Plot line cuts of the valley splitting for panel a of the nature version above
if str(sys.argv[3]) == 'plot_Evs_linecuts_NatureVersion':

    epsilon_xy = float(sys.argv[4])
    xGe_well_max = float(sys.argv[5])
    N_wiggles = float(sys.argv[6])

    ### Calculate the average lattice constant for pure Si
    a_perp_Si = 5.430
    ell_approx = N_wiggles*.1*a_perp_Si/4.

    width = 3.847; height = .3 * width
    plt.rcParams["figure.figsize"] = (width,height)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ### Load data
    eig_arr = np.load("%s/eigArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles))
    Evs_arr = np.load("%s/EvsArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles))
    ell_arr = np.load("%s/ellArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles))
    epsilon_xy_arr = np.load("%s/epxyArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles))
    xGe_well_max_arr = np.load("%s/xGeWellMaxArr_Nint=%d_Fz=%.2f_Nwiggles=%.2f.npy" % (dirS,N_interface,Fz,N_wiggles))

    ### Find the index that most closely matches the input shear strain
    idx = np.argmin(np.absolute(epsilon_xy_arr - epsilon_xy))
    print("epsilon_xy_arr[%d] = %.5f, target = %.5f" % (idx,epsilon_xy_arr[idx],epsilon_xy))

    ### line cuts with constant shear strain
    X = []; Y = []; Y2 = []
    for j in range(xGe_well_max_arr.size):
        X.append(xGe_well_max_arr[j]*1.e2/2.)
        Y.append(Evs_arr[j,0])
        Y2.append(Evs_arr[j,idx])
    X = np.array(X); Y = np.array(Y)

    ax1.plot(X,Y,c = 'r',lw = 1.5,ls = 'dashed')
    ax1.plot(X,Y2,c = 'r',lw = 1.5)

    ax1.set_xticks([0,2,4])
    ax1.set_yticks([0,100,200,300])
    ax1.set_ylim(-10,310)

    ### Find the index that most closely matches the input xGe_well_max
    idx2 = np.argmin(np.absolute(xGe_well_max_arr - xGe_well_max))
    print("xGe_well_max_arr[%d] = %.5f, target = %.5f" % (idx2,xGe_well_max_arr[idx2],xGe_well_max))
    print(xGe_well_max_arr[0])

    ### line cuts with constant xGe_well_max
    X = []; Y = []; Y2 = []
    for i in range(epsilon_xy_arr.size):
        X.append(epsilon_xy_arr[i]*1.e2)
        Y.append(Evs_arr[0,i])
        Y2.append(Evs_arr[idx2,i])
    X = np.array(X); Y = np.array(Y)

    ax2.plot(X,Y,c = 'b',lw = 1.5,ls = 'dashed')
    ax2.plot(X,Y2,c = 'b',lw = 1.5)
    ax2.set_xticks([0,0.2,0.4])
    ax2.set_yticks([0,100,200,300])
    ax2.set_ylim(-10,310)

    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax1.set_yticklabels([])
    ax2.set_yticklabels([])

    plt.subplots_adjust(left = 0.01, bottom=0.05, right=0.99, top=0.96,wspace = 0.15)
    fig.savefig("%s/EvsLineCuts_NatureVersion_Nint=%d_Fz=%.2f_epxy=%.4f_xGeWellMax=%.4f_Nwiggles=%.2f.png" % (dirF,N_interface,Fz,epsilon_xy,xGe_well_max,N_wiggles),dpi = 300)
    plt.show()
