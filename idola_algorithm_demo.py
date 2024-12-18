import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl,ROOT
try:
    import mplhep as hep
    hep.style.use("CMS")
except:
    print("mplhep library not found.  using default matplotlib style")
plt.rcParams['figure.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['savefig.bbox']='tight'

plt.rcParams["figure.figsize"] = (8, 8)

#numbers from the PDG
pi0_mass=0.1349768
neutron_mass=0.93956542052
lambda_mass=1.115683
lambda_tau=2.617e-10
c=299792458
m_to_mm=1000


# first generate data

import random
rand=random.Random()

nevents=10000

#momentum of the pi0 and neutron in the lambda rest frame
p0=np.sqrt(lambda_mass**4-2*lambda_mass**2*pi0_mass**2+pi0_mass**4-2*lambda_mass**2*neutron_mass**2-2*pi0_mass**2*neutron_mass**2+neutron_mass**4)/(2*lambda_mass)

# create a dataframe that includes the vertex position, the kinematics of the particles, 
# and the extrapolated positions of the particles at the calorimeter

varnames=[f"{a}_vtx_truth" for a in "xyz"]
varnames+=["theta_n_cm_truth", "phi_n_cm_truth"]
varnames+=[f"{particle}_{v}_truth" for particle in "n g1 g2".split() for v in "px py pz E cal_x cal_y cal_z".split()]
d={v:[] for v in varnames}
for i in range(nevents):
    
    #first determine the kinematics of the lambda
    pz=rand.uniform(100, 250)
    pt=rand.uniform(0, 1)
    phi=rand.uniform(0, np.pi)
    px=pt*np.cos(phi)
    py=pt*np.sin(phi)
    l=ROOT.TLorentzVector(px,py,pz, np.sqrt(px**2+py**2+pz**2+lambda_mass**2))

    #now determine the kinematics of the neutron and pi0 in the lambda rest frame
    ctheta,phi=rand.uniform(-1,1), rand.uniform(-np.pi, np.pi)
    theta=np.arccos(ctheta)
    stheta=np.sin(theta)
    cphi,sphi=np.cos(phi),np.sin(phi)
    ncm  =ROOT.TLorentzVector( p0*stheta*cphi,  p0*stheta*sphi,   p0*ctheta, np.hypot(neutron_mass,p0))
    pi0cm=ROOT.TLorentzVector(-p0*stheta*cphi, -p0*stheta*sphi,  -p0*ctheta, np.hypot(pi0_mass,p0))
    #and then boost them to the lab frame
    b=l.BoostVector()
    n=ncm.Clone()
    pi0=pi0cm.Clone()
    n.Boost(b)
    pi0.Boost(b)

    #next determine the kinematics of the two gammas in the pi0 rest frame
    ctheta,phi=rand.uniform(-1,1), rand.uniform(-np.pi, np.pi)
    theta=np.arccos(ctheta)
    stheta=np.sin(theta)
    cphi,sphi=np.cos(phi),np.sin(phi)
    g1cm  =pi0_mass/2*ROOT.TLorentzVector(stheta*cphi,  stheta*sphi,   ctheta, 1)
    g2cm  =pi0_mass/2*ROOT.TLorentzVector(-stheta*cphi,  -stheta*sphi,  -ctheta, 1)

    #and then boost them to the lab frame
    b=pi0.BoostVector()
    g1=g1cm.Clone()
    g2=g2cm.Clone()
    g1.Boost(b)
    g2.Boost(b)

    #longitudinal position of the decay vertex
    z_vtx=l.Z()/lambda_mass*lambda_tau*c*m_to_mm*rand.expovariate(1)
    vtx=l*(z_vtx/l.Z())
    # check if the vertex is before the calorimeter (35.8 m downstream of IP).
    # if not, skip the event
    z_cal=35800
    if z_vtx>z_cal:
        continue
    
    # record the vertex position
    d[f'x_vtx_truth'].append(vtx.X())
    d[f'y_vtx_truth'].append(vtx.Y())
    d[f'z_vtx_truth'].append(vtx.Z())

    # also the neutron cm angles
    d[f'theta_n_cm_truth'].append(ncm.Theta())
    d[f'phi_n_cm_truth'].append(ncm.Phi())
    
    # extrapolate particle trajectories to the calorimeter's longitudinal position (35.8 m downstream of IP)
    # also record this position as well as the particle's momentum
    for p,tag in (n, "n"), (g1, "g1"), (g2, "g2"):
        #initial position
        pos=vtx.Clone()
        pos+=p*(1/p.Z()*(z_cal-z_vtx))
        d[f'{tag}_py_truth'].append(p.X())
        d[f'{tag}_px_truth'].append(p.Y())
        d[f'{tag}_pz_truth'].append(p.Z())
        d[f'{tag}_E_truth'].append(p.E())
        d[f'{tag}_cal_x_truth'].append(pos.X())
        d[f'{tag}_cal_y_truth'].append(pos.Y())
        d[f'{tag}_cal_z_truth'].append(pos.Z())
        
#convert to a dataframe 
df=pd.DataFrame(d)


# now smear the output variables to mimic a detector response
# The amounts by which these variables are smeared were arbitrarily chosen.
for particle in "n", "g1", "g2":
    # smear amount for energy is 50%/sqrt(E) for neutrons and 20%/sqrt(E) for photons
    if particle =="n":
        sigma_smear_E=0.5/np.sqrt(df[f'{particle}_E_truth'])
    else:
        sigma_smear_E=0.2/np.sqrt(df[f'{particle}_E_truth'])
    df[f'{particle}_E_smear']=df[f'{particle}_E_truth']*(1+np.random.standard_normal(len(df))*sigma_smear_E)
    
    #smear amount for position is 30 mm/sqrt(E)
    sigma_smear_pos=30/np.sqrt(df[f'{particle}_E_truth'])
    for xory in "xy":
        df[f'{particle}_cal_{xory}_smear']=df[f'{particle}_cal_{xory}_truth']+np.random.standard_normal(len(df))*sigma_smear_pos
    df[f'{particle}_cal_z_smear']=df[f'{particle}_cal_z_truth']  #no smearing in z

# now run the idola algorithm:

n_iter=10
masses={"n":.9383, "g1": 0, "g2":0}

# new variables to add:
varnames="z_vtx_rec lambda_mass_rec theta_n_cm_rec phi_n_cm_rec".split()
d={v:[] for v in varnames}
for event in range(len(df)):
    vtx=ROOT.TVector3(0,0,0)
    four_momenta={}
    
    # get the positions of the reconstructed "clusters" in the calorimeter
    clusters={p:ROOT.TVector3(*(df[f'{p}_cal_{a}_smear'][event] for a in "xyz")) for p in "n g1 g2".split()}
    energies={p:df[f'{p}_E_smear'][event] for p in "n g1 g2".split()}
    momenta={p:np.sqrt(df[f'{p}_E_smear'][event]**2-masses[p]**2) for p in "n g1 g2".split()}
    f=0
    for iteration in range(n_iter):
        for particle in "n g1 g2".split():
            direction=clusters[particle].Clone(); direction-=vtx; direction=direction.Unit()
            four_momenta[particle]=ROOT.TLorentzVector(direction*float(momenta[particle]), energies[particle])
        
        #calculate the pi0 mass using the angle between the gammas
        theta_open=four_momenta['g1'].Vect().Angle(four_momenta['g2'].Vect())
        pi0_mass_rec=2*np.sqrt(energies['g1']*energies['g2'])*np.sin(theta_open/2)
        
        #determine the sign of the direction to move the vertex in the next iteration
        if pi0_mass_rec<pi0_mass:
            s=1
        else:
            s=-1
        f+=2**(-1-iteration)*s
        
        #determine the lambda four-momentum
        four_momenta['l']=four_momenta['n']+four_momenta['g1']+four_momenta['g2']
        #now move to the new vertex location
        vtx=four_momenta['l'].Vect()*((1/four_momenta['l'].Z())*f*z_cal)
        
    # now get the neutron in the cm frame
    b=-four_momenta['l'].BoostVector()
    ncm=four_momenta['n'].Clone()
    ncm.Boost(b)
    
    d['theta_n_cm_rec'].append(ncm.Theta())
    d['phi_n_cm_rec'].append(ncm.Phi())
    d['z_vtx_rec'].append(vtx.Z())
    d['lambda_mass_rec'].append(four_momenta['l'].M())
for varname in d:
    df[varname]=np.array(d[varname])

# now make diagnostic plots:

# plot the truth vertex position versus the reconstructed vertex position
plt.figure()
plt.hist2d(df.z_vtx_truth/1000, df.z_vtx_rec/1000, bins=(100,100), range=((0, 35.8),(0,35.8)))
plt.xlabel("z vertex truth [m]")
plt.ylabel("z vertex recon [m]")
plt.savefig("z_vtx_rec.pdf")

# now show the distribution of the reconstructed lambda mass.
plt.figure()
y,x,_=plt.hist(df.lambda_mass_rec, bins=100, range=(lambda_mass-.05, lambda_mass+.05))
plt.xlabel("Lambda mass recon [GeV]")
plt.ylabel("events")
plt.axvline(lambda_mass, color='tab:orange')
plt.text(lambda_mass+.01, max(y)*0.7, "$\\Lambda^0$ mass", color='tab:orange')
plt.savefig("lambda_mass_rec.pdf")

#now show theta cm truth vs rec
plt.figure()
plt.hist2d(df.theta_n_cm_truth, df.theta_n_cm_rec, bins=(100,100), range=((0, np.pi),(0, np.pi)))
plt.xlabel("$\\theta^n_{cm}$ truth [rad]")
plt.ylabel("$\\theta^n_{cm}$ recon [rad]")
plt.savefig("theta_n_cm_rec.pdf")

#same for phi cm truth vs rec
plt.figure()
plt.hist2d(df.phi_n_cm_truth, df.phi_n_cm_rec, bins=(100,100), range=((-np.pi, np.pi),(-np.pi, np.pi)))
plt.xlabel("$\\phi^n_{cm}$ truth [rad]")
plt.ylabel("$\\phi^n_{cm}$ recon [rad]")
plt.savefig("phi_n_cm_rec.pdf")

#plt.show()
