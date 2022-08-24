# minimizing input file for REAX optimization
variable        dt equal 1.0

units           real
boundary        p p p
# needed for triclinic systems w large skew
box		tilt large

atom_style      charge
read_data       data.cu

mass		1 63.5463
#mass        2 58.6934
# mass		3 15.9999

pair_style	reax/c NULL safezone 2.4 mincap 200
pair_coeff	* * ffield.reax.temp Cu O

neighbor	2 bin
neigh_modify	every 10 delay 0 check no

fix             reax all qeq/reax 1 0.0 10.0 1e-6 reax/c

timestep        ${dt}

thermo          1
thermo_style    custom step etotal pe atoms
thermo_modify   lost ignore

dump            3 all custom 1 forces.dat id x y z fx fy fz
dump_modify	3 sort id

# configuration output section
min_style       quickmin
minimize        1.0e-8 1.0e-6 1000 1000
