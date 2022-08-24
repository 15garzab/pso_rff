#!/usr/bin/bash
### all parameters for this forcefield ... input
pwd=$PWD
#echo "this is for debugging run_lammps.bash"
#echo $pwd

# target structure for temp run
targetdir=$1
rand_dir=$2
#take $@ all parameters array  bash array of all input pareamters to this file.
## $@ equilvalent to  argv[1:]

# module loads should be in batch submission script, right?
#module load intel/2018.2.199 intel-mpi/2018.2.199 mkl/2018.2.199
#module load lammps/16Mar18
# rand dir for temp run from python script
#mkdir $pwd/$rand_dir
#echo "this is the rand dir"
#echo $rand_dir
#echo "this is the current plus rand dir"
#echo "$pwd/$rand_dir"
if [ -d $pwd/$rand_dir ]; then
	echo "ok"
else
	echo "$pwd/$rand_dir"
fi
cd $pwd/$rand_dir
cp $pwd/$targetdir/in.cu .
cp $pwd/$targetdir/data.cu .
#cp $pwd/ffield.reax.temp .
# run the lammps file of interest with variable input
lmp_mpi < in.cu
#lmp_mpi < in.cu
# copy outputs to storage dir
if [ -e log.lammps ]; then	
	if [[ `grep -q "nan" log.lammps` || `grep -q "nan" forces.dat` ]]; then
		firstline=`head -1 log.lammps`
		echo "these ffield parameters yield NaN for the following training entry: "$targetdir
		echo $firstline
		echo $PWD
		energy="nan"
	elif grep -q "Energy initial" log.lammps; then	
		#
		lineno=`grep -n "Energy initial" log.lammps | cut -f1 -d:`
		((lineno=lineno+1))
		energy=`sed -n "${lineno}p" log.lammps | awk 'END {print $NF}'`
		# for debugging cut/sed/awk combo
		#echo $energy
		# only for debugging LAMMPS
		#cp log.lammps "$pwd/storage/ene-"$2
	else
		echo "log.lammps does not contain energy in file "$targetdir
		echo $PWD
	fi
else
	echo "log.lammps is missing"
	echo $PWD
fi

if [ -e forces.dat ]; then
	if grep -q "nan" forces.dat; then
		energy="nan"
		echo $PWD
		echo "Nan in forces.dat for target"$targetdir
	else
		#lineno = 'grep -n forces.dat'
		# only for debugging LAMMPS	
		cp forces.dat "$pwd/storage/forces-"$rand_dir
	fi
else
	echo "forces.dat is missing"
	cp log.lammps "$pwd/failed/log-"$rand_dir
	cp in.cu "$pwd/failed/"$2"-in.cu"
	echo $PWD
fi

cd $pwd

# delete temporary run dir
#rm -rf $pwd/$rand_dir

# output the name of the storage dir
echo $energy
