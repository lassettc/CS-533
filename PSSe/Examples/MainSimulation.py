# File:"C:\Program Files (x86)\PTI\PSSEXplore33\EXAMPLE\My_py.py", generated on TUE, AUG 11 2015   8:29, release 33.05.02
from __future__ import division
from collections import defaultdict
import os,sys

#Change to your PSS/e Location
sys.path.append(r"C:\Program Files (x86)\PTI\PSSEXplore34\PSSPY27") #Give the path to PSSBIN to imoport psspy
sys.path.append(r"C:\Program Files (x86)\PTI\PSSEXplore34\PSSBIN")
os.environ['PATH'] = (r"C:\Program Files (x86)\PTI\PSSEXplore34\PSSPY27;" + r"C:\Program Files (x86)\PTI\PSSEXplore34\PSSBIN;" + os.environ['PATH'])
#import pssarrays
import psspy
import pssarrays
import redirect
import dyntools
import pssplot
import random
import math
import multiprocessing
import time
#import matplotlib
_i=psspy.getdefaultint()
_f=psspy.getdefaultreal()
_s=psspy.getdefaultchar()
redirect.psse2py()

in_file = 'savnw.sav'

#Initializes our case with name defined by in_file#
def Initialize_Case():
	psspy.psseinit(150000) #Initialize size of case, just choose large number
	psspy.case(in_file) #Load example case savnw.sav


#Change initial conditions by scaling loads in case#
import copy
def Change_Init_Conds(Load_Numbs, Load_IDs, Complex_Power, Complex_Current, Complex_Impedance):
	#print Complex_Power[0].strip('()j')
	Initialized_Loads = []*len(Complex_Power)
	
	#Loop through all loads in case, breaks real and imaginary components up
	for x in xrange(0, len(Load_Numbs)):
	
		
		if '+' in Complex_Power[x]:
			Present_Power = Complex_Power[x].strip('()j').split('+')
			
		elif '-' in Complex_Power[x]:
			Present_Power = Complex_Power[x].strip('()j').split('-')
			Present_Power[1] = '-' + Present_Power[1]
			
		if '+' in Complex_Current[x]:	
			Present_Current = Complex_Current[x].strip('()j').split('+') 
		elif '-' in Complex_Current[x]:
			Complex_Current = Complex_Current[x].strip('()j').split('-')			
			Complex_Current[1] = '-' + Complex_Current[1]	
			
		if '+' in Complex_Impedance[x]:
			Present_Impedance = Complex_Impedance[x].strip('()j').split('+') 
		elif '-' in Complex_Impedance[x]:
			Complex_Impedance = Complex_Impedance[x].strip('()j').split('-')			
			Complex_Impedance[1] = '-' + Complex_Impedance[1]	
			
			
		# Present_Power = Complex_Power[x].strip('()j').split('+')
		# Present_Current = Complex_Current[x].strip('()j').split('+') 
		# Present_Impedance = Complex_Impedance[x].strip('()j').split('+') 
		
		

		Real_Power = float(Present_Power[0]) #Real power for current bus
		Reactive_Power = float(Present_Power[1]) #Imaginary power for curret bus 
	
		Dif_Real_Load = Real_Power + Real_Power*random.uniform(-0.5,0.5) #Increase/Decrease current bus real load by up to 50%
		Dif_Reactive_Load = Reactive_Power + Reactive_Power*random.uniform(-0.5,0.5) #Increase/Decrease current bus imaginary load by up to 50%
		
		psspy.load_chng_4(int(Load_Numbs[x]),Load_IDs[x],[_i,_i,_i,_i,_i,_i],[ Dif_Real_Load, Dif_Reactive_Load,_f,_f,_f,_f]) #Tell PSS/e to change the bus loading
		
		#print Dif_Real_Load

#Solve the steady state case, returns P.U. voltages, if solution contains values under 0.95 or above 1.05, minimum and maximum bus voltages
def Solve_Steady():
	Ok_Solution = 1
	psspy.fnsl([0,0,0,1,1,0,99,0]) #Perform Solution
	
	ierr, rarray = psspy.abusreal(-1, 2, 'PU') #Returm P.U. voltages of buses after solution
	#print rarray
	if min(rarray[0]) < 0.95 or max(rarray[0]) > 1.051: #Find if voltages fall within OK operating ranges (1.051 due to some voltage held buses being specified to 1.05)
		Ok_Solution = 0
	
	
	return rarray, Ok_Solution, min(rarray[0]), max(rarray[0])

def reward_function(rarray):
    pass

#Change our PSS/e array to a list
def PSSE_Arrays2_List(In_Array):
	Temp_1 = ''.join(str(e) for e in In_Array)
	Temp_2 = Temp_1.replace("[", "")
	Temp_2 = Temp_2.replace("]", "")
	In_String = Temp_2.split(', ')
	Out_List = list()
	for x in range(0, len(In_String)):	
		Out_List.append(In_String[x])
	return Out_List

#Get rid of duplicates in a list
def Load_Duplicate_Fix(In_List):
	count = 1
	passedList = In_List
	List = list(In_List)
	List2 = list()
	for x in range(0,len(In_List)):
		Count_Name = str(count)
		List[x] = In_List[x] + '_' + Count_Name
		List2.append(Count_Name)
		if x < len(In_List) - 1:
			if In_List[x+1] in List[x]:
				count = count + 1
			else:
				count = 1
	return List, List2, passedList

	
#Create dictionary between two arrays
def Lists_2_Dicts(Array1, Array2):
	Dictionary = dict(zip(Array1, Array2))
	return Dictionary

#Return the load information for current case (all buses), returns how many buses, numbers of buses, complex loads of each bus, complex current of each bus, and complex impedance of each bus	
def Return_Load_Info():	
    ierr, Complex_Power = psspy.aloadcplx(-1, 4, 'MVANOM') #Obtain Complex Power of Loads
    ierr, Complex_Current = psspy.aloadcplx(-1, 4, 'ILNOM') #Obtain Complex Currents of Loads
    ierr, Complex_Impedance = psspy.aloadcplx(-1, 4, 'YLNOM') #Obtain Complex Impedances of Loads
    ierr, Load_Numbers = psspy.aloadint(-1, 4, 'NUMBER') #Obtain Load Numbers
    ierr, Load_Count = psspy.aloadcount(-1, 4) #Obtain Count of Loads
    ierr, Load_Amount = psspy.aloadcplx(-1, 4, 'MVAACT')
    return Load_Count, Load_Numbers, Load_Amount, Complex_Power, Complex_Current, Complex_Impedance

#Ignore
def Update_Dict_Vals(Dict2_Update):
	return

#Convert our complex power, currents and impedances to workable data types	
def ZIP_Loads():
	Load_Count, Load_Numbers, Load_Amount, Complex_Power, Complex_Current, Complex_Impedance = Return_Load_Info()

	Load_Numbers = PSSE_Arrays2_List(Load_Numbers)
	
	loadNumbsAndID, Load_ID_List, Load_Numbers = Load_Duplicate_Fix(Load_Numbers)
	#print Load_Numbers
	cplxPower = PSSE_Arrays2_List(Complex_Power)
	cplxCurrent = PSSE_Arrays2_List(Complex_Current)
	cplImpedance = PSSE_Arrays2_List(Complex_Impedance)
	
	return cplxPower, cplxCurrent, cplImpedance, Load_Numbers, Load_ID_List, Load_Amount


#Created dictionaries for load numbers to complex power/current/impedance
def ZIP_Loadings2_Dicts(Complex_Power, Complex_Current, Complex_Impedance, Load_Numbers):
	Complex_Power_Dict = Lists_2_Dicts(Load_Numbers, Complex_Power)
	Complex_Current_Dict = Lists_2_Dicts(Load_Numbers, Complex_Current)
	Complex_Impedance_Dict = Lists_2_Dicts(Load_Numbers, Complex_Impedance)
	
	return Complex_Power_Dict, Complex_Current_Dict, Complex_Impedance_Dict

#Returns the branch information of case, strucuted as follows: array[0] and array2[0] represents a bus number that is attached to another bus number
def branchData():
	
	ierr, array = psspy.aflowint(-1, -1, -1, 2, 'FROMNUMBER') #Get all buses that are the from bus 
	ierr, array2 = psspy.aflowint(-1, -1, -1, 2, 'TONUMBER') #Get all buses that are the to bus, structured the same as from buses meaning array2[0] is a bus attached to array[0] another bus
	
	return array, array2

#Delete double counted lines, return locations of these double counted values
def deleteDuplicateLineInstances(fromBranch, toBranch):
	newFromBranches = []
	newToBranches = []
	badIndex = []
	goodIndex = []
	for x in range(0, len(fromBranch[0])):


		flag = 1
		for y in range(0, len(fromBranch[0])):
			if fromBranch[0][x] == toBranch[0][y] and toBranch[0][x] == fromBranch[0][y]:
				badIndex.append(y)
				
		if x not in badIndex:
			goodIndex.append(x)
			newFromBranches.append(fromBranch[0][x])
			newToBranches.append(toBranch[0][x])
	
	#print goodIndex
	return [newFromBranches], [newToBranches], goodIndex
	
	
#Remove bad locations that contain doulbe counted lines	
def deleteBadIndices(goodIndeces, listToModify):

	newList = []
	for x in range(0, len(listToModify)):
		if x in goodIndeces:
			#print x
			newList.append(listToModify[x])
		
	return [newList]
		
		
#Keep track of list "ID" meaning we label lines in case there are multiple lines attached between two buses.  If two lines are tied to buses '101' and '102' we label the lines as '1' and '2'
def lineIDListCreation(newFromBranches, newToBranches):
	listOfLineIDs = []
	listOfLineIDs.append(1)
	for x in range(1, len(newFromBranches)):
		
		
		if newFromBranches[x] == newFromBranches[x-1] and newToBranches[x] == newToBranches[x-1]:

			listOfLineIDs.append(2)
		
		else:
			listOfLineIDs.append(1)
				

	
	return listOfLineIDs

#With our predefined lines, trip them which most commonly is used to island a case
def createIsland(toBus, fromBus):
	for x in range(0, len(toBus)):
		print(toBus[x], fromBus	[x])
		psspy.branch_chng(fromBus[x], toBus[x],r"""1""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
	psspy.save(r"""C:\Users\psse\aires\Carter_Case\Poland_Case\Islanded.sav""")	

	
#Return all lines that are attached at interconnected locations for a zone.  We return the lines that are junctions to a certain zone and any other different zone (this allows for getting lines to island our case)
def branchDataZone(zone):
	toBus = []
	fromBus = []
	busInZone = []
	
	busZones = defaultdict(list)
	ierr, array = psspy.aflowint(-1, -1, -1, 2, 'FROMNUMBER')
	ierr, array2 = psspy.aflowint(-1, -1, -1, 2, 'TONUMBER')
	#for x in range(0, len(array[0])):
		#print array[0][x]
		#print array2[0][x]
	ierr, busNumber = psspy.abusint(-1, 2, 'NUMBER')
	ierr, busZone = psspy.abusint(-1, 2, 'ZONE')
	#print busNumber
	
	for y in range(0, len(busNumber[0])):
		busZones[busNumber[0][y]].append(busZone[0][y])
	
	for a in range(0, len(busZones)):
		if busZones[busNumber[0][a]][0] == zone:
			busInZone.append(busNumber[0][a])
	
	for z in range(0, len(array[0])):
		if busZones[array[0][z]] == busZones[array2[0][z]]:
			pass
		else:
			#print 'Branch between different zones'
			if busZones[array[0][z]][0] == zone:
				toBus.append(array[0][z])
				fromBus.append(array2[0][z])
				#for i in range(0, len(busInZone)):
				
				if array[0][z] not in busInZone:
					busInZone.append(array[0][z])
				if array2[0][z] not in busInZone:
					busInZone.append(array2[0][z])		
				
			
			if busZones[array2[0][z]][0] == zone:
				toBus.append(array[0][z])
				fromBus.append(array2[0][z])	

				
				if array[0][z] not in busInZone:
					busInZone.append(array[0][z])
				if array2[0][z] not in busInZone:
					busInZone.append(array2[0][z])		
				



	
	return toBus, fromBus, busInZone, busZones, array, array2

#Return the Load, generation and losses for zones in case
def zoneTotalsPsseFunc():
	ierr, zonesMWload = psspy.azonereal(-1, 2, 'PLOAD')
	ierr, zonesMVARload = psspy.azonereal(-1, 2, 'QLOAD')
	ierr, zonesMWgen = psspy.azonereal(-1, 2, 'PGEN')
	ierr, zonesMVARgen = psspy.azonereal(-1, 2, 'QGEN')	
	ierr, zonesMWloss = psspy.azonereal(-1, 2, 'PLOSS')
	ierr, zonesMVARloss = psspy.azonereal(-1, 2, 'QLOSS')		

	return zonesMWload, zonesMVARload, zonesMWgen, zonesMVARgen, zonesMWloss, zonesMVARloss
	
#Island our case using no dynamics
def islandCaseNoDynamics(zone):
	Initialize_Case()

	toBus, fromBus, busInZone, busZones, fromTotalBus, toTotalBus = branchDataZone(zone)
	#print fromBus
	createIsland(toBus, fromBus)
	
	#print len(busInZone)
	
	ierr, buses = psspy.tree(1, 1)
	while buses != 0:
		ierr, buses = psspy.tree(2, 1)	
	
	psspy.save(r"""C:\Users\psse\aires\Carter_Case\RTS-96\NewOpCases\FirstCase\FirstIslanded.sav""")	
	
	one, two, localMin, localMax = Solve_Steady()
	zonesMWload, zonesMVARload, zonesMWgen, zonesMVARgen, zonesMWloss, zonesMVARloss = zoneTotalsPsseFunc()
	microGridRealPower, microGridReactivePower, microGridRealLoad, microGridReactiveLoad, microGridRealLoss, microGridReactiveLoss, mainGridRealPower, mainGridReactivePower, mainGridRealLoad, mainGridReactiveLoad, mainGridRealLoss, mainGridReactiveLoss = inOutMicroGridTotals(zone, zonesMWload, zonesMVARload, zonesMWgen, zonesMVARgen, zonesMWloss, zonesMVARloss)

	
#Get power, generation and losses for a certain 'microgrid' and main grid + print it out.
def inOutMicroGridTotals(zone, zonesMWload, zonesMVARload, zonesMWgen, zonesMVARgen, zonesMWloss, zonesMVARloss):
    microGridRealPower = zonesMWgen[0][zone-1]
    microGridReactivePower = zonesMVARgen[0][zone-1]
    microGridRealLoad = zonesMWload[0][zone-1]
    microGridReactiveLoad = zonesMVARload[0][zone-1]
    microGridRealLoss = zonesMWloss[0][zone-1]
    microGridReactiveLoss = zonesMVARloss[0][zone-1]
    mainGridRealPower = sum(zonesMWgen[0]) - microGridRealPower
    mainGridReactivePower = sum(zonesMVARgen[0]) - microGridReactivePower
    mainGridRealLoad = sum(zonesMWload[0]) - microGridRealLoad
    mainGridReactiveLoad = sum(zonesMVARload[0]) - microGridReactiveLoad
    mainGridRealLoss = sum(zonesMWloss[0]) - microGridRealLoss
    mainGridReactiveLoss = sum(zonesMVARloss[0]) - microGridReactiveLoss
    return microGridRealPower, microGridReactivePower, microGridRealLoad, microGridReactiveLoad, microGridRealLoss, microGridReactiveLoss, mainGridRealPower, mainGridReactivePower, mainGridRealLoad, mainGridReactiveLoad, mainGridRealLoss, mainGridReactiveLoss

'''
	print '---------------------------Micro grid information---------------------------'
	print 'Total real power: %d MW' %microGridRealPower
	print 'Total reactive power: %d MVAR' %microGridReactivePower
	print 'Total real load: %d MW' %microGridRealLoad
	print 'Total reactive load: %d MVAR' %microGridReactiveLoad
	print 'Total real loss: %d MW' %microGridRealLoss
	print 'Total reactive loss: %d MVAR' %microGridReactiveLoss
	print '---------------------------Main grid information----------------------------'
	print 'Total real power: %d MW' %mainGridRealPower
	print 'Total reactive power: %d MVAR' %mainGridReactivePower
	print 'Total real load: %d MW' %mainGridRealLoad
	print 'Total reactive load: %d MVAR' %mainGridReactiveLoad
	print 'Total real loss: %d MW' %mainGridRealLoss
	print 'Total reactive loss: %d MVAR' %mainGridReactiveLoss
'''
	#return microGridRealPower, microGridReactivePower, microGridRealLoad, microGridReactiveLoad, microGridRealLoss, microGridReactiveLoss, mainGridRealPower, mainGridReactivePower, mainGridRealLoad, mainGridReactiveLoad, mainGridRealLoss, mainGridReactiveLoss
	
#Solve our steady state solution, can implement initial condition/operating point changes.
def steadyStateChangeInitSolution():
    inputFile = 'Dynamics.dyr'
    Initialize_Case()

    cplxPower, cplxCurrent, cplImpedance, Load_Numbers, Load_ID_List, Load_Amount = ZIP_Loads()
    print(Load_Numbers)
    print(Load_ID_List)
    print(cplImpedance)
    print(cplxPower)
    print(Load_Amount)

    Change_Init_Conds(Load_Numbers, Load_ID_List, cplxPower, cplxCurrent, cplImpedance)
    #Change_OpPoint(Load_Numbers, Load_ID_List, cplxPower, cplxCurrent, cplImpedance)

    rarray, Ok_Solution, localMin, localMax = Solve_Steady()
    print(rarray)
    print(Ok_Solution)
    return
    fromBranch, toBranch = branchData()
    newFromBranches, newToBranches, keepIndex = deleteDuplicateLineInstances(fromBranch, toBranch)
    ierr, rarray = psspy.aflowreal(-1, -1, -1, 2, 'MVA')
    lineMVAlist = deleteBadIndices(keepIndex, rarray[0])

    return lineMVAlist, Ok_Solution, localMin, localMax
#Returns pmu vals
def steadyStateSolve(loads, bus_name, load_id):

    for load in range(load):
        psse.load_chng_4(bus_name[load], load_id[load], [_i, _i, _i, _i, _i, _i], [loads[load].real, loads[load].imag, _f, _f, _f, _f])

    rarray, Ok_Solution, localMin, localMax = Solve_Steady()
    return rarray

#returns reward values
def reward(pmu, loads, max_loads, reward_coefficient):
    total_reward = 0.0
    for p in range(len(pmu)):
        if (p < 0.94 or p > 1.06):
            total_reward -= reward_coefficient[p]

    for i in range(len(loads)):
        total_reward += (loads[i] / max_loads[i]) * reward_coefficient[i]

    return total_reward


def begin_policy_rollout():
    Initialize_Case()

    cplxPower, cplxCurrent, cplImpedance, Bus_ids, Load_Numbers, Load_Amount = ZIP_Loads()

    load_buses = []
    load_bus_ids = []

    for j in range(len(Bus_ids)):
        pass



#This just shuffles loads around in the case, it's hard coded for RTS-96 at the moment so it won't work with savnw.  I'll modify for more universality.
def Change_OpPoint(Load_Numbs, Load_IDs, Complex_Power, Complex_Current, Complex_Impedance):
	Real_Power1 = []
	Reactive_Power1 = []
	Real_Power2 = []
	Reactive_Power2 = []
	Real_Power3 = []
	Reactive_Power3 = []	
	
	for x in xrange(0, len(Load_Numbs)):
	
		if '+' in Complex_Power[x]:
			Present_Power = Complex_Power[x].strip('()j').split('+')
		elif '-' in Complex_Power[x]:
			Present_Power = Complex_Power[x].strip('()j').split('-')
			Present_Power[1] = '-' + Present_Power[1]
			
		if '+' in Complex_Current[x]:	
			Present_Current = Complex_Current[x].strip('()j').split('+') 
		elif '-' in Complex_Current[x]:
			Present_Current = Complex_Current[x].strip('()j').split('-')			
			Present_Current[1] = '-' + Present_Current[1]	
			
		if '+' in Complex_Impedance[x]:
			Present_Impedance = Complex_Impedance[x].strip('()j').split('+') 
		elif '-' in Complex_Impedance[x]:
			Present_Impedance = Complex_Impedance[x].strip('()j').split('-')			
			Present_Impedance[1] = '-' + Present_Impedance[1]	
			
			
		# Present_Power = Complex_Power[x].strip('()j').split('+')
		# Present_Current = Complex_Current[x].strip('()j').split('+') 
		# Present_Impedance = Complex_Impedance[x].strip('()j').split('+') 
		
		
		if int(Load_Numbs[x]) < 200:
			Real_Power1.append(float(Present_Power[0]))
			Reactive_Power1.append(float(Present_Power[1]))
			
		elif int(Load_Numbs[x]) < 300:
			Real_Power2.append(float(Present_Power[0]))
			Reactive_Power2.append(float(Present_Power[1]))	
			
		else:
			Real_Power3.append(float(Present_Power[0]))
			Reactive_Power3.append(float(Present_Power[1]))
			
	
	random.shuffle(Real_Power1)
	random.shuffle(Reactive_Power1)
	random.shuffle(Real_Power2)
	random.shuffle(Reactive_Power2)
	random.shuffle(Real_Power3)
	random.shuffle(Reactive_Power3)	
	
	location = 0
	location2 = 0
	location3 = 0
	
	for y in range(0, len(Real_Power1)):
		psspy.load_chng_4(int(Load_Numbs[y]),Load_IDs[y],[_i,_i,_i,_i,_i,_i],[ Real_Power1[y], Reactive_Power1[y],_f,_f,_f,_f])
		location += 1

	for w in range(location, len(Real_Power2) + location):
		psspy.load_chng_4(int(Load_Numbs[w]),Load_IDs[w],[_i,_i,_i,_i,_i,_i],[ Real_Power2[location2], Reactive_Power2[location2],_f,_f,_f,_f])
		location2 += 1
		
	for z in range(location + location2, len(Load_Numbs)):
		psspy.load_chng_4(int(Load_Numbs[z]),Load_IDs[z],[_i,_i,_i,_i,_i,_i],[ Real_Power3[location3], Reactive_Power3[location3],_f,_f,_f,_f])
		location3 += 1
		
		
def main():
	steadyStateChangeInitSolution() #Basic case to solve the case with no dynamcis



if __name__ == "__main__": 
	main()	



