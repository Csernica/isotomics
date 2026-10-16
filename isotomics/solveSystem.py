import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import sympy as sy
from scipy.linalg import null_space
from tqdm import tqdm

from . import basicDeltaOperations as op

def perturbStandard(standardData, theory = True):
    """
    Takes a dictionary with standard data. For each fragment, perturbs every measurement according to its experimental error, then renormalizes. Calculates correction factors by comparing these perturbed values to the predicted abundance of each peak.

    Args:
        standardData: A dictionary; keys are mass selections ("M1", "M2") then fragment Keys ("full", "44"), then information about substitutions, observed abundances, predicted abundances, and errors.
        theory: A boolean. If true, calculates correction factors.

    Returns:
        standardData: The same dictionary as the input, with entries for the perturbed observation as well as correction factors.

    Examples:
        >>> # add example
    """
    #146 us   
    for massSelection in standardData.keys():
        for frag, data in standardData[massSelection].items():
            observed = np.array(data['Observed Abundance'])
            error = np.array(data['Error'])

            #perturb
            perturbed = np.random.normal(observed,error)
            perturbed /= perturbed.sum()
                
            if theory:
                standardData[massSelection][frag]['Perturbed'] = perturbed
                standardData[massSelection][frag]['Correction Factor'] = perturbed / np.array(data['Predicted Abundance'])
                
            else: 
                standardData[massSelection][frag]['Perturbed'] = perturbed
            
    return standardData

def perturbSampleError(sampleData):
    """
    Perturbs sample data according to observed experimental errors. For each mass selection, for each fragment, perturbs based on experimental error, then renormalizes. This can be seen as a companion function to perturbStandard; differs in that it does not calculate correction factors and outputs a new dictionary.

    Args:
        sampleData: A dictionary; keys are mass selections ("M1", "M2") then fragment Keys ("full", "44"), then information about substitutions, observed abundances, and errors.

    Returns:
        perturbedSample: A dictionary; keys are mass selections, then fragment keys. Contains information about observed abundance and substitutions.

    Examples:
    """
    perturbedSample = {}
    for massSelection in sampleData.keys():
        perturbedSample[massSelection] = {}
        for fragKey, fragData in sampleData[massSelection].items():
            perturbedSample[massSelection][fragKey] = {}
            observed = np.array(fragData['Observed Abundance'])
            error = np.array(fragData['Error'])

            #perturb and renormalize
            perturbed = np.random.normal(observed,error)
            perturbed /= perturbed.sum()
            
            perturbedSample[massSelection][fragKey] = {'Observed Abundance': perturbed,
                                                       'Subs': fragData['Subs']}
            
    return perturbedSample

def perturbSampleCorrectionFactors(perturbedSample, perturbedStandard, renormalize = True):
    """
    Applies the calculated correction factors from the standard to our sample. The most important choice here is the renormalize boolean. As discussed in the appendix, renormalizing at this step can remove the "W" factor, leading to a more accurate solution. In some instances, one may wish not to renormalize. By default it should be True.

    Args:
        perturbedSample: A dictionary, the output of perturbSampleError. Keys are mass selections, then fragment keys. Contains information about observed abundance and substitutions.
        perturbedStandard: A dictionary, the output of perturbStandard. Keys are mass selections, then fragment keys. Contains information about the correction factors for each peak.
        renormalize: A boolean. See description.

    Returns:
        correctedSample: A dictionary; keys are mass selections, then fragment keys. Contains information about observed abundance and substitutions.

    Examples:
    """
    correctedSample = {}
    for massSelection in perturbedSample.keys():
        correctedSample[massSelection] = {}
        for fragKey, fragData in perturbedSample[massSelection].items():
            correctedSample[massSelection][fragKey] = {}
            
            observed = fragData['Observed Abundance']
            corrected = observed / perturbedStandard[massSelection][fragKey]['Correction Factor']
            if renormalize:
                corrected /= corrected.sum()
            
            correctedSample[massSelection][fragKey] = {'Observed Abundance': corrected,
                                                       'Subs': fragData['Subs']}
            
    return correctedSample
            
def perturbSampleOCorrection(correctedSample, OCorrection, perturbOverrideList = []):
    """
    Applies M+N Relative abundance correction factors to the sample. These factors scale the observed M+N Relative abundances to correct for unobserved peaks.

    Args:
        correctedSample: A dictionary, the output of perturbSampleCorrectionFactors. Keys are mass selections, then fragment keys. Contains information about observed abundance and substitutions.
        OCorrection: A dictionary, giving a M+N Relative abundance correction factor for each fragment of each mass selection.
        perturbOverrideList: For M1 Iterated correction, do not want to use all mass selections, only M1; set to ['M1']

    Returns:
        OCorrectedSample: A dictionary; keys are mass selections, then fragment keys. Contains information about observed abundance and substitutions.

    Examples:
    """
    OCorrectedSample = {}
    
    for massSelection in correctedSample.keys():
        if perturbOverrideList == [] or massSelection in perturbOverrideList:
            OCorrectedSample[massSelection] = {}
            for fragKey, fragData in correctedSample[massSelection].items():
                OCorrectedSample[massSelection][fragKey] = {}

                observed = fragData['Observed Abundance']
                correctedPA = observed * OCorrection[massSelection][fragKey]

                OCorrectedSample[massSelection][fragKey] = {'Observed Abundance': correctedPA,
                                                           'Subs': fragData['Subs']}
            
    return OCorrectedSample

def perturbSample(sampleData, perturbedStandard, OCorrection, experimentalOCorrectList = [], 
                    correctionFactors = True, abundanceCorrect = True, explicitOCorrect = {}, 
                    perturbOverrideList = []):
    """
    Takes sample data and perturbs it multiple ways--first perturbs experimental error, then applies fractionation correction factors, then applies M+N Relative abundance correction factors. Finally processes the perturbed sample into a dataframe to be looped into the matrix solver.

    Args:
        sampleData: A dictionary; keys are mass selections ("M1", "M2") then fragment Keys ("full", "44"), then information about substitutions, observed abundances, and errors.
        perturbedStandard: A dictionary, the output of perturbStandard. Keys are mass selections, then fragment keys. Contains information about the correction factors for each peak.
        OCorrection: A dictionary, giving a M+N Relative abundance correction factor for each fragment of each mass selection.
        experimentalOCorrectList: A list, containing information about which peaks to use experimental, rather than theoretical, M+N Relative abundance corrections.
        correctionFactors: A boolean, determines whether to apply sample/standard correction factors.
        abundanceCorrect: A boolean, determines whether to apply observed abundance correction factors.
        perturbOverrideList: perturbSample will automatically perturb all sample acquisitions (M1, M2, M3, M4); in some cases, e.g. when doing an iterated correction for M1, we do not want to perturb all, only M1. This can be specified with this list. (E.g. ['M1'])
        explicitOCorrect: For each MNKey and each fragment, may provide bounds on reasonable O correction values.

    Returns:
        measurementData: A dictionary containing a dataframe for each M+N experiment. The dataframe gives the final corrected relative abundances for each peak of each fragment.

    Examples:
    """
    perturbedSample = perturbSampleError(sampleData)
    
    if correctionFactors:
        perturbedSample = perturbSampleCorrectionFactors(perturbedSample, perturbedStandard, renormalize = abundanceCorrect)
    
    if abundanceCorrect:
        for c in experimentalOCorrectList:
            OCorrection = experimentalOCorrection(c['MNKey'], c['fragToCorrect'], c['subToCorrect'], perturbedSample, OCorrection, c['fragsToBenchmarkFrom'])
            
            MNKey = c['MNKey']
            fragKey = c['fragToCorrect']
            
            if MNKey in explicitOCorrect:
                if fragKey in explicitOCorrect[MNKey]:
                    if 'Bounds' in explicitOCorrect[MNKey][fragKey]:
                        if OCorrection[MNKey][fragKey] <= explicitOCorrect[MNKey][fragKey]['Bounds'][0]:
                            OCorrection[MNKey][fragKey] = explicitOCorrect[MNKey][fragKey]['Bounds'][0]

                        if OCorrection[MNKey][fragKey] >= explicitOCorrect[MNKey][fragKey]['Bounds'][1]:
                            OCorrection[MNKey][fragKey] = explicitOCorrect[MNKey][fragKey]['Bounds'][1]

        perturbedSample = perturbSampleOCorrection(perturbedSample, OCorrection, perturbOverrideList = perturbOverrideList)

        
    #Prepare to output as dictionary of dataFrames
    unpacked = {}
    for massSelection, MSData in perturbedSample.items():
        unpacked[massSelection] = {}
        for fragKey, fragData in MSData.items():
            unpacked[massSelection][fragKey] = {}
            for i, subKey in enumerate(fragData['Subs']):
                unpacked[massSelection][fragKey][subKey] = perturbedSample[massSelection][fragKey]['Observed Abundance'][i]
    
    #output as dictionary of dataframes
    measurementData = {}
    for massSelection, MSData in unpacked.items():
        measurementData[massSelection] = pd.DataFrame.from_dict(MSData)
        measurementData[massSelection].fillna(0,inplace = True)

    return measurementData

def OValueCorrectTheoretical(predictedMeasurement, processSample, massThreshold = 4, debug = False):
    """
    A theoretical method of calculating M+N Relative abundance correction factors. Looks at predicted measurements from a stochastic distribution and input deltas and sees how much M+N relative abundance is actually observed in the measurement.

    Args:
        predictedMeasurement: A computed dataset for some input delta values. Should be a perfect dataset.
        processSample: The processed sample input. Checks which peaks were actually observed in order to determine the size of the M+N Relative abundance correction.
        massThreshold: The highest M+N experiment to compute M+N Relative abundance corrections for.
        debug: If true, prints the peaks from predicted measurement and if they were found successfully.

    Returns:
        OValueCorrection: A dictionary, where keys are mass selections, then fragment keys, keyed to floats, where the float gives the M+N Relative abundance correction for that fragment.

    Examples:
        >>> import pandas as pd
        >>> from isotomics import calcIsotopologues as ci
        >>> from isotomics import fragmentAndSimulate as fas
        >>> from isotomics import readInput as ri
        >>> from isotomics import solveSystem as ss
        >>> molecular_df = pd.DataFrame({'IDS': ['C', 'O', 'S'], 'Number': [2, 1, 1], 'deltas': [-10, 0, 0]})
        >>> byAtom = ci.inputToAtomDict(molecular_df, disable=True, M1Only=False)
        >>> MNDict = ci.massSelections(byAtom, massThreshold=1)
        >>> fragmentationDictionary = {'frag': {'01': {'subgeometry': [1, 'x', 1], 'relCont': 1}}}
        >>> predictedMeasurement, _ = fas.predictMNFragmentExpt(MNDict, molecular_df, fragmentationDictionary, abundanceThreshold=0)
        >>> # Simulate a sample where 'Unsub' was not resolved/observed
        >>> smpMeasurement = {
        ...     'M1': {'frag': {
        ...         '13C': {'Adj. Rel. Abundance': predictedMeasurement['M1']['frag']['13C']['Adj. Rel. Abundance']},
        ...         '33S': {'Adj. Rel. Abundance': predictedMeasurement['M1']['frag']['33S']['Adj. Rel. Abundance']},
        ...     }}
        ... }
        >>> processSample = ri.readComputedData(smpMeasurement, error=0.001)
        >>> OCorrection = ss.OValueCorrectTheoretical(predictedMeasurement, processSample, massThreshold=1)
        >>> OCorrection  # < 1.0 because 'Unsub' rel. abundance is missing from observed peaks
        {'M1': {'frag': 0.98750054}}
    """
    OValueCorrection = {}
    for i in range(1, massThreshold+1):
        OValueCorrection['M' + str(i)] = {}

    for MNKey in OValueCorrection.keys():
        for fragKey, fragInfo in predictedMeasurement[MNKey].items():
            relAbund = 0
            for subKey, subData in fragInfo.items():
                if debug:
                    print(MNKey + ' ' + fragKey + ' ' + subKey)
                #If we actually observed the peak
                if MNKey in processSample:
                    if fragKey in processSample[MNKey]:
                        if subKey in processSample[MNKey][fragKey]['Subs']:
                            relAbund += subData['Rel. Abundance']
                            if debug: 
                                print("FOUND")
                                print(relAbund)
            #round to avoid floating point error
            OValueCorrection[MNKey][fragKey] = round(relAbund,8)    
            
    return OValueCorrection

def modifyOValueCorrection(OValueCorrection, variableOCorrect, MNKey, explicitOCorrect = {}, amount = 0.002):
    """
    Perturbs the M+N Relative abundance correction factors, for example if they are only approximately known.

    Args:
        OValueCorrection: A dictionary, where keys are mass selections and values are dictionaries; in each dictionary, the keys are fragment keys and values are floats, where the float gives the observed abundance correction for that fragment.
        variableOCorrect: A copy of the OValueCorrection dictionary, so we are not modifying it directly.
        MNKey: "M1", "M2", etc.
        amount: The size of the perturbation in relative terms (e.g. 2 per mil)
        explicitOCorrect: An override dictionary, where an explicit distribution can be set for each fragment, rather than using the input from OValueCorrection and the calculated standard error.

    Returns:
        variableOCorrect: A perturbed copy of the OValueCorrection dictionary.

    Examples:
    """
    for fragKey, OFactor in OValueCorrection[MNKey].items():
        corrected = False
        #if == 1, no correction performed
        if OFactor != 1:
            if MNKey in explicitOCorrect:
                if fragKey in explicitOCorrect[MNKey]:
                    corrected = True
                    v = np.random.normal(explicitOCorrect[MNKey][fragKey]['Mu,Sigma'][0], explicitOCorrect[MNKey][fragKey]['Mu,Sigma'][1])
                    if 'Bounds' in explicitOCorrect[MNKey][fragKey]:
                        if v <= explicitOCorrect[MNKey][fragKey]['Bounds'][0]:
                            v = explicitOCorrect[MNKey][fragKey]['Bounds'][0]

                        if v >= explicitOCorrect[MNKey][fragKey]['Bounds'][1]:
                            v = explicitOCorrect[MNKey][fragKey]['Bounds'][1]

                    
                    variableOCorrect[MNKey][fragKey] = v
                    
            if corrected == False:
                variableOCorrect[MNKey][fragKey] = np.random.normal(OFactor, OFactor*amount)
            
    return variableOCorrect

def experimentalOCorrection(MNKey, fragToCorrect, subToCorrect, perturbedSample, OValueCorrection, fragsToBenchmarkFrom):
    """
    A complicated procedure to generate experimental M+N Relative abundance correction factors in some fringe cases. If you don't have a good understanding of what this is trying to do, you probably should not be using it.

    Args:
        MNKey: "M1", "M2", etc.
        fragToCorrect: A string, giving the fragment that should be corrected (e.g. "61")
        subToCorrect: The substitution to use to correct; we anticipate the M+N Relative abundance observed for this substitution in this fragment should be the same as the M+N Relative abundance observed for this substitution in another fragment.
        perturbedSample: The perturbed sample data, a dictionary.
        OValueCorrection: A dictionary giving the M+N Relative abundance correction factors.
        fragsToBenchmarkFrom: A list, giving fragments where we expect to have accurately observed the M+N Relative abundance of the substitution in question. E.g. ['full', '133','104]

    Returns:
        OValueCorrection: The same dictionary, updated with the experimental correction.

    Examples:
    """
    #first, check for fragments which did not need M+N Relative abundance correction. We will use these to benchmark those
    #that do. Note that this procedure will mostly be applicable to M+1 measurements, as in other cases, there
    #will always be lost ion beams. 
    perfect = []
    for fragKey in fragsToBenchmarkFrom:
        correction = OValueCorrection[MNKey][fragKey]
        if correction <= 0.999999 or correction >= 1.000001:
            raise Exception("Trying to experimentally correct fragment " + fragToCorrect + " with fragment " + fragKey
                            + ". However, " + fragKey + " has a theoretical M+N Relative abundance correction itself. The procedure will not work.")
        perfect.append(fragKey)

    correctionFactors = []
    #for each fragment benchmark, correct the M+N Relative abundance. Finish by taking the average of these. 
    for fragKey, fragData in perturbedSample[MNKey].items():
            #if this fragment did not require M+N Relative abundance correction
            if fragKey in perfect:
                subIndexBenchmark = perturbedSample[MNKey][fragKey]['Subs'].index(subToCorrect)
                observationBenchmark = perturbedSample[MNKey][fragKey]['Observed Abundance'][subIndexBenchmark]
                
                subIndexToCorrect = perturbedSample[MNKey][fragToCorrect]['Subs'].index(subToCorrect)
                observationToCorrect = perturbedSample[MNKey][fragToCorrect]['Observed Abundance'][subIndexToCorrect]
               
                c = observationBenchmark / observationToCorrect
                correctionFactors.append(c)

    CF = np.array(correctionFactors).mean()
    
    OValueCorrection[MNKey][fragToCorrect] = CF
    
    return OValueCorrection

def constructMatrix(Isotopologues, smp, MNKey, fragmentationDictionary, includeSubs = [], omitSubs = []):
    """
    Constructs the matrix and the measurement vector for the Monte Carlo method to solve. We could increase speed by only constructing the composition matrix once and tracking how to reconstruct the measurement vector, rather than solving it each time. But we have not implemented this.

    Args:
        Isotopologues: A dataFrame containing isotopologues and information about their fragmentation.
        smp: A dataframe giving the corrected and standardized observations by fragments and isotope.
        MNKey: A string, telling us which M+N experiment we are solving ("M1", "M2", etc.)
        fragmentationDictionary: A dictionary, e.g. {'full': {'01': {'subgeometry': [1, 1, 1, 1, 1, 1], 'relCont': 1}},
                                                     '44': {'01': {'subgeometry': [1, 'x', 'x', 1, 1, 'x'], 'relCont': 1}}} which gives information about the fragments, their subgeometries and relative contributions.
        includeSubs: A list of isotopes, if we want to include only certain isotopes in the matrix. If it is nonempty, only isotopes in the list will be included in the matrix. Generally should be empty.
        omitSubs: A list of isotopes, if we wish to omit certain isotopes from the matrix. If it is nonempty, isotopes in the list will not be included in the matrix. Generally should be empty.

    Returns:
        comp: The composition matrix as a numpy array. Columns are isotopologues, rows are observations.
        meas: The measurement vector as a numpy array. Rows correspond to observations.

    Examples:
    """
    CMatrix = []
    MeasurementVector = []
    
    CMatrix.append([1]*len(Isotopologues.index))
    MeasurementVector.append(1)
    
    for fragKey, fragInfo in fragmentationDictionary.items():
        #One matrix/measurement vector row per sub per fragment
        for sub, v in smp[fragKey].items():
            #includeSubs lets us select only specific elements to use in constructing the matrix
            if len(includeSubs) == 0 or sub in includeSubs:
                #omitSubs omits certain isotopes from the matrix
                if sub not in omitSubs:
                    #If the observed intensity of a fragment is 0, we do not include it
                    if v != 0:
                        cFull = np.zeros(len(Isotopologues.index))
                        #The composition matrix may have contributions from multiple subgeometries
                        for subFrag, subFragInfo in fragInfo.items():
                            IsotopologueFragments = Isotopologues[fragKey + '_' + subFrag + ' Subs']
                            cFull = cFull + IsotopologueFragments.isin([sub]).to_numpy() * subFragInfo['relCont']

                        MeasurementVector.append(v)
                        CMatrix.append(cFull)
                
    comp = np.array(CMatrix,dtype=float)
    meas = np.array(MeasurementVector,dtype = float)
    
    return comp, meas

def sanitizeMatrix(M, eps = 10**-8, full = False):
    """
    One inefficient attempt to avoid floating point errors. This checks every entry of a matrix to see if it is sufficiently close to some integer value. If it is, it rounds it to that integer. This is useful to run on matrices that have been manipulated and may be carrying floating point errors.

    Args:
        M: The matrix to sanitize.
        eps: A float. Entries with absolute value less than eps will be made 0.
        full: A boolean. If true, sanitizes all columns. If false, does not sanitize the final (augmented) column. This only works with AugAmount = 1 in GJElim, something to be improved in the future.

    Returns:
        M: The sanitized matrix.

    Examples:
    """
    rows, cols = M.shape
    
    for i, row in enumerate(M):
        for j, col in enumerate(row):
            if not full:
                diff = np.abs(col - round(col))
                #Don't fix last column of Aug Matrix
                if j < (cols-1):
                    if diff < eps and diff > 0:
                        M[i][j] = round(col)
            if full:
                diff = np.abs(col - round(col))
                if j < (cols):
                    if diff < eps and diff > 0:
                        M[i][j] = round(col)
                    
    return M

def GJElim(Matrix, augMatrix = False, AugAmount = 1, store = False, sanitize = False, eps = 10**-8):
    """
    A Gauss-Jordan Elimination algorithm. Useful to track the results of underconstrained systems, to see which isotopologues covary. We can determine which isotopologues are solved for by interrogating the null space of the GJ solution.

    Args:
        Matrix: The matrix to eliminate.
        augMatrix: Whether the matrix is augmented or not.
        AugAmount: The number of columns which are augmented, e.g. to the right of the line.
        store: A boolean. If true, we store the results of every step of the elimination and output them, for debugging.
        sanitize: A boolean. If true, every step, checks the entire matrix and sets entries which are very close to 0 as 0. This helps avoid floating point errors.
        eps: A float. Values with absolute value < epsilon are considered 0s and are not selected as pivots. Helps avoid floating point errors.

    Returns:
        M: The solved matrix after GJ elimination
        rank: An integer, the rank of the solved matrix
        storage: If store == True, returns a list giving the matrix at every step, for debugging.

    Examples:
    """
    M = Matrix.copy()
    rows, cols = M.shape

    r = 0
    c = 0
    pivotRow = None
    
    if augMatrix == True:
        colLimit = cols - AugAmount
    else:
        colLimit = cols
        
    rank = 0
    storage = []
    if store:
        storage.append(M.copy())
    while r < rows and c < colLimit:
        #If there is a nonzero entry in the column, then pivot and eliminate. 
        #Count only values above threshold as nonzero, to avoid e.g. picking "10**-15" as a nonzero entry
        if True in (np.abs(M[r:,c]) > eps):
            pivotRow = (M[r:,c]!=0).argmax(axis=0) + r
            rank += 1

            M[[r, pivotRow]] = M[[pivotRow, r]]

            M[r] = M[r]/ M[r,c]

            for i in range(1,rows-r):
                M[r+i] -= (M[r+i,c]/M[r,c] * M[r])

            for j in range(0,r):
                M[j] -= M[j,c]/M[r,c] * M[r]
                
            r += 1

        c += 1
        if store:
            storage.append(M.copy())
            
        if sanitize: 
            sanitizeMatrix(M)

    if store:
        storage.append(M.copy())
        
    #Even if the "sanitize" condition is not active, run this once at the end to make all low floating point values 0
    #and force things that are close to integers to become integers. 
    sanitizeMatrix(M)
        
    return M, rank, storage

def M1MonteCarlo(standardData, sampleData, OCorrection, isotopologuesDict, fragmentationDictionary, 
                N = 100, GJ = False, debugMatrix = False, includeSubs = [], omitSubs = [], 
                disableProgress = False, theory = True, perturbTheoryOAmt = 0.002,
                experimentalOCorrectList = [], abundanceCorrect = True, 
                debugUnderconstrained = True, plotUnconstrained = False,
                storePerturbedSamples = False, storeOCorrect = False, explicitOCorrect = {}, 
                perturbOverrideList = []):
    """
    This is the main 'solver' routine which takes in sample and standard measurements and solves for the M+1 relative abundances. It does so via a Monte Carlo method to estimate error. There are many options, but most of these are for debugging and can be ignored by the general user. The majority of these options relate to the 'observed abundance correction' described in Csernica and Eiler 2023. 
     
    For each Monte Carlo iteration, it accomplishes the following:
        (1) Modify the O value correction within error
        (2) Modify the standard within error
        (3) Modify the sample within error 
        (4) Solve the Monte Carlo routine. 

    This process is repeated N times and the error bars are reported. The documentation of the arguments below gives an overview, but the user should read Csernica and Eiler, including the supplement, for more details. 

    The most important choice is whether to use a 'theroetical' or 'experimental' O value correction for missing peaks (see Csernica and Eiler 2023). Generally a theoretical one should be used, in which case set abundanceCorrect == True and perturbTheoryOAmt to 0.001 or 0.002. 

    The values for GJ, debugMatrix, includeSubs, omitSubs, debugMatrix, GJ, debugUnderconstrained, plotUnderconstrained, storePerturbedSamples, storeOCorrect, explicitOCorrect, and perturbOverrideList are generally for debugging and can be safely ignored. 

    Args:
        standardData: A dictionary; keys are mass selections ("M1", "M2") then fragment Keys ("full", "44"), then a dictionary with entries for 'subs', 'observed abundance', 'predicted abundance', and 'errors'. The entries of each key are lists with the same order. 
        sampleData: As standardData but without predicted abundance. 
        OCorrection: A dictionary. Keys are mass selections ("M1", "M2"), then fragment keys ("full", "44"), then values of the O value correction for each fragment. See Csernica and Eiler 2023. 
        isotopologuesDict: A dictionary; keys are mass selections, values are dataframes giving all of the isotopologues within that mass selection, including information about their fragmentation. The output of fas.isotopologueDataFrame.
        fragmentationDictionary: A dictionary, e.g. {'full': {'01': {'subgeometry': [1, 1, 1, 1, 1, 1], 'relCont': 1}},
                                                     '44': {'01': {'subgeometry': [1, 'x', 'x', 1, 1, 'x'], 'relCont': 1}}} which gives information about the fragments, their subgeometries and relative contributions.
        N: The number of Monte Carlo simulations to perform.
        GJ: A boolean. If False, the matrix is solved using np.linalg.lstqr. If True, instead uses a Gauss-JOrdan algorithm. This may be useful when debugging or when a system is underconstrained--however, note that if a M+1 system is underconstrained, it should be possible to constrain it by redefining the sites. Hence, GJ is of limited usefulness here. 
        debugMatrix: A boolean; True (if GJ is also True) will return every step of the Gauss-Jordan solution, helping the user debug issues.
        includeSubs: A list of isotopes to include in the solution. If left empty, all isotope will be included. Generally should be empty, unless you want to explore how removing certain constraints affects the solution. 
        omitSubs: As includeSubs, but gives a list of isotopes to omit. Generally should be empty. 
        disableProgress: A boolean; true disables the tqdm bars.
        theory: A boolean, determines whether to calculate fractionation factors based on the difference between predicted and observed values for the standard. Generally should be true. 
        perturbTheoryOAmt: An important parameter because it represents a user choice that has a significant effect on the output. For each run of the Monte Carlo, the O value correction factors may be perturbed by some amount, which is useful as the factors are only known approximately. A value of 0.001 is a good place to start. See especially section A.4 of the supplement to Csernica and Eiler 2023. 
        experimentalOCorrectList: A list, containing information about which peaks to use experimental, rather than theoretical, M+N Relative abundance corrections.
        abundanceCorrect: A boolean, determines whether to apply observed abundance correction.
        debugUnderconstrained: If True, attempts to find the null space of the Gauss-Jordan solution to output which sites are well constrained (do not vary with the null space).
        plotUnconstrained: If True, outputs a plot of the null space to visualize how sites covary with each other in the null space.
        storePerturbedSamples: An option to store the perturbed samples from each step of the MC for further investigation.
        perturbOverrideList: perturbSample will automatically perturb all sample acquisitions (M1, M2, M3, M4); in some cases, e.g. when doing an iterated correction for M1, we do not want to perturb all, only M1. This can be specified with this list. (E.g. ['M1'])
        explicitOCorrect: For each MNKey and each fragment, may define specific bounds on reasonable O correction values. Then, values for the observed abundance correction beyond these bounds are instead set to those bounds. 

    Returns:
        results: A dictionary, with GJ and NUMPY as keys. Each is keyed to a list of solutions from those respective algorithms.

    Examples:
        >>> import pandas as pd
        >>> from isotomics import calcIsotopologues as ci
        >>> from isotomics import fragmentAndSimulate as fas
        >>> from isotomics import readInput as ri
        >>> from isotomics import solveSystem as ss
        >>> molecular_df = pd.DataFrame(
        ...     {'IDS': ['C', 'O', 'S'], 'Number': [2, 1, 1], 'deltas': [-10, 0, 0]},
        ...     index=['C1', 'O1', 'S1'],
        ... )
        >>> fragmentationDictionary = {'frag': {'01': {'subgeometry': [1, 'x', 1], 'relCont': 1}}}
        >>> byAtom = ci.inputToAtomDict(molecular_df, disable=True, M1Only=False)
        >>> MNDict = ci.massSelections(byAtom, massThreshold=1)
        >>> MNDict = fas.trackMNFragments(MNDict, molecular_df, fragmentationDictionary)
        >>> predictedMeasurement, _ = fas.predictMNFragmentExpt(MNDict, molecular_df, fragmentationDictionary, abundanceThreshold=0)
        >>> processStandard = ri.readComputedData(predictedMeasurement, error=0.001, theory=predictedMeasurement)
        >>> processSample = ri.readComputedData(predictedMeasurement, error=0.001)
        >>> OCorrection = ss.OValueCorrectTheoretical(predictedMeasurement, processSample, massThreshold=1)
        >>> isotopologuesDict = fas.isotopologueDataFrame(MNDict, molecular_df)
        >>> M1Results = ss.M1MonteCarlo(
        ...     processStandard,
        ...     processSample,
        ...     OCorrection,
        ...     isotopologuesDict,
        ...     fragmentationDictionary,
        ...     N=5,
        ...     disableProgress=True,
        ... )
        >>> M1Results
        {'GJ': [],
            'NUMPY': [array([0.72862643, 0.01247204, 0.25890153]),
            array([0.72843636, 0.01244413, 0.25911951]),
            array([0.72872649, 0.01255215, 0.25872136]),
            array([0.72867259, 0.01256612, 0.25876129]),
            array([0.72836151, 0.01250019, 0.2591383 ])],
            'Extra Info': {'Perturbed Samples': [],
            'O Correct': [],
            'StoreExpFactors': []}}
    """
    MNKey = "M1"
    Isotopologues = isotopologuesDict[MNKey]

    results = {'GJ':[],"NUMPY":[], "Extra Info":{'Perturbed Samples':[],'O Correct':[],'StoreExpFactors':[]}}
    
    variableOCorrect = copy.deepcopy(OCorrection)
    for i in tqdm(range(N), disable = disableProgress):
        variableOCorrect = modifyOValueCorrection(OCorrection, variableOCorrect, MNKey, explicitOCorrect = explicitOCorrect, amount = perturbTheoryOAmt)
        std = perturbStandard(standardData, theory = theory)
        
        perturbedSample = perturbSample(sampleData, std, variableOCorrect, experimentalOCorrectList = experimentalOCorrectList,abundanceCorrect = abundanceCorrect,explicitOCorrect = explicitOCorrect, perturbOverrideList = perturbOverrideList)
        
        smp = perturbedSample['M1']
       
        if storePerturbedSamples:
            results["Extra Info"]['Perturbed Samples'].append(smp.to_dict())
        if storeOCorrect:
            results['Extra Info']['O Correct'].append(copy.deepcopy(variableOCorrect['M1'])) 
       
        comp, meas = constructMatrix(Isotopologues, smp, MNKey, fragmentationDictionary,
                                    includeSubs = includeSubs, omitSubs = omitSubs)
        
        sol = np.linalg.lstsq(comp, meas, rcond = -1)
        numpy = sol[0]
          
        results["NUMPY"].append(numpy)
        
        if GJ:
            #Optional GJ routine. Generally unnecessary here as the M+1 system will be constrained, unless there are unresolved peaks. (If there are no unresolved peaks and the system is not constrained, you can redefine the sites such that it is constrained.)
            AugMatrix = np.column_stack((comp, meas))
            solve = GJElim(AugMatrix, augMatrix = True, store = True, sanitize = True)
            results['GJ'].append(solve[0][:,-1][:13])
            
            if debugMatrix == True:
                return AugMatrix, solve
            
    if sol[2] < len(Isotopologues):
        if debugUnderconstrained:
            print("Solution is underconstrained")
            print("processM1MCResults will not work with GJ Solution")
            #If we DO have an underconstrained scenario, automatically report it. 
            AugMatrixUnderconstrained = np.column_stack((comp, meas))
            
            print("After solving null space:")
            nullSpaceCycles = findNullSpaceCycles(comp, Isotopologues, plot = plotUnconstrained)
            actuallyConstrained = findFullyConstrained(nullSpaceCycles)
            print("Actually Constrained:")
            for i in actuallyConstrained:
                print(i)

    return results

def PerturbUValue(UValuesSmp):
    """
    Perturbs the full molecule U Values based on their observed errors.

    Args:
        UValuesSmp: A dictionary where keys are isotopes and their values dictionaries giving their measured U Value and the error on that measurement.

    Returns:
        UPerturb: A dictionary where keys are isotopes and values are floats giving their perturbed U Values.

    Examples:
    """
    UPerturb = {}
    for i, v in UValuesSmp.items():
        UPerturb[i] = np.random.normal(v['Observed'],v['Error'])
    
    return UPerturb

def calcUMN(MNKey, dataFrame, UPerturb, UMNSub = []):
    """
    Calculates the U^M+N value for solution from the monte carlo routine.

    Args:
        MNKey: "M1", "M2", etc.
        dataFrame: A dataframe updated with the M+N relative abundances from the solution to some system.
        UPerturb: The perturbed full molecule U Values for this round of the monte carlo routine.
        UMNSub: Sets the specific substitutions that we will use molecular average U values from to calculate UMN. Otherwise it will use all molecular average U values for that UMN. Recommended to use--the procedure only works for substitions that are totally solved for. For example, if one 13C 13C isotopologue is not solved for precisely in fractional abundance space, we should not use 13C13C in the UMN routine. The best candidates tend to be abundant things--36S, 18O, 13C, 34S, and so forth. An eventual goal is to make a routine which automatically checks this.

    Returns:
        UMN: The U^M+N value for this round of the Monte Carlo routine.

    Examples:
    """
    UValueEstimates = []
    for isotope in set(dataFrame['Composition']):
        if isotope in UPerturb:
            if isotope in UMNSub or UMNSub == []:
                est = UPerturb[isotope] / dataFrame[dataFrame["Composition"] == isotope][MNKey + ' M+N Relative Abundance'].sum()
                UValueEstimates.append(est)

    UMN = np.array(UValueEstimates).mean()
    
    return UMN

def processM1MCResults(M1Results, UValuesSmp, isotopologuesDict,  molecularDataFrame, GJ = False, disableProgress = False, UMNSub = []):
    """
    Processes results of M1 Monte Carlo, converting the M+N Relative abundances into delta space and reordering to match the order of the original input dataframe.

    Args:
        M1Results: A dictionary containing the M1 results from the M1 Monte Carlo routine.
        UValuesSmp: A dictionary where keys isotopes and their values dictionaries giving their measured U Value and the error on that measurement.
        isotopologuesDict: A dictionary with keys giving mass selections ("M1", "M2", etc.) and values of dataFrames giving the isotopologues for that mass selection.
        molecularDataFrame: The site-specific dataFrame, i.e. the original input
        GJ: A boolean; if true, looks in the M1Results dictionary for GJ results, rather than NUMPY results.
        disableProgress: A boolean; if True, disables progress bar.
        UMNSub: A list of strings; the strings correspond to isotopes ('13C', '15N') used to calculate the U^M+1 value. Care needs to be taken--if certain isotopologues corresponding to these substitutions are not fully constrained, the routine will fail. This is one reason why it is important to check with a synthetic dataset first, to ensure the procedure works! 

    Returns:
        processedResults: A dictionary, containing lists of the results from every Monte Carlo solution for many variables of interest.

    Examples:
        >>> import pandas as pd
        >>> from isotomics import calcIsotopologues as ci
        >>> from isotomics import fragmentAndSimulate as fas
        >>> from isotomics import readInput as ri
        >>> from isotomics import solveSystem as ss
        >>> molecular_df = pd.DataFrame(
        ...     {'IDS': ['C', 'O', 'S'], 'Number': [2, 1, 1], 'deltas': [-10, 0, 0]},
        ...     index=['C1', 'O1', 'S1'],
        ... )
        >>> fragmentationDictionary = {'frag': {'01': {'subgeometry': [1, 'x', 1], 'relCont': 1}}}
        >>> byAtom = ci.inputToAtomDict(molecular_df, disable=True, M1Only=False)
        >>> bySub = ci.subDictionaryFromAtom(byAtom)
        >>> MNDict = ci.massSelections(byAtom, massThreshold=1)
        >>> MNDict = fas.trackMNFragments(MNDict, molecular_df, fragmentationDictionary)
        >>> predictedMeasurement, _ = fas.predictMNFragmentExpt(MNDict, molecular_df, fragmentationDictionary, abundanceThreshold=0)
        >>> predictedMeasurement['U Values'] = fas.calculateUValues(bySub, massThreshold=1)
        >>> processStandard = ri.readComputedData(predictedMeasurement, error=0.001, theory=predictedMeasurement)
        >>> processSample = ri.readComputedData(predictedMeasurement, error=0.001)
        >>> UValuesSmp = ri.readComputedUValues(predictedMeasurement, error=0.001, UMNSub='13C')
        >>> OCorrection = ss.OValueCorrectTheoretical(predictedMeasurement, processSample, massThreshold=1)
        >>> isotopologuesDict = fas.isotopologueDataFrame(MNDict, molecular_df)
        >>> M1Results = ss.M1MonteCarlo(
        ...     processStandard,
        ...     processSample,
        ...     OCorrection,
        ...     isotopologuesDict,
        ...     fragmentationDictionary,
        ...     N=1,
        ...     disableProgress=True,
        ... )
        >>> processedResults = ss.processM1MCResults(
        ...     M1Results,
        ...     UValuesSmp,
        ...     isotopologuesDict,
        ...     molecular_df,
        ...     UMNSub=['13C'],
        ...     disableProgress=True,
        ... )
        >>> processedResults
        {'CRF Deltas': [[-10.613715956522718, -3.21622108281161, -2.055183007075523]], 'Relative Deltas': [[-0.6199151075985121, -3.21622108281161, -2.055183007075523]], 'M1 M+N Relative Abundance': [[0.7286264337642733, 0.012472039963464437, 0.25890152627226215]], 'UM1': [[0.030362166792276064, 0.030362166792276064, 0.030362166792276064]], 'Calc U Values': [[0.022122677311212154, 0.0003786781576106399, 0.007860811323453266]]}
    """
    MNKey = "M1"
    processedResults = {'CRF Deltas':[],'Relative Deltas':[],MNKey + ' M+N Relative Abundance':[],'UM1':[],'Calc U Values':[]}
    string = "NUMPY"
    if GJ:
        string = "GJ"
    
    out = isotopologuesDict['M1'][['Number','Stochastic','Composition','Stochastic U','Precise Identity']].copy()
    
    for res in tqdm(M1Results[string], disable = disableProgress):
        out[MNKey + ' M+N Relative Abundance'] = res
        
        #Perturb U Values
        UPerturb = PerturbUValue(UValuesSmp)
        
        #Calculate UM1
        UM1 = calcUMN(MNKey, out, UPerturb, UMNSub = UMNSub)

        out['UM1'] = UM1
        out['Calc U Values'] = out[MNKey + ' M+N Relative Abundance'] * out['UM1']
        
        #The Isotopologues Dataframe has the substitutions in a different order than the site-specific dataframe. 
        #This section reassigns the solutions of the isotopologues dataframe to the right order for the 
        #site-specific dataframe
        M1 = [0] * len(out.index)
        UM1 = [0] * len(out.index)
        U = [0] * len(out.index)
        for i, v in out.iterrows():
            identity = v['Precise Identity'].split(' ')[1]
            index = list(molecularDataFrame.index).index(identity)

            M1[index] = v[MNKey + ' M+N Relative Abundance']
            UM1[index] = v['UM1']
            U[index] = v['Calc U Values']

        #calculate relevant information
        normM1 = U / molecularDataFrame['Number']
        #This gives deltas in absolute reference frame
        smpDeltasAbs = [op.ratioToDelta(x,y) for x, y in zip(molecularDataFrame['IDS'], normM1)]
        
        appxStd = molecularDataFrame['deltas']
        
        #This gives deltas relative to standard
        relSmpStdDeltas = [op.compareRelDelta(atomID, delta1, delta2) for atomID, delta1, delta2 in zip(molecularDataFrame['IDS'], appxStd, smpDeltasAbs)]

        processedResults['CRF Deltas'].append(smpDeltasAbs)
        processedResults['Relative Deltas'].append(relSmpStdDeltas)
        processedResults[MNKey + ' M+N Relative Abundance'].append(M1)
        processedResults['UM1'].append(UM1)
        processedResults['Calc U Values'].append(U)
        
    return processedResults

def updateSiteSpecificDfM1MC(processedResults, molecularDataFrame):
    """
    Adds the processed M1MC results to the original dataframe.

    Args:
        processedResults: A dictionary, containing lists of the results from every Monte Carlo solution for many variables of interest.
        molecularDataFrame: The site-specific dataFrame, i.e. the original input

    Returns:
        molecularDataFrame: The updated site-specific dataFrame.

    Examples:
        >>> import pandas as pd
        >>> from isotomics import calcIsotopologues as ci
        >>> from isotomics import fragmentAndSimulate as fas
        >>> from isotomics import readInput as ri
        >>> from isotomics import solveSystem as ss
        >>> molecular_df = pd.DataFrame(
        ...     {'IDS': ['C', 'O', 'S'], 'Number': [2, 1, 1], 'deltas': [-10, 0, 0]},
        ...     index=['C1', 'O1', 'S1'],
        ... )
        >>> fragmentationDictionary = {'frag': {'01': {'subgeometry': [1, 'x', 1], 'relCont': 1}}}
        >>> byAtom = ci.inputToAtomDict(molecular_df, disable=True, M1Only=False)
        >>> bySub = ci.subDictionaryFromAtom(byAtom)
        >>> MNDict = ci.massSelections(byAtom, massThreshold=1)
        >>> MNDict = fas.trackMNFragments(MNDict, molecular_df, fragmentationDictionary)
        >>> predictedMeasurement, _ = fas.predictMNFragmentExpt(MNDict, molecular_df, fragmentationDictionary, abundanceThreshold=0)
        >>> predictedMeasurement['U Values'] = fas.calculateUValues(bySub, massThreshold=1)
        >>> processStandard = ri.readComputedData(predictedMeasurement, error=0.001, theory=predictedMeasurement)
        >>> processSample = ri.readComputedData(predictedMeasurement, error=0.001)
        >>> UValuesSmp = ri.readComputedUValues(predictedMeasurement, error=0.001, UMNSub='13C')
        >>> OCorrection = ss.OValueCorrectTheoretical(predictedMeasurement, processSample, massThreshold=1)
        >>> isotopologuesDict = fas.isotopologueDataFrame(MNDict, molecular_df)
        >>> M1Results = ss.M1MonteCarlo(
        ...     processStandard,
        ...     processSample,
        ...     OCorrection,
        ...     isotopologuesDict,
        ...     fragmentationDictionary,
        ...     N=1,
        ...     disableProgress=True,
        ... )
        >>> processedResults = ss.processM1MCResults(
        ...     M1Results,
        ...     UValuesSmp,
        ...     isotopologuesDict,
        ...     molecular_df.copy(),
        ...     UMNSub=['13C'],
        ...     disableProgress=True,
        ... )
        >>> updated_df = ss.updateSiteSpecificDfM1MC(processedResults, molecular_df.copy())
        >>> print(updated_df.to_string())
           IDS  Number  deltas  CRF Deltas  CRF Deltas Error  Relative Deltas  Relative Deltas Error  M1 M+N Relative Abundance  M1 M+N Relative Abundance Error       UM1  UM1 Error  Calc U Values  Calc U Values Error
        C1   C       2     -10  -10.613716               0.0        -0.619915                    0.0                   0.728626                              0.0  0.030362        0.0       0.022123                  0.0
        O1   O       1       0   -3.216221               0.0        -3.216221                    0.0                   0.012472                              0.0  0.030362        0.0       0.000379                  0.0
        S1   S       1       0   -2.055183               0.0        -2.055183                    0.0                   0.258902                              0.0  0.030362        0.0       0.007861                  0.0
    """
    for key in processedResults.keys():
        molecularDataFrame[key] = np.array(processedResults[key]).T.mean(axis = 1)
        molecularDataFrame[key + ' Error'] = np.array(processedResults[key]).T.std(axis = 1)
        
    return molecularDataFrame

def MonteCarloMN(MNKey, Isotopologues, standardData, sampleData, OCorrection, 
                 fragmentationDictionary, N = 10, includeSubs = [], omitSubs = [], disableProgress = False, perturbTheoryOAmt = 0,abundanceCorrect = True):
    """
    The M+N experiment with N>2 will almost certainly be underconstrained, in contrast to the M+1 which will often be constrained. Additionally, we don't wish to report these results by updating the original dataframe. For these reasons, we define a separate set of functions for the M+N solution.

    Given an MNKey and sample/standard data, solves the MN system via GJ Elimination N times and stores the results.

    Also returns the composition matrix and a full GJ solution, which are useful later on.

    Args:
        MNKey: A string, "M2", "M3", etc.
        Isotopologues: A dataframe containing isotopologues introduced via that MN experiment together with information about how these isotopologues fragment.
        standardData: a dictionary, keys are MNKeys and values are dictionaries. Then the keys are fragments and
                values are dictionaries. Then the keys are "Subs", "Predicted Abundance", "Observed Abundance",
                "Error", giving information about that measurement.
        sampleData: As standardData, but no predicted abundances.
        OCorrection: A dictionary giving the O correction factors by mass selection and fragment.
        N: The number of Monte Carlo runs to perform.
        includeSubs: A list of isotopes, if we want to include only certain isotopes in the matrix. If it is nonempty, only isotopes in the list will be included in the matrix. Generally should be empty.
        omitSubs: A list of isotopes, if we wish to omit certain isotopes from the matrix. If it is nonempty, isotopes in the list will not be included in the matrix. Generally should be empty.
        disableProgress: A boolean; true disables the tqdm bars.
        perturbTheoryOAmt: A float. For each run of the Monte Carlo, the prtvrnt sbundance correction factors can be perturbed; this may be useful because the factors are only known approximately, so this well better estimate error. 0.001 and 0.002 have been useful values before, but it may depend on the system of interest.
        abundanceCorrect: A boolean, determines whether to apply observed abundance correction factors.

    Returns:
        res: A dictionary keying "GJ" to a list of gauss-jordan solutions to the system
        comp: The initial composition matrix
        solve: The solved GJ system, which is useful for finding codependencies
        meas: The initial measurement vector

    Examples:
    """
    res = {}
    res[MNKey] =  {'GJ':[]}
    
        
    variableOCorrect = copy.deepcopy(OCorrection)
    for i in tqdm(range(N), disable = disableProgress):
        variableOCorrect = modifyOValueCorrection(OCorrection, variableOCorrect, MNKey, amount = perturbTheoryOAmt)
        #Perturb sample and standard
        std = perturbStandard(standardData)
        smp = perturbSample(sampleData, std, variableOCorrect, abundanceCorrect = abundanceCorrect)[MNKey]

        
        comp, meas = constructMatrix(Isotopologues, smp, MNKey, fragmentationDictionary,
                                    includeSubs = includeSubs, omitSubs = omitSubs)

        AugMatrix = np.column_stack((comp, meas))

        solve = GJElim(AugMatrix, augMatrix = True)

        res[MNKey]['GJ'].append(solve[0][:,-1])
        
    return res, comp, solve, meas

def processMNMonteCarloResults(MNKey, results, UValuesSmp, dataFrame, molecularDataFrame, MNDictStd, UMNSub = [], disableProgress = False):
    """
    Given solutions from the GJ solver monte carlo routine and a dataFrame listing which isotopologues correspond to each solution, calculates M+N Relative abundances. Then perturbs and applies a UMN value and calculates deltas and clumped deltas. Stores these values in a dictionary for statistics to be run on them.

    Args:
        MNKey: A string, "M2"
        results: A dictionary containing the results of the Monte Carlo
        UValuesSmp: A dictionary where keys are isotopes and values are dictionaries; the interior keys are
            "Observed" and "Error".
        dataFrame: A dataFrame with the isotopologues corresponding to each of the GJ solutions.
        molecularDataFrame: A dataframe with basic information about the sites of the molecule.
        MNDictStd: A dictionary, where keys are MN Keys and values are dataframes containing the isotopologues and their concentrations for the calculated standard.
        UMNSub: A list of substitutions to use to calculate the UMN values.
        disableProgress: Set True to disable tqdm progress bar

    Returns:
        processedResults: A dictionary containing values for several important measures from each Monte Carlo run.

    Examples:
    """
    processedResults = {MNKey + ' M+N Relative Abundance':[],'U' + MNKey:[],'U Values':[], 'Deltas':[], 'Clumped Deltas Stochastic': [], 'Clumped Deltas Relative': []}
    rank = len(dataFrame.index)
    
    for sol in tqdm(results[MNKey]['GJ'], disable = disableProgress):
        dataFrame[MNKey + ' M+N Relative Abundance'] = sol[:rank]

        UPerturb = PerturbUValue(UValuesSmp)

        UMN = calcUMN(MNKey, dataFrame, UPerturb, UMNSub = UMNSub)

        dataFrame['U' + MNKey] = UMN
        dataFrame['U Values'] = dataFrame[MNKey + ' M+N Relative Abundance'] * dataFrame['U' + MNKey]

        dataFrame = computeMNUValues(dataFrame, MNKey, molecularDataFrame, applyUMN = False)
        dataFrame = updateRelClumpedDeltas(dataFrame, MNKey, MNDictStd)

        for key in processedResults:
            #need to copy because we are taking the results from a dataFrame
            processedResults[key].append(dataFrame[key].values.copy())
            
    return processedResults

def computeMNUValues(MNSolution, MNKey, molecularDataFrame, applyUMN = True, clumpU = False):
    """
    Takes a dataframe containing the M+N Solution in M+N Relative abundance space, transfers these to U value space, and then calculates clumped and site-specific delta values from these U Values.

    Args:
        MNSolution: A dataframe containing the M+N results in M+N Relative abundance space as well as the U^M+N value.
        MNKey: "M2", "M3", etc.
        molecularDataFrame: The original dataframe containing information about sites of the molecule.
        applyUMN: A boolean, determines whether to calculate U Values using U^M+N or not.
        clumpU: A boolean; set to True if applying some numerical routine to deal with large clumps.

    Returns:
        MNSolution: The dataframe updated with clumped and site-specific deltas.

    Examples:
    """
    if applyUMN:
        MNSolution['U Values'] = MNSolution[MNKey + ' M+N Relative Abundance'] * MNSolution["U" + MNKey]
        
    if clumpU:
        string = "Clump Adjusted U Values"
        deltaString = 'Clump Adjusted Deltas'
    else:
        string = "U Values"
        deltaString = "Deltas"
    
    #calculate site specific deltas
    deltas = []
    for i, v in MNSolution.iterrows():
        # | gives multiple substitutions, & gives multiple isotopologues
        #if we have a single isotopic substitution (but potentially multiple unresolved isotopologues with that substitution)
        if '|' not in i:
            n = 0
            contributingAtoms = i.split(' & ')
            for atom in contributingAtoms:
                ID = atom.split(' ')[1]
                #molecularDataFrame.index gives atom IDs, .index() function returns associated index
                indexOfId = list(molecularDataFrame.index).index(ID)
                n += molecularDataFrame['Number'][indexOfId]


            siteSpecificR = v[string] / n
            #could still have an error, i.e. "13C C-1 & 17O O-4", so check
            try:
                delta = op.ratioToDelta(v['Composition'],siteSpecificR)
                deltas.append(delta)
            except:
                deltas.append('N/A')
            
        else:
            deltas.append('N/A')
            
    #calculate clumped deltas
    clumpedDeltas = [1000*(x/y-1) for x, y in zip(MNSolution[string].values, MNSolution['Stochastic U'].values)]
    clumpedCulled = []
    
    for i in range(len(clumpedDeltas)):
        if deltas[i] == 'N/A':
            if np.abs(clumpedDeltas[i]) < 10**(-8):
                clumpedCulled.append(0)
            else:
                clumpedCulled.append(clumpedDeltas[i])
        else:
            clumpedCulled.append('N/A')

    MNSolution[deltaString] = deltas
    MNSolution['Clumped Deltas Stochastic'] = clumpedCulled
    
    return MNSolution

def updateRelClumpedDeltas(MNSolution,MNKey,MNDictStd):
    """
    Compute relative clumped deltas, rather than in the stochastic reference frame, for a M+N solution.

    Args:
        MNSolution: A dataframe containing the M+N results in U Value space.
        MNKey: "M2", "M3", etc.
        MNDictStd: A dictionary, where keys are MN Keys and values are dataframes containing the isotopologues and their concentrations for the calculated standard.

    Returns:
        MNSolution: The same dataframe with relative clumped deltas added.

    Examples:
    """
    appxUSmp = [checkNumber(UVal, Number) for UVal, Number in zip(MNSolution['U Values'].values, MNSolution['Number'].values)]
    
    appxUStd = [computeStdUVal(condensed, MNKey, MNDictStd) for condensed, Number in zip(MNSolution['Condensed'].values, MNSolution['Number'].values)]
    
    relMNComparison = 1000 * (np.array(appxUSmp) / appxUStd - 1)
    
    MNSolution['Clumped Deltas Relative'] = relMNComparison

    return MNSolution

def computeStdUVal(condensed, MNKey, MNDictStd):
    """
    Helper function for updateRelClumpedDeltas; computes the UValue of a constrained isotopologue in the standard. Only returns U Values for isotopologues which are fully constrained; otherwise returns np.nan

    Args:
        condensed: The byAtom string for an isotopologue
        MNKey: "M2", "M3", etc.
        MNDictStd: A dictionary, where keys are MN Keys and values are dataframes containing the isotopologues and their concentrations for the calculated standard.

    Returns:
        The calculated U value for the standard, or np.nan

    Examples:
    """
    if '&' not in condensed:
        UVal = MNDictStd[MNKey][condensed]['Conc'] / MNDictStd['M0']['000000000000000000000']['Conc']
        return UVal
    else:
        return np.nan
    
def checkNumber(UVal, Number):
    """
    Helper function for updateRelClumpedDeltas; checks if an isotopologue is fully constrained; if not, does not attempt to calculate relative clumped deltas.

    Args:
        UVal: A float, the solved U Value for some set of isotopologues
        Number: The number of isotopologues included in the U Value

    Returns:
        UVal if Number == 1, otherwise np.nan.

    Examples:
    """
    if Number == 1:
        return UVal
    else:
        return np.nan
    
def stringMeansAndStds(key, processedResults):
    """
    Used to calculate means and standard deviations for the M+N routine, after running Monte Carlo and storing the results in processed results. Both the "Deltas" and "Clumped Deltas" have "N/A" in addition to numerical results, so need some extra processing in order to deal with these strings. This function does that, changing those entries to np.nan and then calculating means and standard deviations.

    Args:
        key: "Deltas" or "Clumped Deltas", for the thing one wants to run statistics on.
        processedResults: A dictionary with the results of each monte carlo run.

    Returns:
        A tuple, giving the means and standard deviations for the key of interest.

    Examples:
    """
    changeDeltas = []
    for y in processedResults[key]:
        z = [replaceNA(x) for x in y]
        changeDeltas.append(np.array(z,dtype = float))

    #get a runtime warning if we take mean of all NaN. This avoids the warning. 
    means = []
    for i in np.array(changeDeltas,dtype = float).T:
        means.append(cleanNanMean(i))

    #get a runtime warning if we take mean of all NaN. This avoids the warning. 
    std = []
    for i in np.array(changeDeltas,dtype = float).T:
        std.append(cleanNanStd(i))

    return np.array(means), np.array(std)

def replaceNA(x):
    if x == 'N/A':
        return np.nan
    return x

def cleanNanMean(x):
    if np.all(np.isnan(x)):
        return np.nan
    return np.mean(x)

def cleanNanStd(x):
    if np.all(np.isnan(x)):
        return np.nan
    return np.std(x)

def updateMNMonteCarloResults(dataFrame, processedResults):
    """
    Updates the dataFrame containing the GJ Solution with the results of the monte carlo routine. For each key in processed results, fills in the mean and standard deviation of the monte carlo runs.

    Args:
        dataFrame: A dataframe containing the results of a single monte carlo run.
        processedResults: A dictionary containing the results of all monte carlo runs.

    Returns:
        dataFrame: The dataframe updated with results of the monte carlo runs.

    Examples:
    """
    #The dataframe likely contains some extraneous information from the last MC run, so remove all of this.
    dataFrame = dataFrame[['Stochastic U','Composition','Number','Condensed']].copy()

    for key in processedResults.keys():
        #Deltas and Clumped Deltas include "N/A", so need special attention
        if key != 'Deltas' and key != 'Clumped Deltas Stochastic':
            dataFrame[key] = np.array(processedResults[key]).T.mean(axis = 1)
            dataFrame[key + ' Error'] = np.array(processedResults[key]).T.std(axis = 1)

        if key == 'Deltas' or key == 'Clumped Deltas Stochastic':
            mean, standard = stringMeansAndStds(key, processedResults)
            dataFrame[key] = mean
            dataFrame[key + ' Error'] = standard
            
    return dataFrame

def checkSolutionIsotopologues(solve, Isotopologues, massKey, numerical = True):
    """
    Given a solution to an augmented matrix which associates isotopologues with their M+N Relative abundance, recovers the identity of each isotopologue from its column in the augmented matrix. Computes the stochastic abundance of each constraint, as well as their compositions and number.

    You can think of this as a lookup function, which goes through the rows of the solved GJ solution and determines which columns (isotopologues) contribute to each.

    Args:
        solve: The output of GJElim
        Isotopologues: A dataFrame containing the MN population of Isotopologues
        massKey: 'M1', 'M2', etc.
        numerical: True if we want to pull out U Values or M+N Relative Abundance. When running the MC routine, we don't, we only need to know which row corresponds to which isotopologues.

    Returns:
        A Pandas dataFrame containing information about the isotopologues corresponding to the GJ Solution.

    Examples:
    """
    #Take everything but the final column, which is just the answer
    solution = solve[0][:,:-1]
    rank = solve[1]
    
    uniqueAnswers = []
    stochasticValues = []
    composition = []
    number = []
    isotopologueStringList = []
    
    MatrixRows = []

    for i in range(len(solution)):
        MatrixRows.append(solution[i])
        stoch = 0
        c = None
        isotopologueString = None

        if i >= rank:
            break

        rowIsotopologues = []
        n = 0

        for j in range(len(solution[i])):
            if solution[i][j] > 0:
                string = ""
            elif solution[i][j] < 0:
                string = "MINUS "           
            
            if solution[i][j] != 0:
                sol = solution[i][j]
                n += 1

                if sol != 1:
                    rowIsotopologues.append(string + str(sol) + " " + Isotopologues['Precise Identity'][j])
                else:
                    rowIsotopologues.append(Isotopologues['Precise Identity'][j])

                stoch += sol*Isotopologues['Stochastic U'][j]

                if c == None:
                    c = Isotopologues['Composition'][j]
                elif c != Isotopologues['Composition'][j]:
                    c = c + " & " + Isotopologues['Composition'][j]
                    
                if isotopologueString == None:
                    isotopologueString = Isotopologues.index[j]
                elif isotopologueString != Isotopologues.index[j]:
                    isotopologueString = isotopologueString + " & " + Isotopologues.index[j]

        uniqueAnswers.append(rowIsotopologues)
        stochasticValues.append(stoch)
        composition.append(c)
        number.append(n)
        isotopologueStringList.append(isotopologueString)

    if numerical: 
        #take the measured values
        values = solve[0][:rank,-1]

    condensed = [' & '.join(x) for x in uniqueAnswers]

    #output as dataFrame
    output = {}
    
    if numerical:
        output[massKey +' M+N Relative Abundance'] = values
        output['Matrix Row'] = MatrixRows

    output['Stochastic U'] = stochasticValues
    output['Composition'] = composition
    output['Number'] = number
    output['Condensed'] = isotopologueStringList
    
    dfOutput = pd.DataFrame.from_dict(output)
    dfOutput.index = condensed
    
    return dfOutput

def findNullSpaceCycles(comp, Isotopologues, plot = False):
    """
    For underconstrained systems, it is hard to know which variables are correlated with one another. This function shows us. It does so by finding the null space, performing GJ elimination on it, then searching each row of the null space to see where nonzero entries are--a nonzero entry means two isotopologues are codependent.

    It then picks out the strings corresponding to codependent indices.

    It then constructs a dictionary where keys are strings corresponding to isotopologues and values are lists of strings. If an isotopologue can covary with another, then the list contains that isotopologue.

    It optionally plots a visualization of these codependencies.

    Args:
        comp: The composition matrix for GJ elimination
        Isotopologues: The dataFrame giving all isotopologues of an M+N measurement.
        plot: If true, plots a visualization of codependencies.

    Returns:
        nullSpaceCycles: A dictionary; keys are isotopologues, and their values are sets of the isotopologues which they covary with.

    Examples:
    """
    ns = null_space(comp)
    sanNS = sanitizeMatrix(ns, full = True)
    simpleBasis = GJElim(sanNS.T)[0]

    cycles = []
    for i, row in enumerate(simpleBasis):
        coDep = []
        for j, col in enumerate(row):
            if col != 0: 
                coDep.append(j)
        cycles.append(coDep)

    precise = Isotopologues['Precise Identity'].values
    strCycles = []
    for c in cycles:
        strC = [precise[x] for x in c]
        strCycles.append(strC)

    nrows, ncols = simpleBasis.shape
    nullSpaceCycles = {}
    for i in range(ncols):
        nullSpaceCycles[precise[i]] = set()

    for cyc in strCycles:
        for i in cyc:
            for j in cyc:
                nullSpaceCycles[str(i)].add(str(j))
                
    if plot:
        labels = []
        lengths = []
        for i, v in nullSpaceCycles.items():
            labels.append(i)
            lengths.append(len(v))
            
        edges = []
        for i, v in nullSpaceCycles.items():
            for connection in v:
                if(connection, i) not in edges:
                    edges.append((i, connection))
                    
        G = nx.Graph()
        for i, v in nullSpaceCycles.items():
            if len(v) != 1:
                G.add_node(i)
        for i, v in nullSpaceCycles.items():
            if len(v) != 1:
                for connection in v:
                    if (connection, i) not in G.edges():
                        G.add_edge(connection, i)
        
        pos = nx.drawing.layout.spring_layout(G,k=3)
        plt.figure(1,figsize=(12,12)) 
        nx.draw(G, pos=pos, cmap = plt.get_cmap('jet'),with_labels=True)
        plt.show()

    return nullSpaceCycles

def findFullyConstrained(nullSpaceCycles):
    """
    Given the null space cycles, as a dictionary, finds which isotopologues do not vary with any others. I.e. these isotopologues are solved for precisely. Outputs these as a list.

    Args:
        nullSpaceCycles: A dictionary; keys are isotopologues, and their values are sets of the isotopologues which they covary with.

    Returns:
        actuallyConstrained: A list of the isotopologues which do not covary with any other isotopologues.

    Examples:
    """
    actuallyConstrained = []
    for i, v in nullSpaceCycles.items():
        if len(v) == 0:
            actuallyConstrained.append(i)

    return actuallyConstrained
