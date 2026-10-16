import pandas as pd
from . import fragmentAndSimulate as fas
from . import readInput as ri
from . import solveSystem as ss
from . import calcIsotopologues as ci
import matplotlib.pyplot as plt
from . import basicDeltaOperations as op

import json
import copy
import warnings

#Name of the optional row in the input csv giving the relative contribution of each fragment subgeometry
RELATIVE_CONTRIBUTION_ROW = 'Relative Contribution'

def parseFragmentColumn(col):
    '''
    Splits a fragment column name into its fragment and subgeometry. A fragment formed by a single pathway is given as 'Fragment 44'; a fragment formed by several pathways has one column per pathway, 'Fragment 82_01', 'Fragment 82_02', etc.

    Inputs:
        col: A string, format like 'Fragment 44' or 'Fragment 82_01'.

    Outputs:
        fragKey, geometryNumber: e.g. ('44', '01') or ('82', '02').
    '''
    newName = col.split(' ')[1]
    if '_' in newName:
        fragKey, geometryNumber = newName.split('_')
        return fragKey, geometryNumber
    return newName, '01'

def process_fragment_column(col, fragSubset, molecularDf, fragmentationDictionary, relContributions):
    '''
    Read and process a single fragment column. Helper function for moleculeFromCsv.

    Inputs:
        col: A string, format like 'Fragment 44' or 'Fragment 82_01'.
        fragSubset: A list of fragment names, like ['44','full']
        molecularDf: The molecular DataFrame
        fragmentationDictionary: A dictionary specifying the fragments and their geometries
        relContributions: A dictionary giving the relative contribution of each fragment column, e.g. {'Fragment 82_01': 0.4}. Columns not included default to 1.

    Outputs:
        fragmentationDictionary, molecularDf

        As above, but either filled with this subgeometry (fragmentationDictionary, if its fragment is in fragSubset) or with the column removed (molecularDf, if it is not).
    '''
    fragKey, geometryNumber = parseFragmentColumn(col)

    #If the fragment is not desired, drop it
    if fragKey not in fragSubset:
        molecularDf.drop(columns=col, inplace=True)
        return fragmentationDictionary, molecularDf

    subgeometries = fragmentationDictionary.setdefault(fragKey, {})
    if geometryNumber in subgeometries:
        raise ValueError(f"Fragment {fragKey} subgeometry {geometryNumber} is defined more than once (column '{col}').")
    subgeometries[geometryNumber] = {'subgeometry':list(molecularDf[col]), 'relCont':relContributions.get(col, 1)}

    return fragmentationDictionary, molecularDf

def otherFragmentReps(fragmentationDictionary, molecularDf):
    """
    Computes alternative representations of the fragments for easier processing.

    Inputs:
        fragmentationDictionary: A dictionary specifying the fragments and their geometries

    Outputs:
        expandedFrags: An ATOM depiction of each fragment, where an ATOM depiction has one entry for each atom (rather than for each site). See fragmentAndSimulate for details.
        fragSubgeometryKeys: A list of strings, e.g. 44_01, 44_02, corresponding to each subgeometry of each fragment. A fragment will have multiple subgeometries if there are multiple fragmentation pathways to form it.
    """
    condensedFrags = []
    fragSubgeometryKeys = []
    for fragKey, subFragDict in fragmentationDictionary.items():
        for subFragNum, subFragInfo in subFragDict.items():
            #for each subgeometry, get which sites it samples and its key. Add to list.
            condensedFrags.append(subFragInfo['subgeometry'])
            fragSubgeometryKeys.append(fragKey + '_' + subFragNum)

    #Expand to ATOM depictions of the fragments.
    expandedFrags = [fas.expandFrag(x, molecularDf['Number']) for x in condensedFrags]

    return expandedFrags, fragSubgeometryKeys

def moleculeFromCsv(path, deltas = False, fragSubset = None):
    '''
    Reads in a .csv file to get basic information about that molecule.

    The .csv has one row per site, with columns 'Site Names', 'Element', 'Number Atoms', and one 'Fragment' column per fragment giving whether each site is retained (1) or lost (0). If a fragment forms by several pathways, give one column per pathway ('Fragment 82_01', 'Fragment 82_02') and add a final row named 'Relative Contribution' giving the share of the fragment formed by each pathway; these must sum to 1 for each fragment. Fragments with a single column default to 1.

    Inputs:
        path: a string locating the .csv, e.g. 'Example Input.csv'
        deltas: A list of delta values, one for each site, in the order of the rows of the .csv.
        fragSubset: A list of fragments to include. Defaults to None, in which case it is filled programatically to contain all fragments.

    Outputs:
        initializedMolecule: A dictionary, containing:

            molecularDataFrame: A dataframe containing basic information about the molecule. 
            expandedFrags: An ATOM depiction of each fragment, where an ATOM depiction has one entry for each atom (rather than for each site). See fragmentAndSimulate for details.
            fragSubgeometryKeys: A list of strings, e.g. 44_01, 44_02, corresponding to each subgeometry of each fragment. A fragment will have multiple subgeometries if there are multiple fragmentation pathways to form it.
            fragmentationDictionary: A dictionary like the allFragments variable, but only including the subset of fragments selected by fragSubset.
    '''
    #Read csv
    molecularDf = pd.read_csv(path, index_col='Site Names')
    molecularDf.index.name = None

    #Columns must be renamed from the input csv to the format the code expects.
    molecularDf.rename(columns={'Element': 'IDS', 'Number Atoms': 'Number'}, inplace=True)
    fragCols = [col for col in molecularDf.columns if col.startswith('Fragment ')]

    #Pull out the optional relative contribution row, then restore integer columns (the extra row makes them read as floats)
    relContributions = {}
    if RELATIVE_CONTRIBUTION_ROW in molecularDf.index:
        relRow = molecularDf.loc[RELATIVE_CONTRIBUTION_ROW]
        relContributions = {col: float(relRow[col]) for col in fragCols if pd.notna(relRow[col])}
        molecularDf = molecularDf.drop(index=RELATIVE_CONTRIBUTION_ROW)
        molecularDf['Number'] = molecularDf['Number'].astype(int)
        for col in fragCols:
            molecularDf[col] = molecularDf[col].astype(int)

    #Fill in programatically in the default case; 'Fragment 82_01' and 'Fragment 82_02' are both fragment '82'
    if fragSubset is None:
        fragSubset = list(dict.fromkeys(parseFragmentColumn(col)[0] for col in fragCols))

    #Fill a dictionary with the fragment information and which sites each fragment samples
    fragmentationDictionary = {}
    for col in fragCols:
        molecularDf[col] = molecularDf[col].replace(0, 'x')
        fragmentationDictionary, molecularDf = process_fragment_column(col, fragSubset, molecularDf, fragmentationDictionary, relContributions)

    #The pathways forming each fragment must account for all of it
    for fragKey, subgeometries in fragmentationDictionary.items():
        total = sum(info['relCont'] for info in subgeometries.values())
        if abs(total - 1) > 1e-6:
            raise ValueError(f"The relative contributions of fragment {fragKey}'s subgeometries sum to {total:g}, but must sum to 1. "
                             f"Add a '{RELATIVE_CONTRIBUTION_ROW}' row to the input csv giving each 'Fragment {fragKey}_NN' column's share.")

    #calculate some other objects relating to fragments that are used in computations
    expandedFrags, fragSubgeometryKeys = otherFragmentReps(fragmentationDictionary, molecularDf)

    molecularDf['deltas'] = deltas
    
    #Construct output
    initializedMolecule = {'molecularDataFrame':molecularDf,
                           'expandedFrags':expandedFrags,
                           'fragSubgeometryKeys':fragSubgeometryKeys,
                           'fragmentationDictionary':fragmentationDictionary}
    
    return initializedMolecule

def extractExperimentalErrors(errorPath, UValueError = 0.001, MNError = 0.001):
    '''
    Read in error bars from a user provided csv. If no CSV is provided, instead return a single float giving error for the molecular average measurement (U Value) and for the M+N Experiment. All errors are relative errors given as fractions, both in the csv and as arguments: a value of 0.001 corresponds to a 1 per mil error. 

    Inputs:
        errorPath: Path to the CSV. 
        UValueError: A float giving the error on the U Value. Only used if no errorPath provided.
        MNError: A float giving the error on the M+N measurements. Only used if no errorPath provided.

    Outputs:
        UValueErrors, MNErrors: Dictionaries. UValueErrors looks like: {'13C':0.0001}, while MNErrors looks like: {'M1':{'44':{'15N':0.002,'13C':0.001,'Unsub':0.001}}}

        Alternatively:
        UValueError, MNError: The same as the inputs. 
    '''
    if errorPath:
        errorDf = pd.read_csv(errorPath, header = None)
        errorDf[0] = op.standardizeSubstitutionNames(errorDf[0], 'the error file')

        targetHeaders = ['Molecular Average']
        fragmentPrefix = 'Fragment'

        errorDict = {}

        for rowIdx, row in errorDf.iterrows():
            headerCell = row[0]
            if any(header in headerCell for header in targetHeaders) or headerCell.startswith(fragmentPrefix):
                #Clean up the header to not inlude 'Fragment' prefix
                keyName = headerCell[len(fragmentPrefix):].strip() if headerCell.startswith(fragmentPrefix) else headerCell

                #Initialize Dictionary
                errorDict[keyName] = {}

                # Loop through subsequent rows to gather data until another target header or the end of the DataFrame
                for j in range(rowIdx+1, len(errorDf)):
                    nextRow = errorDf.iloc[j]
                    nextHeaderCell = nextRow[0]
                    # Check if the next row is another header
                    if any(header in nextHeaderCell for header in targetHeaders) or nextHeaderCell.startswith(fragmentPrefix):
                        break  # Found another header, stop processing this section
                    else:
                        # This row is data to be included for the current header
                        errorDict[keyName][op.standardizeSubstitutionName(nextHeaderCell)] = float(nextRow[1])

        #Input file has '13C/Unsub' for U Values. Later code expects '13C'. 
        UValueErrors = {subKey.split('/')[0]: subValue for subKey, subValue in errorDict.get('Molecular Average', {}).items()}
        errorDict.pop('Molecular Average', None)

        #Fill in MNErrors in expected format
        MNErrors = {'M1':errorDict}

        return UValueErrors, MNErrors

    else:
        return UValueError, MNError
    

def addClumps(byAtom, molecularDataFrame, clumpD):
    '''
    A function which adjusts the abundances of target isotopologues to include clumps. 

    Inputs:
        byAtom: A dictionary where keys are isotoplogues and values are dictionaries containing details about those isotopologues. 
        molecularDataFrame: A dataframe containing basic information about the molecule. 
        clumpD: Specifies information about clumps to add; otherwise the isotome follows the stochastic assumption. Currently works only for mass 1 substitutions (e.g. 1717, 1317, etc.) See ci.introduceClump for details.

    Outputs:
        byAtom: The same dictionary, with concentrations modified to include clumps. 
    '''
    stochD = copy.deepcopy(byAtom)
    
    for clumpNumber, clumpInfo in clumpD.items():
        byAtom = ci.introduceClump(byAtom, clumpInfo['Sites'], clumpInfo['Amount'], molecularDataFrame)
        
    for clumpNumber, clumpInfo in clumpD.items():
        ci.checkClumpDelta(clumpInfo['Sites'], molecularDataFrame, byAtom, stochD)

    return byAtom

def simulateMeasurement(initializedMolecule, abundanceThreshold = 0, UValueList = [],
                        massThreshold = 1, clumpD = {}, outputPath = None, disableProgress = True, calcFF = False, fractionationFactors = {}, omitMeasurements = {}, ffstd = 0.05, unresolvedDict = {}, outputFull = False):
    '''
    Simulates M+N measurements of an alanine molecule with input deltas specified by the input dataframe molecularDataFrame. 

    Inputs:
        molecularDataFrame: A dataframe containing basic information about the molecule. 
        expandedFrags: An ATOM depiction of each fragment, where an ATOM depiction has one entry for each atom (rather than for each site). See fragmentAndSimulate for details.
        fragSubgeometryKeys: A list of strings, e.g. 44_01, 44_02, corresponding to each subgeometry of each fragment. A fragment will have multiple subgeometries if there are multiple fragmentation pathways to form it.
        fragmentationDictionary: A dictionary like the allFragments variable, but only including the subset of fragments selected by fragSubset.
        abundanceThreshold: A float; Does not include measurements below this M+N relative abundance, i.e. assuming they will not be  measured due to low abundance. 
        UValueList: A list giving specific substitutions to calculate molecular average U values for ('13C', '15N', etc.)
        massThreshold: An integer; will calculate M+N relative abundances for N <= massThreshold
        clumpD: Specifies information about clumps to add; otherwise the isotome follows the stochastic assumption. Currently works only for mass 1 substitutions (e.g. 1717, 1317, etc.) See ci.introduceClump for details.
        outputPath: A string, e.g. 'output', or None. If it is a string, outputs the simulated spectrum as a json. 
        disableProgress: Disables tqdm progress bars when True.
        calcFF: When True, computes a new set of fractionation factors for this measurement.
        fractionationFactors: A dictionary, specifying a fractionation factor to apply to each ion beam. This is used to apply fractionation factors calculated previously to this predicted measurement (e.g. for a sample/standard comparison with the same experimental fractionation)
        omitMeasurements: omitMeasurements: A dictionary, {}, specifying measurements which I will not observed. For example, omitMeasurements = {'M1':{'61':'2H'}} would mean I do not observe the 2H ion beam of the 61 fragment of the M+1 experiment, regardless of its abundance. 
        ffstd: A float; if new fractionation factors are calculated, they are pulled from a normal distribution centered around 1, with this standard deviation.
        unresolvedDict: A dictionary, specifying which unresolved ion beams add to each other.
        outputFull: A boolean. Typically False, in which case beams that are not observed are culled from the dictionary. If True, includes this information; this should only be used for debugging, and will likely break the solver routine. 
        
    Outputs:
        predictedMeasurement: A dictionary giving information from the M+N measurements. 
        MN: A dictionary where keys are mass selections ("M1", "M2") and values are dictionaries giving information about the isotopologues of each mass selection.
        fractionationFactors: The calculated fractionation factors for this measurement (empty unless calcFF == True)
    '''
    #Accept the legacy deuterium label 'D' in these arguments
    UValueList = op.standardizeSubstitutionArgument(UValueList, "the UValueList argument")
    fractionationFactors = op.standardizeSubstitutionArgument(fractionationFactors, "the fractionationFactors argument")
    omitMeasurements = op.standardizeSubstitutionArgument(omitMeasurements, "the omitMeasurements argument")
    unresolvedDict = op.standardizeSubstitutionArgument(unresolvedDict, "the unresolvedDict argument")
    #Unpack initialized molecule dictionary
    molecularDataFrame = initializedMolecule['molecularDataFrame']
    fragmentationDictionary = initializedMolecule['fragmentationDictionary']

    byAtom = ci.inputToAtomDict(molecularDataFrame, disable = disableProgress)
    if clumpD != {}:
        byAtom = addClumps(byAtom, molecularDataFrame, clumpD)

    #bySub is an representation of data, a dictionary where keys are substitutions (e.g., '13C'), and values are their abundances. 
    bySub = ci.subDictionaryFromAtom(byAtom)
    uValues = fas.calculateUValues(bySub, massThreshold = massThreshold,subList = UValueList)

    MN = ci.massSelections(byAtom, massThreshold = massThreshold)
    MN = fas.trackMNFragments(MN, molecularDataFrame, fragmentationDictionary, unresolvedDict = unresolvedDict)
        
    predictedMeasurement, fractionationFactors = fas.predictMNFragmentExpt(MN, molecularDataFrame,
                                                  fragmentationDictionary, calcFF = calcFF, ffstd = ffstd,
                                                  abundanceThreshold = abundanceThreshold, fractionationFactors = fractionationFactors, omitMeasurements = omitMeasurements, unresolvedDict = unresolvedDict, outputFull = outputFull)
    predictedMeasurement['U Values'] = uValues
    
    if outputPath != None:
        output = json.dumps(predictedMeasurement)

        f = open(outputPath + ".json","w")
        f.write(output)
        f.close()
        
    return predictedMeasurement, MN, fractionationFactors



def plotOutput(simulationOutput, relativeDeltasActual = None, ylim = False):
    '''
    Simple function to plot the output of a simulation.

    Inputs:
        simulationOutput: A dataframe containing the results of a M+1 simulation. returned by simulateSmpStd. 
        relativeDeltasActual: A list, giving the actual delta values for the sample standard comparison. The ordering of the list should be the same as the indices of simulationOutput. If "None", do not plot these. 

    Outputs:
        fig: The matplotlib figure. 
    '''
    fig, ax = plt.subplots(figsize = (8,4))

    computedDeltas, computedDeltasErr = simulationOutput['Relative Deltas'], simulationOutput['Relative Deltas Error']

    if relativeDeltasActual:
        plt.scatter(range(len(relativeDeltasActual)),relativeDeltasActual, label = 'Actual', marker = 's', color = 'r')
    plt.errorbar(range(len(computedDeltas)),computedDeltas,computedDeltasErr, fmt = 'o',c='k',capsize = 5,
                label = 'Computed')
    plt.legend()
    ax.set_xticks(range(len(computedDeltas)))
    ax.set_xticklabels(simulationOutput.index, rotation = 45)
    ax.set_ylabel(r"$\delta_{STD}$")
    if ylim:
        ax.set_ylim(*ylim)

    return fig

def simulateSmpStd(path, deltasStd, deltasSmp, deltasStdAppx, abundanceThreshold = 0, UValueList = [], massThreshold = 1,  disableProgress = True, calcFF = False, omitMeasurements = {}, ffstd = 0.05, plot = True, MonteCarloN = 100, perturbTheoryOAmt = 0, errorPath = False, MNError = 0, UValueError = 0, resultsFileName = 'output.csv', outputPrecision = 3, UMNSub = '13C'):
    '''
    Parent function which constructs and runs a full sample standard comparison. 

    Inputs:
        path: The path to a CSV file containing data about a molecule. 
        deltaStd: Delta values for the standard, used in the simulation. 
        deltaSmp: Delta values for the sample used in the simulation. 
        deltasStdAppx: Delta values for the approximation of the standard used for the forward model of the standard. 
        abundanceThreshold, UValueList, massThreshold, disableProgress, calcFF, omitMeasurements, ffstd: Inputs to simulateMeasurement. See that function for documentation. 
        plot: Construct a plot showing the comparison. 
        MonteCarloN, perturbTheoryOAmt: Used to run the Monte Carlo routine; see ss.M1MonteCarlo.
        errorPath: The path to a CSV containing information about the experimental errors, as relative errors given as fractions (0.001 = 1 per mil). 
        MNError: IF no errorPath, then use this relative error, as a fraction (0.001 = 1 per mil), on every observed peak. 
        UValueError: IF no errorPath, use this relative error, as a fraction (0.001 = 1 per mil), on the molecular average measurement. 
        resultsFileName: Name the output CSV. 
        outputPrecision: Adjust precision on the output CSV. 
        UMNSub: Choose which isotope to anchor to when calculating site-specific data. 

    Outputs:
        simulationOutput: A dataframe containing the results of the simulation. Also produces a cleaned CSV containing the most relevant pieces of information. Also plots these data (if plot). 
    '''
    #Accept the legacy deuterium label 'D' in these arguments
    UValueList = op.standardizeSubstitutionArgument(UValueList, "the UValueList argument")
    omitMeasurements = op.standardizeSubstitutionArgument(omitMeasurements, "the omitMeasurements argument")
    MNError = op.standardizeSubstitutionArgument(MNError, "the MNError argument")
    UValueError = op.standardizeSubstitutionArgument(UValueError, "the UValueError argument")
    UMNSub = op.standardizeSubstitutionArgument(UMNSub, "the UMNSub argument")
    simArgs = {
    'abundanceThreshold': abundanceThreshold,
    'UValueList': UValueList,
    'massThreshold': massThreshold,
    'disableProgress': disableProgress,
    'calcFF': calcFF,
    'omitMeasurements': omitMeasurements,
    'ffstd': ffstd}

    #Simulate observations of the standard
    thisStd = moleculeFromCsv(path, deltas = deltasStd)
    stdMeasurement, stdMNDict, stdFF = simulateMeasurement(thisStd, **simArgs)

    #Simulate observations of the sample
    thisSmp = moleculeFromCsv(path, deltas = deltasSmp)
    smpMeasurement, smpMNDict, smpFF = simulateMeasurement(thisSmp, **simArgs)

    #Simulate a forward model of the standard. 
    forwardModel = moleculeFromCsv(path, deltas = deltasStdAppx)
    forwardModelPredictions, MNDict, FF = simulateMeasurement(forwardModel,abundanceThreshold = 0,
                                                                        massThreshold = 1,
                                                                            unresolvedDict = {})

    #Get error bars used for the simulation from an input CSV or use the same specified value for each.
    if errorPath and (MNError or UValueError):
        warnings.warn("errorPath was given, so the MNError and UValueError arguments are ignored; errors are read from the file.")
    UValueError, MNError = extractExperimentalErrors(errorPath, UValueError = UValueError, MNError = MNError)

    #'Read in' the 'experimental' data (from smpMeasurement, stdMeasurement)
    processStandard = ri.readComputedData(stdMeasurement, error = MNError, theory = forwardModelPredictions)
    processSample = ri.readComputedData(smpMeasurement, error = MNError)
    UValuesSmp = ri.readComputedUValues(smpMeasurement, error = UValueError, UMNSub = UMNSub)
    
    #Get estimates for the OValueCorrection
    OCorrection = ss.OValueCorrectTheoretical(forwardModelPredictions, processSample, massThreshold = 1)
    isotopologuesDict = fas.isotopologueDataFrame(MNDict, forwardModel['molecularDataFrame'])

    #Solve the system, update the dataframe
    M1Results = ss.M1MonteCarlo(processStandard, processSample, OCorrection, isotopologuesDict,
                                forwardModel['fragmentationDictionary'], 
                                N = MonteCarloN, perturbTheoryOAmt = perturbTheoryOAmt, disableProgress = disableProgress)

    processedResults = ss.processM1MCResults(M1Results, UValuesSmp, isotopologuesDict, forwardModel['molecularDataFrame'], UMNSub = [UMNSub], disableProgress = disableProgress)

    simulationOutput = ss.updateSiteSpecificDfM1MC(processedResults, forwardModel['molecularDataFrame'])

    #output nice csv
    cleanSimulationOutput = simulationOutput.drop(['deltas','M1 M+N Relative Abundance', 'M1 M+N Relative Abundance Error', 'UM1','UM1 Error','Calc U Values','Calc U Values Error'], axis=1, inplace=False)
    #Round off decimals
    toRound = ['CRF Deltas', 'CRF Deltas Error', 'Relative Deltas','Relative Deltas Error']
    cleanSimulationOutput[toRound] = cleanSimulationOutput[toRound].round(decimals=outputPrecision) 
    cleanSimulationOutput.to_csv(resultsFileName, index_label='Site Names')

    #Construct Plot
    if plot:
        relativeDeltasActual = [op.compareRelDelta(atomID, delta1, delta2) for atomID, delta1, delta2 in zip(forwardModel['molecularDataFrame']['IDS'], deltasStd, deltasSmp)]

        plotOutput(simulationOutput, relativeDeltasActual = relativeDeltasActual)

    return simulationOutput
