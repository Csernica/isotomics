import numpy as np
import pandas as pd

from . import basicDeltaOperations as op
from . import calcIsotopologues as ci

'''
This code extracts the concentrations of isotopologues of interest from the dictionary of all isotopologues   
in order to predict the outcomes of meaurements. It also allows one to fragment the isotopologues to compute the     
outcome of fragment measurements.                                                                                                            

It assumes one has access to a dictionary with information about the isotopologues. See calcIsotopologues.py. 
'''

def calculateUValues(bySub, massThreshold = 3, subList = None):
    '''
    Takes in an isotopologue dictionary structured bySub (see :func:`calcIsotopologues.subDictionaryFromAtom`) and calculates the U Values for that isotopologue distribution.  

    Args:
        bySub: A dictionary with information about all isotopologues of a molecule, sorted by substitution. 
        massThreshold: A mass cutoff; isotopologues with cardinal mass change above this will not be included unless indicated in subList. 
        subList: A list giving specific substitutions to calculate U values for ('13C', '15N', etc.). If substitutions are given, calculates U values only for these substitutions. Otherwise, calculates all U values below the mass threshold. 

    Returns:
        UValues: A dictionary giving the U Values. 

    Example:
        >>> import pandas as pd
        >>> from isotomics import calcIsotopologues as ci
        >>> from isotomics import fragmentAndSimulate as fas
        >>> molecular_df = pd.DataFrame({'IDS': ['C', 'O','S'], 'Number': [2, 1, 1], 'deltas': [-10, 0, 0]})
        >>> atomDict = ci.inputToAtomDict(molecular_df, disable = False, M1Only = False)
        >>> subDict = ci.subDictionaryFromAtom(atomDict)
        >>> UValues = fas.calculateUValues(subDict, massThreshold = 3)
        >>> UValues
        {'33S': 0.007877,
        '34S': 0.0441626,
        '17O': 0.00037989999999999996,
        '17O-33S': 2.9924723000000005e-06,
        '17O-34S': 1.6777371740000004e-05,
        '18O': 0.0020052,
        '18O-33S': 1.5794960400000004e-05,
        '13C': 0.0221364,
        '13C-33S': 0.0001743684228,
        ...
        
    '''
    if subList is None:
        subList = []

    unsubConc = bySub['']['Conc']

    UValues = {}
        
    for sub, info in bySub.items():
        if sub != '':
            if info['Mass'] <= massThreshold and not subList:
                UValues[sub] = info['Conc'] / unsubConc
            elif sub in subList:
                UValues[sub] = info['Conc'] / unsubConc
                
    return UValues
    

def fragMult(z, y):
    '''
    Fragments an individual site of an isotopologue. z should be 1 or 'x'. 
    
    Args:
        z: specifies whether a site is retained (1) or lost ('x')
        y: The mass of a substitution at that site

    Returns:
        'x', specifying that the site is lost, or y, specifying that the site remains. 

    Example:
        >>> # add example
    '''
    if z not in (1, 'x'):
        raise ValueError("Cannot fragment successfully, each site must be lost ('x') or retained (1)")
    return 'x' if (z == 'x' or y == 'x') else y
    
def expandFrag(siteDepict, numberAtSite):
    '''
    Creates an ATOM depiction of a fragment from a SITE depiction of a fragment.
     
    For example, suppose I have the isotopologue [0,(0,1)].
    And the fragmentation vector [01]. (Note the length of two, because there are two sites here). 
     
    I need the fragmentation vector in a form I can apply to the isotopologue. Therefore, I exapnd the fragmentation vector to yield [011] (first site lost, second site retained). 

    This function accomplishes this. 
    
    Args:
        siteDepict: SITE depiction of fragmentation vector.

    Returns:
        atomDepict: expanded depiction of fragmentation vector

    Example:
        >>> from isotomics import fragmentAndSimulate as fas
        >>> fas.expandFrag([1,'x',1], [2,1,1])
        [1, 1, 'x', 1]
    '''
    atomDepict = []
    for i, v in enumerate(siteDepict):
        atomDepict += [v] * numberAtSite[i]
    
    return atomDepict

def fragmentOneIsotopologue(fragmentationVector, isotopologue):
    '''
    Applies the ATOM fragmentation vector to the ATOM depiction of an isotopologue. Raises a warning if they are not the same length. Returns the ATOM depiction of the isotopologue with "x" in positions that are lost.
    
    Args:
        fragmentationVector: The ATOM depiction of the fragmentation vector
        isotopologue: The ATOM depiction of the isotopologue, a string

    Returns:
        A string giving the ATOM depiction of a fragmented isotopologue. 

    Example:
        >>> # add example
    '''
    #important to raise this--otherwise one may inadvertantly fragment incorrectly. 
    if len(fragmentationVector) != len(isotopologue):
           raise Exception("Cannot fragment successfully, as the fragment and the isotopologue you want to fragment have different lengths")
            
    a = [fragMult(x,y) for x, y in zip(fragmentationVector, isotopologue)]
    
    if len(a) != len(isotopologue):
        raise Exception("Cannot fragment successfully, the resulting fragment has a different length than the input isotopologue.")
    
    return ''.join(a)

def fragmentIsotopologueDict(atomIsotopologueDict, fragmentationVector, relContribution = 1):
    '''
    Applies the same fragmentation vector to all isotopologues of an input isotopologue dict and stores the results. This operation corresponds to the "fragmentation" operation from the M+N paper. Combines isotopologues which fragment to yield the same product. For the version which tracks those isotopologues, see "fragmentAndTrackIsotopologues"
    
    Args:
        atomIsotopologueDict: A dictionary containing some set of isotopologues, often a M1, M2, ... set, keyed by their ATOM depiction. 
        fragmentationVector: An ATOM depiction of a fragment
        relContribution: A float between 0 and 1, giving the relative contribution of this fragmentation geometry to the observed ion beam at that mass

    Returns: 
        fragmentedDict: A dictionary where the keys are the ATOM isotopologues after fragmentation (i.e. "0000x") and the values are the concentrations of those isotopologues. Note that this may combine isotopologues from the input dictionary which fragment in the same way; i.e. 001 and 002 both fragment to yield "00x". 

    Example:
        >>> import pandas as pd
        >>> from isotomics import calcIsotopologues as ci
        >>> from isotomics import fragmentAndSimulate as fas
        >>> molecular_df = pd.DataFrame({'IDS': ['C', 'O','S'], 'Number': [2, 1, 1], 'deltas': [-10, 0, 0]})
        >>> atomDict = ci.inputToAtomDict(molecular_df, disable = False, M1Only = False)
        >>> fragmentationVectorBySite = [1,'x',1]
        >>> fragmentationVectorByAtom = fas.expandFrag(fragmentationVectorBySite, molecular_df['Number'])
        >>> fragmentedDict = fas.fragmentIsotopologueDict(atomDict, fragmentationVector = fragmentationVectorByAtom)
        >>> fragmentedDict
        {'00x0': 0.929744362909474,
        '00x1': 0.007323596346637928,
        '00x2': 0.041059928401425944,
        '00x4': 9.787790806093197e-05,
        '01x0': 0.02058119311510928,
        '01x1': 0.0001621180581677158,
        '01x2': 0.0009089189990653254,
        '01x4': 2.1666645240000144e-06,
        '11x0': 0.00011389838081832625,
        '11x1': 8.97177545705956e-07,
        '11x2': 5.030048632727416e-06,
        '11x4': 1.199053814226848e-08}
    '''
    fragmentedDict = {}
    for isotopologue, value in atomIsotopologueDict.items():
        newIsotopologue = fragmentOneIsotopologue(fragmentationVector, isotopologue)
        if newIsotopologue not in fragmentedDict:
            fragmentedDict[newIsotopologue] = 0
        fragmentedDict[newIsotopologue] += (value['Conc'] * relContribution)
        
    return fragmentedDict
    
def computeSubs(isotopologue, IDs):
    '''
    Given an ATOM depiction of an isotopologue, computes which substitutions are present. 
    
    Args:
        isotopologue: The ATOM string depiction of an isotopologue
        IDs: The string of site elements, i.e. the output of :func:`calcIsotopologues.strSiteElements`

    Returns:
        A string giving substitutions present in that isotopologue, separated by "-". I.e. "17O-17O"

    Example:
        >>> import pandas as pd
        >>> from isotomics import calcIsotopologues as ci
        >>> from isotomics import fragmentAndSimulate as fas
        >>> molecular_df = pd.DataFrame({'IDS': ['C', 'O','S'], 'Number': [2, 1, 1], 'deltas': [-10, 0, 0]})
        >>> fas.computeSubs('11x4', ci.strSiteElements(molecular_df))
        '13C-13C-36S'
    '''
    subs = []
    for i in range(len(isotopologue)):
        if isotopologue[i] != 'x':
            element = IDs[i]
            isotope_sub = op.isotope_label(element, isotopologue[i])
            if isotope_sub != '':
                subs.append(isotope_sub)
                
    if subs == []:
        return "Unsub"
        
    return '-'.join(subs)

def computeMass(isotopologue, IDs):
    '''
    Used to predict and generate spectra with exact masses. 
    
    Args:
        isotopologue: A string, the ATOM depiction of an isotopologue.
        IDs: The string of site elements, i.e. the output of :func:`calcIsotopologues.strSiteElements`

    Returns:
        mass: A float, giving the exact mass of the isotopologue. 

    Example:
        >>> import pandas as pd
        >>> from isotomics import calcIsotopologues as ci
        >>> from isotomics import fragmentAndSimulate as fas
        >>> molecular_df = pd.DataFrame({'IDS': ['C', 'O','S'], 'Number': [2, 1, 1], 'deltas': [-10, 0, 0]})
        >>> fas.computeMass('11x4', ci.strSiteElements(molecular_df))
        61.97379038
    '''
    mass = 0
    for i in range(len(isotopologue)):
        if isotopologue[i] != 'x':
            element = IDs[i]
            mass += op.isotope_mass(element, isotopologue[i])
        
    return mass

def _fragment_representations(fragmentationDictionary, molecularDataFrame):
    """
    Build expanded fragment vectors and subgeometry keys from the
    fragmentation dictionary.
    """
    condensedFrags = []
    fragSubgeometryKeys = []
    for fragKey, subFragDict in fragmentationDictionary.items():
        for subFragNum, subFragInfo in subFragDict.items():
            condensedFrags.append(subFragInfo['subgeometry'])
            fragSubgeometryKeys.append(fragKey + '_' + subFragNum)

    atomFragList = [expandFrag(x, molecularDataFrame['Number']) for x in condensedFrags]
    return atomFragList, fragSubgeometryKeys

def predictMNFragmentExpt(MNDict, molecularDataFrame, fragmentationDictionary, abundanceThreshold = 0, omitMeasurements = None, fractionationFactors = None, calcFF = False, ffstd = 0.05, randomseed = 25, unresolvedDict = None, outputFull = False):
    '''
    Predicts the results of M+N experiements across a range of mass selected populations and fragments. This is the big 'wrapper function' that accomplishes everything and there are many options for customizing the output. 

    The output contains four distinct measures of abundance. Typically, you will want to use the Adj. Relative Abundance values. 
    
    For each M+N experiment, and for each fragment, you get:
    'Abs. Abundance': The absolute abundance of an observed ion beam, compared to all isotopologues of this compound. For example, if you have a molecule with two carbons, and your fragment has 2 carbons, the absolute abundance of 13C will be about 2%, because that is the abundance of 13C in that compound. 

    'Rel. Abundance': The relative abundance of that ion beam relative to all *ion beams in that fragment*. That is, if our fragment has ion beams for 13C, 33S, and Unsub, this gives [13C] / [13C] + [33S] + [Unsub]. It is a "M+N relative abundance", in the nomenclature of Csernica and Eiler 2023. 

    Combined Rel. Abundance: As Rel. Abundance but *includes* abundance from unresolved peaks. For example, if I measure 13C in a compound with 13C and 17O, and the 13C and 17O peaks cannot be distinguished, then the abundance of the peak labeled '13C' includes a contribution from 17O. The algorithm accomplishes this by adding the abundances of these two together, and their sum is given in the Combined Rel. Abundance of 13C. The user can specify the unresolved peaks via unresolvedDict. 

    Adj. Relative Abundance: As Combined Rel. Abundance, but adjusts for unobserved ion beams. For example, suppose my fragment has the ion beams 13C, 33S, and Unsub, but I only observe 13C and Unsub. Then when I calculate my M+1 relative abundance, I can only calculate [13C] / [13C] + [Unsub]. This value is referred to as the Adj. Relative abundance. *This should be considered the primary output of the function, unless you are doing something clever*. 

    Args:
        MNDict: A dictionary where the keys are "M0", "M1", etc. and the values are dictionaries containing all isotopologues from the ATOM dictionary with a specified cardinal mass difference. You generate this by running :func:`calcIsotopologues.massSelections`
        molecularDataFrame: A dataFrame containing information about the molecule.
        fragmentationDictionary: A dictionary, e.g. {'full': {'01': {'subgeometry': [1, 1, 1, 1, 1, 1], 'relCont': 1}},
                                                     '44': {'01': {'subgeometry': [1, 'x', 'x', 1, 1, 'x'], 'relCont': 1}}}
                                 which gives information about the fragments, their subgeometries and relative contributions. The subgeometries and relative contributions allow the user to simulate cases where the same fragment is formed via pathways which sample different sets of sites. In this circumstance, they may write the different fragmentation vectors and the relative contribution of each. The relative contributions should sum to 1. 
                                 
                                 If you have input data from a labeling experiment which enumerates how much each site contributes to an observed fragment, then currently you have to use this information to hypothesize (or guess) specific fragment subgeometries which are present and input the information that way.                                 

        abundanceThreshold: Does not include measurements below a certain relative abundance, i.e. assuming they will not be  measured due to low abundance. 
        omitMeasurements: A dictionary, {}, specifying measurements which I will not observed. For example, omitMeasurements = {'M1':{'61':'D'}} would mean I do not observe the D ion beam of the 61 fragment of the M+1 experiment, regardless of its abundance. 
        fractionationFactors: A dictionary, specifying a fractionation factor to apply to each ion beam. This is used to apply fractionation factors calculated previously to this predicted measurement (e.g. for a sample/standard comparison with the same experimental fractionation). 
        calcFF: A boolean, specifying whether new fractionation factors should be calculated via this function. If True, fractionFactors should be left empty. 
        ffstd: A float. If new fractionation factors are calculated, they are generated from a normal distribution with mean 1 and standard deviation of ffstd. 
        randomseed: An integer. If new fractionation factors are calculated, we initialize this random seed; this allows us to generate the same factors if we run multiple times. 
        unresolvedDict: A dictionary, specifying which unresolved ion beams add to each other. 
        outputFull: A boolean. Typically False, in which case beams that are not observed are culled from the output. If True, includes this information; this should only be used for debugging, and will break the solver routine. 

    Returns: 
        predictedMeasurement: A dictionary containing information from the M+N measurements. 
        calculatedFF: The calculated fractionation factors for this measurement (empty unless calcFF == True)

    Example:
        >>> import pandas as pd
        >>> from isotomics import calcIsotopologues as ci
        >>> from isotomics import fragmentAndSimulate as fas
        >>> molecular_df = pd.DataFrame({'IDS': ['C', 'O', 'S'], 'Number': [2, 1, 1], 'deltas': [-10, 0, 0]})
        >>> byAtom = ci.inputToAtomDict(molecular_df, disable=True, M1Only=False)
        >>> MNDict = ci.massSelections(byAtom, massThreshold=1)
        >>> frag_by_site = [1, 'x', 1]
        >>> fragmentationDictionary = {'frag': {'01': {'subgeometry': frag_by_site, 'relCont': 1}}}
        >>> predictedMeasurement, fractionationFactors = fas.predictMNFragmentExpt(
        ...     MNDict,
        ...     molecular_df,
        ...     fragmentationDictionary,
        ...     abundanceThreshold=0,
        ... )
        >>> predictedMeasurement['M1']['frag']
        {'33S': {'Abs. Abundance': 0.0073061703996178, 'Rel. Abundance': 0.2591689615803483, 'Combined Rel. Abundance': 0.2591689615803483, 'Adj. Rel. Abundance': 0.25916896158034836}, 'Unsub': {'Abs. Abundance': 0.0003523694471010285, 'Rel. Abundance': 0.012499465342690658, 'Combined Rel. Abundance': 0.012499465342690658, 'Adj. Rel. Abundance': 0.01249946534269066}, '13C': {'Abs. Abundance': 0.020532221713101362, 'Rel. Abundance': 0.728331573076961, 'Combined Rel. Abundance': 0.728331573076961, 'Adj. Rel. Abundance': 0.7283315730769611}}
    '''
    if omitMeasurements is None:
        omitMeasurements = {}
    if fractionationFactors is None:
        fractionationFactors = {}
    if unresolvedDict is None:
        unresolvedDict = {}

    atomFragList, fragSubgeometryKeys = _fragment_representations(fragmentationDictionary, molecularDataFrame)
    predictedMeasurement = {}
    calculatedFF = {}
    siteElements = ci.strSiteElements(molecularDataFrame)
    np.random.seed(randomseed)
    #For each population (M1, M2, M3) that we mass select
    for massSelection, MN in MNDict.items():
        #add a key to output dictionary
        if massSelection not in predictedMeasurement:
            predictedMeasurement[massSelection] = {}
            
        if calcFF == True:
            calculatedFF[massSelection] = {}

        #For each fragment we will observe
        for j, fragment in enumerate(atomFragList):

            #add a key to output dictionary
            if fragSubgeometryKeys[j] not in predictedMeasurement[massSelection]:
                predictedMeasurement[massSelection][fragSubgeometryKeys[j]] = {}
                
            if calcFF == True:
                calculatedFF[massSelection][fragSubgeometryKeys[j]] = {}
 
            #fragment the mass selection accordingly 
            fragKey, fragNum = fragSubgeometryKeys[j].split('_')
            relContribution = fragmentationDictionary[fragKey][fragNum]['relCont']
            fragmentedIsotopologues = fragmentIsotopologueDict(MN, fragment, relContribution = relContribution)

            #compute the absolute abundance of each substitution
            predictSpectrum = {}

            for key, item in fragmentedIsotopologues.items():
                sub = computeSubs(key, siteElements)
                    
                if sub not in predictSpectrum:
                    predictSpectrum[sub] = {'Abs. Abundance':0}
                predictSpectrum[sub]['Abs. Abundance'] += item
            
            #Fractionate
            if calcFF == True:
                for sub in predictSpectrum.keys():
                    ff = np.random.normal(1,ffstd)
                    calculatedFF[massSelection][fragSubgeometryKeys[j]][sub] = ff
                    predictSpectrum[sub]['Abs. Abundance'] *= ff
            
            elif fractionationFactors:
                for sub in predictSpectrum.keys():
                    predictSpectrum[sub]['Abs. Abundance'] *= fractionationFactors[massSelection][fragSubgeometryKeys[j]][sub]
                    
            predictedMeasurement[massSelection][fragSubgeometryKeys[j]] = predictSpectrum
    
    predictedMeasurement = combineFragmentSubgeometries(predictedMeasurement, fragmentationDictionary)
    
    predictedMeasurement = computeMNRelAbundances(predictedMeasurement, omitMeasurements = omitMeasurements, abundanceThreshold = abundanceThreshold, unresolvedDict = unresolvedDict, outputFull = outputFull)
                             
    return predictedMeasurement, calculatedFF

def combineFragmentSubgeometries(allMeasurementInfo, fragmentationDictionary):
    '''
    Takes fragments with multiple subgeometries and combines their measurements. For example, if frag 82 is made via 82_01 (relCont = 0.4) and 82_02 (relCont = 0.6) this function adds the values of these subfragments to give the actual measurement. 
    
    Args:
        allMeasurementInfo: A dictionary containing information about the measurement including fragment subgeometries.
        fragmentationDictionary: A dictionary giving information about the fragments and their subgeometries. 

    Returns:
        combinedAllMeasurementInfo: A dictionary containing information about the measurement including only full fragments. 

    Example:
        >>> # add example
    '''
    combinedAllMeasurementInfo = {}
    for massSelection, fragmentData in allMeasurementInfo.items():
        #only take MN experiments
        if massSelection[0] != 'M':
            combinedAllMeasurementInfo[massSelection] = fragmentData
        else:
            combinedAllMeasurementInfo[massSelection] = {}
            for fullFragKey, isotopicData in fragmentData.items():
                fragKey, fragNum = fullFragKey.split('_')
               
                if fragKey not in combinedAllMeasurementInfo[massSelection]:
                    combinedAllMeasurementInfo[massSelection][fragKey] = {}

                for isotopicSub, subData in isotopicData.items():
                    if isotopicSub not in combinedAllMeasurementInfo[massSelection][fragKey]:
                        combinedAllMeasurementInfo[massSelection][fragKey][isotopicSub] = {'Abs. Abundance':0}

                    combinedAllMeasurementInfo[massSelection][fragKey][isotopicSub]['Abs. Abundance'] += subData['Abs. Abundance']
                    
    return combinedAllMeasurementInfo
        
def computeMNRelAbundances(allMeasurementInfo, omitMeasurements = None, abundanceThreshold = 0, unresolvedDict = None, outputFull = False):
    '''
    Compute relative abundances from a MN experiment.
    
    Args:
        allMeasurementInfo: A dictionary containing information about the absolute abundance of peaks observed in the measurement. 
        omitMeasurements: Allows a user to manually specify ion beams to not measure. For example, omitMeasurements = {'M1':{'61':'D'}} would mean I do not observe the D ion beam of the 61 fragment of the M+1 experiment, regardless of its abundance. 
        abundanceThreshold: gives a relative abundance threshold (e.g. 0.01) below which peaks will not be observed. If a simulated ion beam has relative abundance below this threshold, it is culled from the predicted measurement. 
        unresolvedDict: A dictionary, specifying which unresolved ion beams add to each other. 
        outputFull: False by default. Can be set True to include information about all ion beams, not only the observed ones. This is useful for debugging. outputFull: False by default. Can be set True to include information about all ion beams, not only the observed ones. This is useful for debugging. 

    Returns:
        allMeasurementInfo: A dictionary, containing information about the relative abundances of peaks observed in the measurement. 

    Example:
        >>> # add example
    '''
    
    if omitMeasurements is None:
        omitMeasurements = {}
    if unresolvedDict is None:
        unresolvedDict = {}

    for massSelection, fragmentData in allMeasurementInfo.items():
        #only take MN experiments
        if massSelection[0] == 'M':
            #By fragment
            for fragKey, isotopicData in fragmentData.items():
                #compute relative abundance of each substitution
                totalAbundance = 0
                
                #Get abundance of each sub
                for isotopicSub, subData in isotopicData.items():
                    totalAbundance += subData['Abs. Abundance']
            
                #compute relative abundances
                for isotopicSub, subData in isotopicData.items():
                    subData['Rel. Abundance'] = subData['Abs. Abundance'] / totalAbundance
                
                #Coalescing peaks--if we are moving abundance from one substitution to another
                for isotopicSub, subData in isotopicData.items():
                    #check to see if we have to 
                    try:
                        #if we do, set the coalesced relative abundance of the old sub to 0
                        newSub = unresolvedDict[massSelection][fragKey][isotopicSub]
                        subData['Combined Rel. Abundance'] = 0
                    except:
                        newSub = isotopicSub
                        
                    #Then find the new substitution
                    newSubData = allMeasurementInfo[massSelection][fragKey][newSub]
                    
                    #Add the old subs relative abundance to the new sub
                    if 'Combined Rel. Abundance' not in newSubData:
                        newSubData['Combined Rel. Abundance'] = subData['Rel. Abundance']
                    else:
                        newSubData['Combined Rel. Abundance'] += subData['Rel. Abundance']
                            
                #Calculate adjusted relative abundance, which does not include contributions from peaks below some
                #threshold
                shortSpectrum = {}
                totalRelAbund = 0
                try:
                    forbiddenPeaks = omitMeasurements[massSelection][fragKey]
                except:
                    forbiddenPeaks = []

                for isotopicSub, subData in isotopicData.items():
                    #If the peak is observed, count it
                    if subData['Combined Rel. Abundance'] > abundanceThreshold and isotopicSub not in forbiddenPeaks:
                        shortSpectrum[isotopicSub] = subData
                        totalRelAbund += subData['Combined Rel. Abundance']
                    #Otherwise, either 1) set Adj. Rel. Abundance to 0, keeping it in the spectrum or
                    #                  2) cull it from the spectrum 
                    else:
                        if outputFull:
                            shortSpectrum[isotopicSub] = subData
                            shortSpectrum[isotopicSub]['Adj. Rel. Abundance'] = 0

                #calculate adj. rel. abundance for the qualifying peaks
                for isotopicSub, subData in shortSpectrum.items():
                    #If we added adj. rel. abundance = 0 the previous step, we don't want to repeat that calculation
                    if 'Adj. Rel. Abundance' not in subData:
                        subData['Adj. Rel. Abundance'] = subData['Combined Rel. Abundance'] / totalRelAbund
                    
                allMeasurementInfo[massSelection][fragKey] = shortSpectrum
                
    return allMeasurementInfo

def trackMNFragments(MN, expandedFrags, fragSubgeometryKeys, molecularDataFrame, unresolvedDict = None):
    '''
    Fragments and tracks isotopologues across a range of mass selections and fragments. 
    
    Args:
        MN: A dictionary, where the key is an MNKey and the values give information about all isotopologues associated with that mass selection. 
        expandedFrags: A list of the expanded fragments
        fragSubgeometryKeys: A list of the fragment subgeometry keys 
        molecularDataFrame: The initial dataframe with information about the molecule
        unresolvedDict: A dictionary, specifying which unresolved ion beams add to each other. 

    Returns:
        MN: The same dictionary, with information about fragmentation added. 

    Example:
        >>> # add example
    '''
    if unresolvedDict is None:
        unresolvedDict = {}

    unsubString = list(MN['M0'].keys())[0]
    UnsubConc = MN['M0'][unsubString]['Conc']

    for key in list(MN.keys()):
        massSelection = MN[key]
        for i, fragment in enumerate(expandedFrags):
            fragmentAndTrackIsotopologues(massSelection, fragment, fragSubgeometryKeys[i], UnsubConc, molecularDataFrame, unresolvedDict = unresolvedDict)
            
    return MN

def fragmentAndTrackIsotopologues(massSelection, atomFrag, fragmentKey, unsubConc, molecularDataFrame, unresolvedDict = None):
    '''
    Fragments isotopologues and tracks which parent isotopologues end up in which product. For the version that combines isotopologues, for simulating measurements, see fragmentIsotopologueDict (that is, if 001 and 002 both form 00x on fragmentation, this function tracks 001 and 002 explictly; fragmentIsotopologueDict only reports 00x). This function fills in a dictionary with the isotopologues introduced to be fragmented by identifying the product and substitutions of each. 
    
    Args:
        massSelection: A subset of isotopologues indexed using the ATOM depiction. 
        atomFrag: An ATOM depiction fragment
        fragmentKey: A string giving the identity of the fragment. 
        unsubConc: The concentration of the unsubstituted isotopologue. 
        molecularDataFrame: A dataFrame containing information about the molecule.
        unresolvedDict: {'133':{'17O':'13C'}}

    Returns:
        massSelection: The same dictionary, updated to include information about fragmentation. 

    Example:
        >>> # add example
    '''
    if unresolvedDict is None:
        unresolvedDict = {}

    siteElements = ci.strSiteElements(molecularDataFrame)
    
    fragmentedDict = {}
    for isotopologue, value in massSelection.items():
        value['Stochastic U'] = value['Conc'] / unsubConc
        frag = [fragMult(x,y) for x, y in zip(atomFrag, isotopologue)]
        newIsotopologue = ''.join(frag)
        massSelection[isotopologue][fragmentKey + ' Identity'] = newIsotopologue
        
        sub = computeSubs(newIsotopologue, siteElements)
            
        #If unresolved peaks are a problem
        if fragmentKey in unresolvedDict:
            if sub in unresolvedDict[fragmentKey]:
                sub = unresolvedDict[fragmentKey][sub]
            
        massSelection[isotopologue][fragmentKey + ' Subs'] = sub
        
    return massSelection

def isotopologueDataFrame(MNDictionary, molecularDataFrame):
    '''
    Given a dictionary containing different mass selections, iterates through each mass selection. Extracts the isotopologues from each and puts them into a dataframe, identifying their concentration, substitution, as well as a long string giving a "precise identity", i.e. including explicit labels. Returns these as a dictionary with keys "M0", "M1", etc. where the values are dataFrames of the isotopologues. 
    
    Args:
        MNDictionary: A dictionary containing different mass selections, i.e. the output of fragmentAndTrackIsotopologues
        molecularDataFrame: A dataFrame containing information about the molecule.

    Returns:
        isotopologuesDict: A dictionary where the keys are "M0", "M1", etc. and the values are dataFrames giving the isotopologues with those substitutions. 

    Example:
        >>> # add example
    '''
    
    isotopologuesDict = {}
    siteElements = ci.strSiteElements(molecularDataFrame)
    
    for key in list(MNDictionary.keys()):
        massSelection = MNDictionary[key]
    
        Isotopologues = pd.DataFrame.from_dict(massSelection).T
        Isotopologues.rename(columns={'Conc':'Stochastic',"Subs": "Composition"},inplace = True)
        
        preciseStrings = []
        
        expandedIndices = []
        for i, n in enumerate(molecularDataFrame.Number):
            expandedIndices += n * [molecularDataFrame.index[i]]
        
        for i, v in Isotopologues.iterrows():
            Subs = [ci.uEl(element, int(number)) for element, number in zip(siteElements, i)]
           
            Precise = [x + " " + y for x, y in zip(Subs, expandedIndices) if x != '']
            output = '   |   '.join(Precise)
            preciseStrings.append(output)
        Isotopologues['Precise Identity'] = preciseStrings
        Isotopologues.sort_values('Composition',inplace = True)
        
        isotopologuesDict[key] = Isotopologues
        
    return isotopologuesDict

def predictTraditionalMeasurement(molecularDataFrame, atomFrag, byAtom):
    '''
    Currently a shell compared to predictMNFragmentation. Occasionally, it will be worthwhile for the user to simulate a 'traditional' fragmentation measurement (i.e., without mass selection). This implements that and returns a dictionary where keys are substitutions and values are the abundances. 

    Less developed; it does not account for fractionation and the like and is not integrated into any other routines. I include it here in case a user wants to know how to implement this. 

    Currently applies to one fragment at a time. Need to build a loop for multiple fragments. 

    Args:
        molecularDataFrame: Basic information about the molecule. 
        atomFrag: The ATOM depiction of the fragment
        byAtom: A dictionary with all isotopologues to fragment. 

    Returns:
        predictSpectrum: A dictionary. Keys are subKeys (e.g., 'D-D-13C') and values are the abundances of those after fragmentation. 

    Example:
        >>> # add example
    '''
    predictSpectrum = {}
    siteElements = ci.strSiteElements(molecularDataFrame)
    fragmentedIsotopologues = fragmentIsotopologueDict(byAtom, atomFrag, relContribution = 1)

    for isoKey, isoData in fragmentedIsotopologues.items():
        subKey = computeSubs(isoKey, siteElements)
                    
        if subKey not in predictSpectrum:
            predictSpectrum[subKey] = {'Abs. Abundance':0}
        predictSpectrum[subKey]['Abs. Abundance'] += isoData

    return predictSpectrum
