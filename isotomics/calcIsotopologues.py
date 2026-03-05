import itertools
import copy
import math

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import basicDeltaOperations as op

'''                                                 
This code calculates a dictionary giving all possible isotopologues of a molecule and their concentrations, based on input information about the sites and their isotopic composition.                                                 

It assumes one has access to a dataframe specifying details about a molecule. See the tutorial.
'''

#The possible substitutions, by cardinal mass, for each element. 
setsOfElementIsotopes = {'H':(0,1),'N':(0,1),'C':(0,1),'O':(0,1,2),'S':(0,1,2,4)}

def calculateSetsOfSiteIsotopes(molecularDataFrame):
    """
    Compute the isotope combinations and multinomial coefficients for each site.

    The dataframe divides the molecule into sites, each containing indistinguishable
    atoms. A site may be indistinguishable in principle (e.g., symmetry) or in
    practice (e.g., fragmentation). For single-atom sites, the possible isotopes
    are the element isotopes. For multi-atom sites, the possible isotope states
    are generated from combinations with replacement.

    In this notation, substitutions are encoded as tuples. For example, a
    two-carbon site is represented as ``((0, 0), (0, 1), (1, 1))``, where
    ``0`` corresponds to 12C and ``1`` corresponds to 13C.

    Args:
        molecularDataFrame (pandas.DataFrame): Molecule definition table with at
            least ``'IDS'`` (element symbol per site) and ``'Number'`` (atoms per
            site) columns.

    Returns:
        tuple[list, list]:
            - setsOfSiteIsotopes: Tuple/list structure where item ``i`` contains
              the allowed isotope combinations at site ``i``.
            - multinomialCoefficients: Tuple/list structure where item ``i``
              contains multinomial multiplicities matching
              ``setsOfSiteIsotopes[i]``.

    Example:
        >>> import pandas as pd
        >>> from isotomics import calcIsotopologues as ci
        >>> molecular_df = pd.DataFrame({
        ...     'Site Names': ['Calphabeta', 'Ccarboxyl', 'Ocarboxyl', 'Namine', 'Hretained', 'Hlost'],
        ...     'IDS': ['C', 'C', 'O', 'N', 'H', 'H'],
        ...     'Number': [2, 1, 2, 1, 6, 2],
        ... })
        >>> sets_of_site_isotopes, multinomial_coefficients = ci.calculateSetsOfSiteIsotopes(molecular_df)

        Example output (abridged)::

            sets_of_site_isotopes[0]
            ((0, 0), (0, 1), (1, 1))

            multinomial_coefficients[0]
            [[1, 1], [2, 2], [1, 1]]
    """
    elIDs = molecularDataFrame['IDS'].values
    numberAtSite = molecularDataFrame['Number'].values

    siteList = [(x,y) for x,y in zip(elIDs, numberAtSite)]

    #Determine the set of Site Isotopes for every site
    setsOfSiteIsotopes = []
    multinomialCoefficients = []

    for site in siteList:
        el = site[0]
        n = site[1]

        if n == 1:
            setsOfSiteIsotopes.append(setsOfElementIsotopes[el])
            multinomialCoefficients.append([1] * len(setsOfElementIsotopes[el]))

        else:
            siteIsotopes = tuple(itertools.combinations_with_replacement(setsOfElementIsotopes[el], n))
            setsOfSiteIsotopes.append(siteIsotopes)

            #Determining the multinomial coefficients takes a bit of work. There must be a more elegant way to do so.
            #First we generate all possible isotopic structures
            siteIsotopicStructures = itertools.product(setsOfElementIsotopes[el],repeat = n)

            #Then we sort each structure, i.e. so that [0,1] and [1,0] both become [0,1]
            sort = [sorted(list(x)) for x in siteIsotopicStructures]

            #Then we count how many times each structure occurs; i.e. [0,1] may appear twice
            counts = []
            for siteComposition in siteIsotopes:
                c = 0
                for isotopeStructure in sort:
                    if list(siteComposition) == isotopeStructure:
                        c += 1
                counts.append(c)

            #Finally, we expand the counts. Suppose we have a set of site Isotopes with [0,0], [0,1], [1,1] with 
            #multinomial Coefficients of [1,2,1], respectively. We want to output [[1,1],[2,2],[1,1]] rather than 
            #[1,2,1], because doing so will allow us to take advantage of the optimized itertools.product function
            #to calculate the symmetry number of isotopologues with many multiatomic sites.

            #One can check that by doing so, "multinomialCoefficients" and "setsOfSiteIsotopes", the output variables
            #from this section, have the same form. 
            processedCounts = [[x] * n for x in counts]

            multinomialCoefficients.append(processedCounts)
            
    return setsOfSiteIsotopes, multinomialCoefficients

def calcAllIsotopologues(setsOfSiteIsotopes, multinomialCoefficients, M1Only = False):
    """
    This takes in both a set of site isotopes and a set of multinomial coefficients, which are the outputs of calculateSetsOfSiteIsotopes. You take those outputs and feed them directly into here. Then this function generates the set of all isotopologues and their symmetry numbers. 

    For large systems, set ``M1Only=True`` to compute only the unsubstituted and
    M+1 isotopologues. (My personal computer runs out of storage around 10 million isotopoplogues).

    Args:
        setsOfSiteIsotopes (list[tuple]): Site-level isotopes from
            :func:`calculateSetsOfSiteIsotopes`.
        multinomialCoefficients (list[tuple]): Site-level multiplicities from
            :func:`calculateSetsOfSiteIsotopes`.
        M1Only (bool, optional): If ``True``, return only unsubstituted and M+1
            isotopologues. Defaults to ``False``.

    Returns:
        tuple[list[tuple], list[int]]:
            - setOfAllIsotopologues: Isotopologue tuple per molecular state.
            - symmetryNumbers: Multiplicity for each isotopologue, same order as
              ``setOfAllIsotopologues``.

    Examples:
        >>> import pandas as pd
        >>> from isotomics import calcIsotopologues as ci
        >>> molecular_df = pd.DataFrame({'IDS': ['C', 'C'], 'Number': [2, 1], 'deltas': [0, 0]})
        >>> sets_of_site_isotopes, multinomial_coefficients = ci.calculateSetsOfSiteIsotopes(molecular_df)
        >>> isotopologues, symmetry_numbers = ci.calcAllIsotopologues(sets_of_site_isotopes, multinomial_coefficients, M1Only=True)
        >>> isotopologues, symmetry_numbers
        ([((0, 0), 0),
        ((0, 0), 1),
        ((0, 1), 0),
        ((0, 1), 1),
        ((1, 1), 0),
        ((1, 1), 1)],
        [1, 1, 2, 2, 1, 1])
    """
    if M1Only:
        setOfM1Isotopologues, symmetryNumbers = calcThroughM1Isotopologues(setsOfSiteIsotopes)
        return setOfM1Isotopologues, symmetryNumbers
    
    i = 0
    setOfAllIsotopologues = []
    symmetryNumbers = []
    for isotopologue in itertools.product(*setsOfSiteIsotopes):
        setOfAllIsotopologues.append(isotopologue)

    #As setsOfSiteIsotopes and multinomialCoefficients are in the same form, we can use the optimized itertools.product
    #again to efficiently calculate the symmetry numbers
    for isotopologue in itertools.product(*multinomialCoefficients):
        flat = [x[0] if type(x) == list else x for x in isotopologue]
        n = np.array(flat).prod()

        symmetryNumbers.append(n)
                             
    return setOfAllIsotopologues, symmetryNumbers

def calcThroughM1Isotopologues(setsOfSiteIsotopes):
    """
    This is a reduced alternative to :func:`calcAllIsotopologues` for that only calculates through the M+1 isotopologues. Way faster and avoids memory issues if all you are doing is a M+1 experiment. Typically called by :func:`calcAllIsotopologues` with M1Only = True rather than directly. 

    Args:
        setsOfSiteIsotopes (list[tuple]): Site-level isotopes from
            :func:`calculateSetsOfSiteIsotopes`.

    Returns:
        tuple[list[tuple], list[int]]:
            - setOfM1Isotopologues: Unsubstituted isotopologue plus M+1 entries.
            - symmetryNumbers: Multiplicity for each isotopologue.

    Examples:
        >>> from isotomics import calcIsotopologues as ci
        >>> molecular_df = pd.DataFrame({'IDS': ['C','C'], 'Number': [2,1], 'deltas': [0,0]})
        >>> sets_of_site_isotopes, multinomial_coefficients = ci.calculateSetsOfSiteIsotopes(molecular_df)
        >>> isotopologues, symmetry_numbers = ci.calcThroughM1Isotopologues(sets_of_site_isotopes)
        >>> isotopologues, symmetry_numbers
        ([[(0, 0), 0], ((0, 1), 0), ((0, 0), 1)], [1, 2, 1])
    """
    symmetryNumbers = []
    setOfM1Isotopologues = []
    unsubstitutedIsotopologue = []

    for siteIsotope in setsOfSiteIsotopes:
        #siteIsotope will contain integers (for single sites) or tuples (for multiatomic sites). So here we are
        #checking only single sites
        if 0 in siteIsotope:
            unsubstitutedIsotopologue.append(0)

        else:
            n = len(siteIsotope[0])
            zeroTup = tuple([0]*n)
            unsubstitutedIsotopologue.append(zeroTup)

    setOfM1Isotopologues.append(unsubstitutedIsotopologue)
    symmetryNumbers.append(1)

    for index, siteIsotope in enumerate(setsOfSiteIsotopes):
        isotopologue = copy.copy(unsubstitutedIsotopologue)
        
        if 1 in siteIsotope:
            #siteIsotope will contain integers (for single sites) or tuples (for multiatomic sites). So here we are
            #checking only single sites
            isotopologue[index] = 1
            symmetryNumbers.append(1)
            setOfM1Isotopologues.append(tuple(isotopologue))

        else:
            n = len(siteIsotope[0])

            singleM1Sub = [0] * n
            singleM1Sub[-1] = 1
            singleM1SubTup = tuple(singleM1Sub)

            if singleM1SubTup in siteIsotope:
                isotopologue[index] = singleM1SubTup
                setOfM1Isotopologues.append(tuple(isotopologue))
                symmetryNumbers.append(n)
            
    return setOfM1Isotopologues, symmetryNumbers

def siteSpecificConcentrations(molecularDataFrame):
    """
    Convert per-site deltas into isotope concentration arrays.

    The output structure is indexed by cardinal mass difference: entry ``i``
    contains concentrations for substitutions of mass ``M+i`` at each site.

    Args:
        molecularDataFrame (pandas.DataFrame): Molecule definition table with
            ``'IDS'``, ``'Number'``, and ``'deltas'`` columns.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            Arrays for M0, M1, M2, M3, and M4 site concentrations, respectively.

    Examples:
        >>> import pandas as pd
        >>> from isotomics import calcIsotopologues as ci
        >>> molecular_df = pd.DataFrame({'IDS': ['C', 'N'], 'Number': [1, 1], 'deltas': [-10, 0]})
        >>> concentration_array = ci.siteSpecificConcentrations(molecular_df)
        >>> [arr.shape for arr in concentration_array]
        [(2,), (2,), (2,), (2,), (2,)]
    """
    elIDs = molecularDataFrame['IDS'].values
    numberAtSite = molecularDataFrame['Number'].values
    deltas = molecularDataFrame['deltas'].values
    
    concentrationList = []
    for index in range(len(elIDs)):
        element = elIDs[index]
        delta = deltas[index]
        concentration = op.deltaToConcentration(element, delta)
        concentrationList.append(concentration)

    #put site-specific concentrations into a workable form
    unsub = []
    M1 = []
    M2 = []
    M3 = []
    M4 = []
    for concentration in concentrationList:
        unsub.append(concentration[0])
        M1.append(concentration[1])
        M2.append(concentration[2])
        M3.append(0)
        M4.append(concentration[3])

    concentrationArray = np.array(unsub), np.array(M1), np.array(M2), np.array(M3), np.array(M4)
    
    return concentrationArray

def calculateIsotopologueConcentrations(setOfAllIsotopologues, symmetryNumbers, concentrationArray, disable = False):
    """
    Compute concentrations for each isotopologue in the setOfAllIsotopologues based on the site-specific concentrations and assuming stochasticitiy. 

    For isotopologues with symmetry numbers greater than 1, the computed concentration sums across all versions of that isotopologue. Therefore, the sum of the concentrations in the output should be 1. 

    Args:
        setOfAllIsotopologues (list[tuple]): Isotopologue tuples.
        symmetryNumbers (list[int]): Multiplicity for each isotopologue.
        concentrationArray (tuple[numpy.ndarray, ...]): Output of
            :func:`siteSpecificConcentrations`.
        disable (bool, optional): If ``True``, hide tqdm progress bars.
            Defaults to ``False``.

    Returns:
        dict[str, dict]: Mapping from isotopologue string representation to a
        record with:

            - ``'Conc'``: concentration of that isotopologue class.
            - ``'num'``: symmetry number / multiplicity.

    Examples:
        >>> import pandas as pd
        >>> from isotomics import calcIsotopologues as ci
        >>> molecular_df = pd.DataFrame({'IDS': ['C','C'], 'Number': [2,1], 'deltas': [0,0]})
        >>> sets_of_site_isotopes, multinomial_coefficients = ci.calculateSetsOfSiteIsotopes(molecular_df)
        >>> isotopologues, symmetry_numbers = ci.calcAllIsotopologues(sets_of_site_isotopes, multinomial_coefficients)
        >>> concentration_array = ci.siteSpecificConcentrations(molecular_df)
        >>> d = ci.calculateIsotopologueConcentrations(isotopologues, symmetry_numbers, concentration_array, disable=True)
        >>> d
        {'(0, 0)0': {'Conc': 0.9671962109820917, 'num': 1},
            '(0, 0)1': {'Conc': 0.010813253638779786, 'num': 1},
            '(0, 1)0': {'Conc': 0.021626507277559572, 'num': 2},
            '(0, 1)1': {'Conc': 0.00024178435136311605, 'num': 2},
            '(1, 1)0': {'Conc': 0.00012089217568155803, 'num': 1},
            '(1, 1)1': {'Conc': 1.3515745241198188e-06, 'num': 1}}

        >>> s = 0 
        >>> for i, v in d.items():
        ...     s += v['Conc']
        >>> s
        0.9999999999999999
    """
    d = {}
    for i, isotopologue in enumerate(tqdm(setOfAllIsotopologues, disable = disable)):
        number = symmetryNumbers[i]
        isotopeConcList = []
        for index, value in enumerate(isotopologue):
            if type(value) == tuple:
                isotopeConc = [concentrationArray[sub][index] for sub in value]
                isotopeConcList += isotopeConc
            else:
                isotopeConc = concentrationArray[value][index]
                isotopeConcList.append(isotopeConc)      

        isotopologueConc = np.array(isotopeConcList).prod()

        string = ''.join(map(str, isotopologue))

        d[string] = {'Conc':isotopologueConc * number,'num':number}
        
    return d

def condenseStr(text):
    """
    Convert expanded isotopologue strings to ATOM strings.

    Expanded strings for multi-atom sites (for example ``"(0,1)0"``) are
    flattened to ATOM-style strings (for example ``"010"``).

    Args:
        text (str): Expanded isotopologue string.

    Returns:
        str: Condensed ATOM string.

    Examples:
        >>> from isotomics import calcIsotopologues as ci
        >>> ci.condenseStr('(0,1)0')
        '010'
    """
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace(',', '')
    text = text.replace(' ', '')
    
    return text
 
def uEl(el, n):
    """
    Map an element and cardinal mass shift to substitution label.

    Args:
        el (str): Element symbol (``'C'``, ``'H'``, ``'N'``, ``'O'``, or ``'S'``).
        n (int | str): Cardinal mass shift. ``0`` or ``'x'`` returns an empty
            string.

    Returns:
        str: Substitution label (for example ``'13C'`` or ``'18O'``), or an empty
        string for unsubstituted positions.

    Examples:
        >>> from isotomics import calcIsotopologues as ci
        >>> ci.uEl('C', 1)
        '13C'
        >>> ci.uEl('C', 0)
        ''
    """
    if n == 0:
        return ''
    if n == 'x':
        return ''
    if el == 'C':
        if n == 1:
            return '13C'
    if el == 'H':
        if n == 1:
            return 'D'
    if el == 'O':
        if n == 1:
            return '17O'
        if n == 2:
            return '18O'
    if el == 'N':
        if n == 1:
            return '15N'
    if el == 'S':
        if n == 1:
            return '33S'
        if n == 2:
            return '34S'
        if n == 4:
            return '36S'

def strSiteElements(molecularDataFrame):
    """
    The molecular dataframe is defined with sites, which each have a number and an element ID.

    This function uses those values to compute an isotoplogue string where each atom position is represented explicitly. For example, if we have a site with element ID 'N' and number 2, this function will return 'NN' for that site. It repeats this process on all sites to compile a full string of site elements. 
 
    Args:
        molecularDataFrame (pandas.DataFrame): Molecule definition table with
            ``'IDS'`` and ``'Number'`` columns.

    Returns:
        str: Expanded element string by atomic position.

    Examples:
        >>> import pandas as pd
        >>> from isotomics import calcIsotopologues as ci
        >>> molecular_df = pd.DataFrame({'IDS': ['N', 'O'], 'Number': [2, 1]})
        >>> ci.strSiteElements(molecular_df)
        'NNO'
    """
    elIDs = molecularDataFrame['IDS'].values
    numberAtSite = molecularDataFrame['Number'].values

    siteList = [(x,y) for x,y in zip(elIDs, numberAtSite)]
    siteElementsList = [site[0] * site[1] for site in siteList]
    siteElements = ''.join(siteElementsList)
    
    return siteElements

def calcAtomDictionary(isotopologueConcentrationDict, molecularDataFrame, disable = False):
    """
    Build an isotopologue dictionary keyed by ATOM strings.

    Converts isotopologue keys from expanded notation to ATOM notation and adds
    derived metadata including total mass shift and substitution labels.

    Args:
        isotopologueConcentrationDict (dict): Output of
            :func:`calculateIsotopologueConcentrations`.
        molecularDataFrame (pandas.DataFrame): Molecule definition table.
        disable (bool, optional): If ``True``, hide tqdm progress bars.
            Defaults to ``False``.

    Returns:
        dict[str, dict]: Dictionary keyed by ATOM string. Each value includes
        ``'Number'``, ``'Full'``, ``'Conc'``, ``'Mass'``, and ``'Subs'``.

    Examples:
        >>> import pandas as pd
        >>> from isotomics import calcIsotopologues as ci
        >>> molecular_df = pd.DataFrame({'Site Names': ['C1'], 'IDS': ['C'], 'Number': [1], 'deltas': [0]}).set_index('Site Names')
        >>> by_atom = ci.inputToAtomDict(molecular_df, disable=True)
        >>> sorted(by_atom.keys())
        ['0', '1']
    """
    siteElements = strSiteElements(molecularDataFrame)
    
    byAtom = {}
    for i, v in tqdm(isotopologueConcentrationDict.items(), disable = disable):
        ATOM = condenseStr(i)
        byAtom[ATOM] = {}
        byAtom[ATOM]['Number'] = v['num']
        byAtom[ATOM]['Full'] = i
        byAtom[ATOM]['Conc'] = v['Conc']
        byAtom[ATOM]['Mass'] = np.array(list(map(int,ATOM))).sum()
        byAtom[ATOM]['Subs'] = '-'.join(filter(None, [uEl(element, int(number)) for element, number in zip(siteElements, ATOM)]))
    
    return byAtom

def subDictionaryFromAtom(atomDict):
    """
    The 'atom' dictionary has keys corresponding to atom strings. Often, it is useful to retrieve information by substituion rather than atom string. For example, one may want to know the concentration of all isotopologues with a 13C substitution. This function reorganizes the atom dictionary by substitution. 

    Args:
        atomDict (dict): Output of       :func:`inputToAtomDict`.

    Returns:
        dict[str, dict]: Dictionary keyed by substitution string. Each value
        includes aggregated ``'Number'``, ``'Full'``, ``'Conc'``, ``'Mass'``, and
        ``'ATOM'`` fields.

    Examples:
        >>> import pandas as pd
        >>> from isotomics import calcIsotopologues as ci
        >>> molecular_df = pd.DataFrame({'IDS': ['C', 'O','S'], 'Number': [2, 1,1], 'deltas': [-10, 0,0]})
        >>> atomDict = ci.inputToAtomDict(molecular_df, disable = False, M1Only = False)
        >>> subDict = ci.subDictionaryFromAtom(atomDict)
        >>> subDict
        {'': {'Number': 1,
        'Full': ['(0, 0)00'],
        'Conc': 0.9275321060832549,
        'Mass': 0,
        'ATOM': ['0000']},
        '33S': {'Number': 1,
        'Full': ['(0, 0)01'],
        'Conc': 0.0073061703996178,
        'Mass': 1,
        'ATOM': ['0001']},
        '34S': {'Number': 1,
        'Full': ['(0, 0)02'],
        'Conc': 0.04096222938811236,
        'Mass': 2,
        'ATOM': ['0002']},
        '36S': {'Number': 1,
        'Full': ['(0, 0)04'],
        'Conc': 9.764501493580858e-05,
        'Mass': 4,
        'ATOM': ['0004']},
        '17O': {'Number': 1,
        'Full': ['(0, 0)10'],
        'Conc': 0.0003523694471010285,
        'Mass': 1,
        'ATOM': ['0010']},
        ...
    """
    bySub = {}
    for i, v in atomDict.items():
        Subs = v['Subs']
        if Subs not in bySub:
            bySub[Subs] = {'Number': 0, 'Full': [], 'Conc': 0, 'Mass': 0, 'ATOM': []}
        bySub[Subs]['Number'] += v['Number']
        bySub[Subs]['Full'].append(v['Full'])
        bySub[Subs]['Conc'] += v['Conc']
        #note mass is reassigned for each substitution. Should be fine as they are all identical. 
        bySub[Subs]['Mass'] = v['Mass']
        bySub[Subs]['ATOM'].append(i)

    return bySub

def inputToAtomDict(molecularDataFrame, disable = False, M1Only = False):
    """
    The highest level function for calculating the isotopologues of a system. 
    
    This takes in a molecular dataframe, giving information about a compound. It then calculates the isotopologues and their concentrations, number, and substitutions, and compiles these into a dictionary. 

    It is referred to as an "Atom" dict because the representation of each isotopologue gives its substitution pattern by atom. For example, an isotopologue is given as a string such as '0001', where each position corresponds to an atom and the value at that position corresponds to the isotope present. '0001', for example, has the first three atoms unsubstituted and the fourth substituted. 

    This will probably fail on personal computers past 10 million isotopologues or so. M1Only causes it to only compute the M+1 isotopologues, which is much faster and gives you everything you need for a M+1 experiment. 

    For isotopologues with symmetry numbers above 1, the concentrations in this dictionary give the sum across all instances of that isotopologue. Therefore, the sum of all concentrations in this dictionary is 1.

    This function also records the total mass shift of each isotopologue, the substitutions present, and a string called 'Full' which gives the isotopologue in expanded format (e.g., '(0, 0)00' rather than '0000'). This broader description may be useful to see information about sites. It is also worthwhile to note that certain combinations that may seem reasonable will not appear based on the structure of 'Full'. For example '(1, 0)00' will not appear because the first site is a two-atom site and these fill in from the right; the only combinations are '(0,0)', '(0,1)', and '(1,1)'. Therefore, the isotopologue '1000' does not exist but '0100' does.  

    Args:
        molecularDataFrame (pandas.DataFrame): Molecule definition table.
        disable (bool, optional): If ``True``, hide tqdm progress bars.
            Defaults to ``False``.
        M1Only (bool, optional): If ``True``, compute only unsubstituted and M+1
            isotopologues. Defaults to ``False``.

    Returns:
        dict[str, dict]: By-atom isotopologue dictionary.

    Examples:
        >>> import pandas as pd
        >>> from isotomics import calcIsotopologues as ci
        >>> molecular_df = pd.DataFrame({'IDS': ['C', 'O','S'], 'Number': [2, 1,1], 'deltas': [-10, 0,0]})
        >>> atomDict = ci.inputToAtomDict(molecular_df, disable = False, M1Only = False)
        >>> atomDict
        {'0000': {'Number': 1,
            'Full': '(0, 0)00',
            'Conc': 0.9275321060832549,
            'Mass': 0,
            'Subs': ''},
            '0001': {'Number': 1,
            'Full': '(0, 0)01',
            'Conc': 0.0073061703996178,
            'Mass': 1,
            'Subs': '33S'},
            '0002': {'Number': 1,
            'Full': '(0, 0)02',
            'Conc': 0.04096222938811236,
            'Mass': 2,
            'Subs': '34S'},
            '0004': {'Number': 1,
            'Full': '(0, 0)04',
            'Conc': 9.764501493580858e-05,
            'Mass': 4,
            'Subs': '36S'},
            '0010': {'Number': 1,
            'Full': '(0, 0)10',
            'Conc': 0.0003523694471010285,
            'Mass': 1,
            'Subs': '17O'},
            ...

        >>> totalC = 0
        >>> for isotopologues, isoData in atomDict.items():
        ...     totalC += isoData['Conc']
        >>> totalC
        1.0000000000000002
    """
    if disable == False:
        print("Calculating Isotopologue Concentrations")
    siteElements = strSiteElements(molecularDataFrame)
    siteIsotopes, multinomialCoeff = calculateSetsOfSiteIsotopes(molecularDataFrame)
    bigA, SN = calcAllIsotopologues(siteIsotopes, multinomialCoeff, M1Only = M1Only)
    concentrationArray = siteSpecificConcentrations(molecularDataFrame)
    d = calculateIsotopologueConcentrations(bigA, SN, concentrationArray, disable = disable)

    if disable == False:
        print("Compiling Isotopologue Dictionary")
    byAtom = calcAtomDictionary(d, molecularDataFrame, disable = disable)
    
    return byAtom

def massSelections(isotopologueDictionary, massThreshold = 4):
    """
    Takes a by atom or by sub isotopologue dictionary and pulls out only those isotopologues with a cardinal mass increase <= massThreshold. 

    Args:
        isotopologueDictionary (dict[str, dict]): Either an atom or sub dictionary; the output of :func:`inputToAtomDict` or :func:`subDictionaryFromAtom`.
        massThreshold (int, optional): Highest mass class to include. Defaults to
            ``4``.

    Returns:
        dict[str, dict]: Dictionary with keys ``'M0'`` ... ``'M{massThreshold}'``;
        each value contains isotopologues from that mass class.

    Examples:
        >>> import pandas as pd
        >>> from isotomics import calcIsotopologues as ci
        >>> molecular_df = pd.DataFrame({'IDS': ['C', 'O','S'], 'Number': [2, 1,1], 'deltas': [-10, 0,0]})
        >>> atomDict = ci.inputToAtomDict(molecular_df, disable = False, M1Only = False)
        >>> selectedIsotopologues = ci.massSelections(atomDict, massThreshold=2)
        >>> selectedIsotopologues
        'M0': {'0000': {'Number': 1,
            'Full': '(0, 0)00',
            'Conc': 0.9275321060832549,
            'Mass': 0,
            'Subs': ''}},
            'M1': {'0001': {'Number': 1,
            'Full': '(0, 0)01',
            'Conc': 0.0073061703996178,
            'Mass': 1,
            'Subs': '33S'},
            '0010': {'Number': 1,
            'Full': '(0, 0)10',
            'Conc': 0.0003523694471010285,
            'Mass': 1,
            'Subs': '17O'},
            '0100': {'Number': 2,
            'Full': '(0, 1)00',
            'Conc': 0.020532221713101362,
            'Mass': 1,
            'Subs': '13C'}},
            'M2': {'0002': {'Number': 1,
            'Full': '(0, 0)02',
            'Conc': 0.04096222938811236,
            'Mass': 2,
            'Subs': '34S'},
            ...

    >>> import pandas as pd
    >>> from isotomics import calcIsotopologues as ci
    >>> molecular_df = pd.DataFrame({'IDS': ['C', 'O','S'], 'Number': [2, 1,1], 'deltas': [-10, 0,0]})
    >>> atomDict = ci.inputToAtomDict(molecular_df, disable = False, M1Only = False)
    >>> subDict = ci.subDictionaryFromAtom(atomDict)
    >>> selectedIsotopologues = ci.massSelections(subDict, massThreshold=2)
    >>> selectedIsotopologues
    {'M0': {'': {'Number': 1,
        'Full': ['(0, 0)00'],
        'Conc': 0.9275321060832549,
        'Mass': 0,
        'ATOM': ['0000']}},
        'M1': {'33S': {'Number': 1,
        'Full': ['(0, 0)01'],
        'Conc': 0.0073061703996178,
        'Mass': 1,
        'ATOM': ['0001']},
        '17O': {'Number': 1,
        'Full': ['(0, 0)10'],
        'Conc': 0.0003523694471010285,
        'Mass': 1,
        'ATOM': ['0010']},
        '13C': {'Number': 2,
        'Full': ['(0, 1)00'],
        'Conc': 0.020532221713101362,
        'Mass': 1,
        'ATOM': ['0100']}},
        'M2': {'34S': {'Number': 1,
        'Full': ['(0, 0)02'],
        'Conc': 0.04096222938811236,
        'Mass': 2,
        'ATOM': ['0002']},
        ...
    """
    MNDict = {}
    
    for i in range(massThreshold+1):
        MNDict['M' + str(i)] = {}
        
    for i, v in isotopologueDictionary.items():
        for j in range(massThreshold+1):
            if v['Mass'] == j:
                MNDict['M' + str(j)][i] = v
            
    return MNDict

def introduceClump(clumpD, siteList, clumpAmount, molecularDataFrame):
    """
    Introduce a mass-1 clump while preserving site-specific abundances.

    The function increases the concentration of the selected clumped isotopologue,
    decreases the relevant singly substituted isotopologues, and adjusts the
    unsubstituted isotopologue to conserve totals.

    Args:
        clumpD (dict[str, dict]): By-atom isotopologue dictionary.
        siteList (list[str]): Site names to clump (for example
            ``['Cmethyl', 'Cgamma']``).
        clumpAmount (float): Clump amount in concentration space (not CAP Delta).
        molecularDataFrame (pandas.DataFrame): Molecule definition table.

    Returns:
        dict[str, dict]: Updated by-atom dictionary including the clump.

    Examples:
        >>> import copy
        >>> import pandas as pd
        >>> from isotomics import calcIsotopologues as ci
        >>> molecular_df = pd.DataFrame({'Site Names': ['C1', 'C2'], 'IDS': ['C', 'C'], 'Number': [1, 1], 'deltas': [0, 0]}).set_index('Site Names')
        >>> stoch = ci.inputToAtomDict(molecular_df, disable=True)
        >>> clumped = ci.introduceClump(copy.deepcopy(stoch), ['C1', 'C2'], 1e-6, molecular_df)
        >>> clumped['11']['Conc'] > stoch['11']['Conc']
        True
    """
    siteNameList = []

    siteName = molecularDataFrame.index
    siteNumber = molecularDataFrame.Number

    for i, name in enumerate(siteName):
        siteNameList += [name] * siteNumber[i]
        
    unsub = "0" * len(siteNameList)

    clumpD[unsub]['Conc'] += clumpAmount
    clumpSubList = list(unsub)
    for site in siteList:
        #For multiatomic sites, we need to pick the last index, as this is what all variants of the singly-substituted
        #isotopologue are indexed under
        reverseIndex = siteNameList[::-1].index(site)
        index = len(siteNameList) - reverseIndex - 1

        singleSubList = list(unsub)
        singleSubList[index] = "1"
        clumpSubList[index] = "1"

        singleSubString = "".join(singleSubList)
        #Have to multiply by number of occurences of that isotopologue; each one sees the increase/decrease in Conc
        n = clumpD[singleSubString]['Number']
        clumpD[singleSubString]['Conc'] -= clumpAmount * n


    clumpSubString = "".join(clumpSubList)
    n = clumpD[clumpSubString]['Number']
    clumpD[clumpSubString]['Conc'] += clumpAmount * n
    
    return clumpD

def checkClumpDelta(siteList, molecularDataFrame, clumpD, stochD):
    """
    Print CAP Delta for a specified clump relative to stochastic abundance.

    Args:
        siteList (list[str]): Site names defining the clump.
        molecularDataFrame (pandas.DataFrame): Molecule definition table.
        clumpD (dict[str, dict]): By-atom dictionary with clump applied.
        stochD (dict[str, dict]): Baseline stochastic by-atom dictionary.

    Returns:
        None: Prints the CAP Delta value for the requested clump.

    Examples:
        >>> import copy
        >>> import pandas as pd
        >>> from isotomics import calcIsotopologues as ci
        >>> molecular_df = pd.DataFrame({'Site Names': ['C1', 'C2'], 'IDS': ['C', 'C'], 'Number': [1, 1], 'deltas': [0, 0]}).set_index('Site Names')
        >>> stoch = ci.inputToAtomDict(molecular_df, disable=True)
        >>> clumped = ci.introduceClump(copy.deepcopy(stoch), ['C1', 'C2'], 1e-6, molecular_df)
        >>> ci.checkClumpDelta(['C1', 'C2'], molecular_df, clumped, stoch)
    """
    siteNameList = []

    siteName = molecularDataFrame.index
    siteNumber = molecularDataFrame.Number

    for i, name in enumerate(siteName):
        siteNameList += [name] * siteNumber[i]
        
    unsubStr = "0" * len(siteNameList)
    clumpSubList = ['0'] * len(siteNameList)
    
    for site in siteList:
        #For multiatomic sites, we need to pick the last index, as this is what all variants of the singly-substituted
        #isotopologue are indexed under
        reverseIndex = siteNameList[::-1].index(site)
        index = len(siteNameList) - reverseIndex - 1

        clumpSubList[index] = "1"

    clumpSubString = "".join(clumpSubList)
    
    a = stochD[clumpSubString]['Conc'] / stochD[unsubStr]['Conc']
    b = clumpD[clumpSubString]['Conc'] / clumpD[unsubStr]['Conc']
    capDelta = 1000 * (b/a -1)
    print("Clump at " + ' '.join(siteList) + ' is ' + str(capDelta))
