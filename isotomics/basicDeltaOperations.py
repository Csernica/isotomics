import numpy as np
import pandas as pd

'''
This code takes care of basic manipulations between delta, ratio, and concentration space. Currently, it works for H, C, N, O, and S. 
'''
                                                                                                                       

#DEFINE STANDARDS
#Doubly isotopic atoms are given as standard ratios. The standards are: VPDB for carbon, AIR for nitrogen, VSMOW for H/O, and CDT for sulfur. 
STD_Rs = {"H": 0.00015576, "C": 0.011180, "N": 0.003676, "17O": 0.0003799, "18O": 0.0020052,
         "33S":0.007877,"34S":0.0441626,"36S":0.000105274}
    
def deltaToConcentration(atomIdentity,delta):
    """
    Convert a delta value into internal isotope concentrations.

    For isotope systems with multiple rare isotopes (for example O and S),
    additional isotope concentrations are set using the mass-scaling rules
    used by Isotomics.

    Args:
        atomIdentity: Isotope key identifying the ratio system (for example,
            ``"13C"``, ``"15N"``, ``"D"``, ``"17O"``, ``"33S"``).
        delta: Delta value in per mil for the specified isotope system.

    Returns:
        tuple[float, float, float, float]:
        Concentrations in internal isotope ordering:
        ``(unsubstituted, M+1, M+2, M+3)``.
        For systems with fewer than four components, unused entries are ``0``.

    Example:
        >>> deltaToConcentration("13C", -10)
        (0.989052963984032, 0.010947036015968062, 0, 0)
    """
    if type(delta) == tuple:
        return twoDeltasToConcentration(atomIdentity, delta)
    
    if atomIdentity == 'C' or atomIdentity == '13C':
        ratio = (delta/1000+1)*STD_Rs['C']
        concentrationSub = ratio/(1+ratio)
        
        return (1-concentrationSub,concentrationSub,0,0)
    
    if atomIdentity == 'H' or atomIdentity == 'D':
        ratio = (delta/1000+1)*STD_Rs['H']
        concentrationSub = ratio/(1+ratio)
        
        return (1-concentrationSub,concentrationSub,0,0)
    
    if atomIdentity == 'N' or atomIdentity == '15N':
        ratio = (delta/1000+1)*STD_Rs['N']
        concentrationSub = ratio/(1+ratio)
        
        return (1-concentrationSub,concentrationSub,0,0)
    
    elif atomIdentity == '17O' or atomIdentity == 'O':
        r17 = (delta/1000+1)*STD_Rs['17O']
        delta18 = 1/0.52 * delta
        r18 = (delta18/1000+1)*STD_Rs['18O']
        
        o17 = r17/(1+r17+r18)
        o18 = r18/(1+r17+r18)
        o16 = 1-(o17+o18)
        
        return (o16,o17,o18,0)
    
    elif atomIdentity == '33S' or atomIdentity == 'S':
        r33 = (delta/1000+1)*STD_Rs['33S']
        delta34 = delta/0.515
        r34 = (delta34/1000+1)*STD_Rs['34S']
        delta36 = delta34*1.9
        r36 = (delta36/1000+1)*STD_Rs['36S']
        
        s33 = r33/(1+r33 + r34 + r36)
        s34 = r34/(1+r33+r34+r36)
        s36 = r36/(1+r33+r34+r36)

        s32 = 1-(s33+s34+s36)
        
        return (s32,s33,s34,s36)
        
    elif atomIdentity == '34S':
        r34 = (delta/1000+1)*STD_Rs['34S']
        delta33 = delta * 0.515
        r33 = (delta/1000+1)*STD_Rs['33S']
        delta36 = delta*1.9
        r36 = (delta36/1000+1)*STD_Rs['36S']
        
        s33 = r33/(1+r33 + r34 + r36)
        s34 = r34/(1+r33+r34+r36)
        s36 = r36/(1+r33+r34+r36)

        s32 = 1-(s33+s34+s36)
        
        return (s32,s33,s34,s36)
                
    else:
        raise Exception('Sorry, I do not know how to deal with ' + atomIdentity)

def twoDeltasToConcentration(atomIdentity, deltaTuple):
    """
    Convert paired delta constraints into isotope concentrations.

    This variant handles systems where two deltas are explicitly constrained
    (17/18O or 33/34S), instead of deriving the second via mass scaling.

    Args:
        atomIdentity: ``"O"`` or ``"S"``.
        deltaTuple: Two-element tuple. First value is 17O or 33S delta;
            second value is 18O or 34S delta.

    Returns:
        tuple[float, float, float, float]:
        Concentration vector for the isotope system.
        For sulfur, ordering is ``(32S, 33S, 34S, 36S)``.

    Example:
        >>> twoDeltasToConcentration("O", (-10, -20))
        (0.9976642714007893, 0.00037522253013810825, 0.001960506069072605, 0)
    """
    if atomIdentity == 'O':
        delta17 = deltaTuple[0]
        delta18 = deltaTuple[1]
        
        r17 = (delta17/1000+1)*STD_Rs['17O']
        r18 = (delta18/1000+1)*STD_Rs['18O']
        
        o17 = r17/(1+r17+r18)
        o18 = r18/(1+r17+r18)
        o16 = 1-(o17+o18)
        
        return (o16,o17,o18,0)
    
    elif atomIdentity == 'S':
        delta33 = deltaTuple[0]
        delta34 = deltaTuple[1]
        
        r33 = (delta33/1000+1)*STD_Rs['33S']
        r34 = (delta34/1000+1)*STD_Rs['34S']
        delta36 = delta34*1.9
        r36 = (delta36/1000+1)*STD_Rs['36S']
        
        s33 = r33/(1+r33 + r34 + r36)
        s34 = r34/(1+r33+r34+r36)
        s36 = r36/(1+r33+r34+r36)

        s32 = 1-(s33+s34+s36)
        
        return (s32,s33,s34,s36)
    
def concentrationToM1Ratio(concentrationTuple):
    """
    Compute the M+1-to-unsubstituted ratio from a concentration vector.

    Args:
        concentrationTuple: Four-element concentration vector in the internal
            isotope ordering.

    Returns:
        float: Ratio of the mass-1 isotope to the unsubstituted isotope.

    Example:
        >>> concentrationToM1Ratio((0.989052963984032, 0.010947036015968062, 0, 0))
        0.0110682
    """
    return concentrationTuple[1]/concentrationTuple[0] 

def ratioToDelta(atomIdentity, ratio):
    """
    Convert an isotope ratio into a delta value.

    Args:
        atomIdentity: Isotope key of interest.
        ratio: Isotope ratio relative to the unsubstituted isotope.

    Returns:
        float: Delta value for the provided isotope ratio.

    Example:
        >>> ratioToDelta("13C", 0.0110682)
        -10.000000000000009
    """
    delta = 0
    if atomIdentity == 'D':
        atomIdentity = 'H'
        
    if atomIdentity in 'HCN' or atomIdentity in ['13C','15N']:
        #in case atomIdentity is 2H, 13C, 15N, take last character only
        delta = (ratio/STD_Rs[atomIdentity[-1]]-1)*1000
        
    elif atomIdentity == 'O' or atomIdentity == '17O':
        delta = (ratio/STD_Rs['17O']-1)*1000
        
    elif atomIdentity == '18O':
        delta = (ratio/STD_Rs['18O']-1)*1000
        
    elif atomIdentity == 'S' or atomIdentity == '33S':
        delta = (ratio/STD_Rs['33S']-1)*1000
        
    elif atomIdentity == '34S':
        delta = (ratio/STD_Rs['34S']-1)*1000
        
    elif atomIdentity == '36S':
        delta = (ratio/STD_Rs['36S']-1)*1000
        
    else:
        raise Exception('Sorry, I do not know how to deal with ' + atomIdentity)
        
    return delta

def compareRelDelta(atomID, deltaStd, deltaSmp):
    """
    Compute sample-relative-to-standard delta in CRF space.

    This comparison accounts for non-additivity of delta values.

    Args:
        atomID: Isotope key passed to :func:`deltaToConcentration`.
        deltaStd: Delta value of the standard (denominator).
        deltaSmp: Delta value of the sample (numerator).

    Returns:
        float: Sample-relative-to-standard delta value.

    Example:
        >>> compareRelDelta("13C", -10, -20)
        -10.101010101010166
    """
    rStd = concentrationToM1Ratio(deltaToConcentration(atomID,deltaStd))
    rSmp = concentrationToM1Ratio(deltaToConcentration(atomID,deltaSmp))

    relDelta = 1000*(rSmp/rStd-1)
    
    return relDelta