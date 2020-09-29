import numba
import numpy as np
import awkward

@numba.njit
def do_count_photon_children(mother_index_flat, pdgId_flat, pt_flat, offsets_in):
    content_out = np.empty(len(mother_index_flat), dtype=mother_index_flat.dtype)
    idx_out = 0

    for record_index in range(len(offsets_in) - 1):
        start_src, stop_src = offsets_in[record_index], offsets_in[record_index + 1]

        for index in range(stop_src - start_src):
            mother_pdgId = pdgId_flat[start_src + index]
            nphotonchildren = 0
            if abs(mother_pdgId) in [511, 521, 531]:
                for possible_child in range(index, stop_src - start_src):
                    if (mother_index_flat[start_src + possible_child] == index) \
                    and (pdgId_flat[start_src + possible_child] == 22) \
                    and (pt_flat[start_src + possible_child] < 1.0):
                        nphotonchildren += 1
            content_out[idx_out] = nphotonchildren
            idx_out += 1
    return content_out

def count_photon_children(genPartIdxMother, pdgId, pt):
    return awkward.JaggedArray.fromoffsets(
        pdgId._offsets, 
        do_count_photon_children(
            genPartIdxMother._content, 
            pdgId._content, 
            pt._content, 
            pdgId._offsets
            )
        )
