#ifndef __MUILM_GATEUP_METHOD_H__
#define __MUILM_GATEUP_METHOD_H__

enum MuiLLMgateupmlpMethod {
    // Basic method where Gate/Up projections + mul are done distinctly
    gateupmlp_UNFUSED = 0,
    // Method where the Gate/Up projections + mul are all fused
    gateupmlp_FUSED = 1,
    // Method where the Gate/Up projections are done in the same kernel
    // but split between blocks to have more blocks.
    // A final reduction is done in an epilogue kernel
    gateupmlp_SPLIT = 2
};

#endif /* __MUILM_GATEUP_METHOD_H__ */