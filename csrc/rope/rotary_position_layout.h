#ifndef __MUILLM_ROTARY_POSITION_LAYOUT_H__
#define __MUILLM_ROTARY_POSITION_LAYOUT_H__

typedef enum muillm_rotary_cache_layout {
  // [S, E] where S is the size of the cache, E embedding size
  ROTARY_CACHE_SE_LAYOUT = 0,
  // [B, T, E] where B is the input batch size, T number of tokens in the input, E embedding size
  ROTARY_CACHE_BTE_LAYOUT = 1
} muillm_rotary_cache_layout_t;

#endif /* __MUILLM_ROTARY_POSITION_LAYOUT_H__ */