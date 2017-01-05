#pragma once

#include "common.h"
#include "convolution.h"
#include "wavelets.h"

#ifdef TYPE
#error TYPE should not be defined here.
#else

#define TYPE float
#include "wt.template.h"
#undef TYPE

#define TYPE double
#include "wt.template.h"
#undef TYPE

#ifdef HAVE_C99_COMPLEX
    #define TYPE float_complex
    #include "wt.template.h"
    #undef TYPE

    #define TYPE double_complex
    #include "wt.template.h"
    #undef TYPE
#endif

#endif /* TYPE */
