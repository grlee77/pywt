#pragma once

#include "common.h"

#ifdef TYPE
#error TYPE should not be defined here.
#else

#define TYPE float
#include "convolution.template.h"
#undef TYPE

#define TYPE double
#include "convolution.template.h"
#undef TYPE

#ifdef HAVE_C99_COMPLEX
    #define TYPE float_complex
    #include "convolution.template.h"
    #undef TYPE

    #define TYPE double_complex
    #include "convolution.template.h"
    #undef TYPE
#endif

#endif /* TYPE */
