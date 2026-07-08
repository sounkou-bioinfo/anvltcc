
<!-- README.md is generated from README.Rmd. Please edit that file -->

# anvltcc

`anvltcc` compiles [anvl](https://github.com/r-xla/anvl) traced array
programs to C kernels through
[tccquickr](https://github.com/sounkou-bioinfo/tccquickr) and the TinyCC
just-in-time compiler from
[Rtinycc](https://github.com/sounkou-bioinfo/Rtinycc). It is a
millisecond-latency CPU compile path for anvl programs with no XLA
dependency: TinyCC compiles the emitted kernel in-process, so the
trace-to-callable round trip suits development loops and environments
where the PJRT/XLA toolchain is unavailable.

The bridge works because anvl’s quickr backend already lowers graphs to
declared R — `declare(type(...))` — which is exactly tccquickr’s input
dialect. `anvltcc` reuses anvl’s own quickr lowering rules wherever they
emit code inside tccquickr’s declared subset, and registers bridge-owned
`tinycc` rules for the primitives that have a native tccquickr surface
form: `dot_general` with the standard contraction becomes a `%*%`
contraction nest, and aval-checked broadcasts pass through to
tccquickr’s typed scalar-broadcast and recycle accesses.

## From an anvl trace to a running C kernel

``` r
library(anvltcc)

sigmoid_matmul <- function(x, w) {
  1 / (1 + anvl::nv_exp(-anvl::nv_matmul(x, w)))
}
x <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
w <- matrix(c(0.5, -1, 2, 1, 0, 1), nrow = 3)
```

The traced graph lowers to declared R (anvl’s rules plus the bridge’s
`tinycc` rules), which is also plain R:

``` r
graph <- anvl::trace_fn(sigmoid_matmul, args = list(
  x = anvl::nv_array(x, backend = "plain"),
  w = anvl::nv_array(w, backend = "plain")
))
lowered <- graph_to_tinycc_r_function(graph)
lowered
#> function (x1, x2) 
#> {
#>     declare(x1 = type(x1 = double(2L, 3L)), x2 = type(x2 = double(3L, 
#>         2L)))
#>     anvl_tcc_v1 <- x1 %*% x2
#>     anvl_tcc_v2 <- -anvl_tcc_v1
#>     anvl_tcc_v3 <- exp(anvl_tcc_v2)
#>     anvl_tcc_v4 <- 1
#>     anvl_tcc_v5 <- anvl_tcc_v4 + anvl_tcc_v3
#>     anvl_tcc_v6 <- 1
#>     anvl_tcc_v7 <- anvl_tcc_v6/anvl_tcc_v5
#>     anvl_tcc_out <- anvl_tcc_v7
#>     anvl_tcc_out
#> }
#> <environment: 0x5b03b74081e0>
#> attr(,"anvltcc_constants")
#> named list()
all.equal(as.numeric(lowered(x, w)), as.numeric(1 / (1 + exp(-(x %*% w)))))
#> [1] TRUE
```

tccquickr plans the loop nests and emits the C kernel:

``` r
cat(tinycc_source(sigmoid_matmul, x = x, w = w))
#> #include <math.h>
#> #include <stddef.h>
#> #include <stdlib.h>
#> 
#> double *tccq_c_1196229647(const double *input_0001, const double *input_0002, int result_count_0001) {
#>   if (result_count_0001 < 0) {
#>     return NULL;
#>   }
#>   double *output = (double *)malloc(sizeof(double) * (size_t)result_count_0001);
#>   if (output == NULL) {
#>     return NULL;
#>   }
#>   double *buffer_value_0001 = (double *)malloc(sizeof(double) * (size_t)(2 * 2));
#>   if (buffer_value_0001 == NULL) {
#>     free(output);
#>     return NULL;
#>   }
#>   for (int axis_0001 = 0; axis_0001 < 2; ++axis_0001) {
#>     for (int axis_0002 = 0; axis_0002 < 2; ++axis_0002) {
#>       double acc_value_0001 = 0.0;
#>       for (int axis_0003 = 0; axis_0003 < 3; ++axis_0003) {
#>         acc_value_0001 = acc_value_0001 + (input_0001[axis_0001 + axis_0003 * 2] * input_0002[axis_0003 + axis_0002 * 3]);
#>       }
#>       buffer_value_0001[axis_0001 + axis_0002 * 2] = acc_value_0001;
#>     }
#>   }
#>   for (int axis_0001 = 0; axis_0001 < 2; ++axis_0001) {
#>     for (int axis_0002 = 0; axis_0002 < 2; ++axis_0002) {
#>       output[axis_0001 + axis_0002 * 2] = (                 1.0 / (                 1.0 + exp((-buffer_value_0001[axis_0001 + axis_0002 * 2]))));
#>     }
#>   }
#>   free(buffer_value_0001);
#>   return output;
#> }
```

TinyCC compiles it in-process, and the callable agrees with R:

``` r
kernel <- tinycc_jit(sigmoid_matmul, x = x, w = w)
all.equal(as.numeric(kernel(x, w)), as.numeric(1 / (1 + exp(-(x %*% w)))))
#> [1] TRUE
```

## What is covered

- Elementwise primitives, scalar broadcast, and reductions, through
  anvl’s own quickr rules.
- `dot_general` with no batch dimensions and the standard `(2, 1)`
  contraction, through the bridge’s `%*%` rule.
- `broadcast_in_dim` whose operand is scalar/length-1 or already the
  target shape, as an aval-checked passthrough.
- Graph constants (closed-over arrays), bound automatically to the
  compiled callable.

A primitive configuration outside this surface falls back to anvl’s
quickr rule; when that rule emits loops or mutation, tccquickr refuses
with a structured diagnostic naming the construct, so failures state the
next lowering rule to add rather than miscompiling. Graphs must
currently have one output leaf and no static arguments.

## Design

The bridge follows anvl’s documented extension pattern: `.onLoad`
registers a `tinycc` interpretation-rule type and attaches rules to
primitives via `primitive[["tinycc"]] <- rule`. The codegen driver walks
the graph once with a two-level rule lookup — `tinycc` first, `quickr`
fallback — so the bridge owns exactly the primitives whose quickr code
would leave tccquickr’s declared subset, and anvl’s rule coverage does
the rest.
