as_anvl_args <- function(args) {
  lapply(args, function(arg) {
    if (inherits(arg, "AnvlArray")) {
      arg
    } else if (is.numeric(arg) && length(arg) == 1L && is.null(dim(arg))) {
      # anvl only auto-broadcasts true scalars (shape integer(0)).
      anvl::nv_scalar(arg, backend = "plain")
    } else {
      anvl::nv_array(arg, backend = "plain")
    }
  })
}

tinycc_plan <- function(f, args, mode) {
  graph <- anvl::trace_fn(f, args = as_anvl_args(args))
  lowered <- graph_to_tinycc_r_function(graph)
  analysis <- tccquickr::tccq_analyze(lowered)
  if (!S7::prop(analysis, "success")) {
    stop(
      paste(
        c(
          "anvltcc: tccquickr cannot compile the lowered graph:",
          vapply(
            S7::prop(analysis, "diagnostics"),
            function(diagnostic) {
              sprintf(
                "- %s: %s",
                S7::prop(diagnostic, "code"),
                S7::prop(diagnostic, "message")
              )
            },
            character(1)
          )
        ),
        collapse = "\n"
      ),
      call. = FALSE
    )
  }
  plan <- tccquickr::tccq_plan_backend(
    S7::prop(analysis, "value"),
    tccquickr::tccq_rtinycc_backend(),
    tccquickr::tccq_backend_context(mode = mode, target = "c")
  )
  if (!S7::prop(plan, "success")) {
    stop(
      paste(
        c(
          "anvltcc: the TinyCC backend declined the lowered graph:",
          vapply(
            S7::prop(plan, "diagnostics"),
            function(diagnostic) {
              sprintf(
                "- %s: %s",
                S7::prop(diagnostic, "code"),
                S7::prop(diagnostic, "message")
              )
            },
            character(1)
          )
        ),
        collapse = "\n"
      ),
      call. = FALSE
    )
  }
  list(
    products = S7::prop(S7::prop(plan, "value"), "products"),
    constants = attr(lowered, "anvltcc_constants"),
    lowered = lowered
  )
}

#' Compile an anvl-traceable function to a TinyCC kernel
#'
#' Traces `f` with [anvl::trace_fn()], lowers the graph through
#' [graph_to_tinycc_r_function()], and JIT-compiles the result with
#' tccquickr's Rtinycc backend. The returned function takes plain R
#' scalars/vectors/arrays in the traced argument order and returns plain R
#' values; traced constants are bound automatically.
#'
#' @param f (`function`)\cr Function over anvl arrays to trace.
#' @param ... Named example arguments for tracing: `AnvlArray`s, or plain R
#'   values which are wrapped with the `"plain"` anvl backend.
#' @return A compiled function over plain R values.
#' @export
tinycc_jit <- function(f, ...) {
  compiled <- tinycc_plan(f, list(...), mode = "jit")
  callable <- S7::prop(compiled$products, "attrs")$callable
  constants <- compiled$constants
  function(...) {
    do.call(callable, c(list(...), constants))
  }
}

#' Show the C source for an anvl-traceable function
#'
#' Same pipeline as [tinycc_jit()] but stops at source emission.
#'
#' @inheritParams tinycc_jit
#' @return (`character(1)`) the generated C kernel source.
#' @export
tinycc_source <- function(f, ...) {
  compiled <- tinycc_plan(f, list(...), mode = "source")
  S7::prop(compiled$products, "attrs")$source
}
