#' Bridge lowering rules
#'
#' A tinycc rule has the same calling convention as anvl's quickr rules
#' (`function(prim_name, inputs, params, out_syms, input_nodes, out_avals,
#' ctx)`) and returns a list of statements, or `NULL` to decline so the
#' driver falls back to the primitive's quickr rule. A rule may only claim
#' a primitive configuration whose emitted code lies inside tccquickr's
#' declared subset; configurations that would need loops or mutation must
#' decline.
#'
#' @name tinycc-rules
#' @keywords internal
NULL

aval_shape <- function(aval) {
  as.integer(tengen::shape(aval))
}

register_tinycc_rules <- function() {
  # AnvlPrimitive rules live in an environment, so mutating through a local
  # handle attaches the rule package-wide; assigning through `anvl::` would
  # try to rebind the namespace object instead.
  prim_dot_general <- anvl::prim_dot_general
  prim_broadcast_in_dim <- anvl::prim_broadcast_in_dim

  # dot_general with no batch dimensions and the standard (2, 1) contraction
  # is exactly R's `%*%`, which tccquickr lowers as a typed contraction
  # nest. Every other configuration declines to anvl's quickr rule.
  prim_dot_general[["tinycc"]] <- function(
    prim_name,
    inputs,
    params,
    out_syms,
    input_nodes,
    out_avals,
    ctx = NULL
  ) {
    lhs_shape <- aval_shape(input_nodes[[1L]]$aval)
    rhs_shape <- aval_shape(input_nodes[[2L]]$aval)
    contracting <- lapply(params$contracting_dims, as.integer)
    batching <- lapply(params$batching_dims, as.integer)
    standard <- length(batching[[1L]]) == 0L &&
      length(batching[[2L]]) == 0L &&
      length(lhs_shape) == 2L &&
      length(rhs_shape) %in% c(1L, 2L) &&
      identical(contracting[[1L]], 2L) &&
      identical(contracting[[2L]], 1L)
    if (!standard) {
      return(NULL)
    }
    list(as.call(list(
      as.name("<-"),
      out_syms[[1L]],
      as.call(list(as.name("%*%"), inputs[[1L]], inputs[[2L]]))
    )))
  }

  # A broadcast whose operand is scalar/length-1, or whose operand shape
  # already equals the target, is the identity under tccquickr's typed
  # scalar-broadcast and recycle accesses. Rank-changing broadcasts decline.
  prim_broadcast_in_dim[["tinycc"]] <- function(
    prim_name,
    inputs,
    params,
    out_syms,
    input_nodes,
    out_avals,
    ctx = NULL
  ) {
    shape_in <- aval_shape(input_nodes[[1L]]$aval)
    shape_out <- as.integer(params$shape)
    passthrough <- prod(shape_in) == 1L || identical(shape_in, shape_out)
    if (!passthrough) {
      return(NULL)
    }
    list(as.call(list(as.name("<-"), out_syms[[1L]], inputs[[1L]])))
  }

  invisible(NULL)
}
