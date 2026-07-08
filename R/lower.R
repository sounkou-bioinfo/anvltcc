#' Lower an anvl graph to a declared R function
#'
#' Walks an [`anvl::AnvlGraph`]'s calls in order and emits one declared R
#' function in tccquickr's input dialect. Each primitive is lowered by its
#' `tinycc` bridge rule when one claims the configuration, and by anvl's own
#' `quickr` rule otherwise, so the bridge only owns the primitives whose
#' quickr code would leave tccquickr's declared subset.
#'
#' Graph constants become trailing formals; their traced values are attached
#' as the `anvltcc_constants` attribute so callers can complete the call.
#'
#' @param graph (`AnvlGraph`)\cr A graph traced by [anvl::trace_fn()].
#' @return A function whose body starts with `declare(type(...))`, suitable
#'   for [tccquickr::tccq_analyze()] and plain R evaluation alike.
#' @export
graph_to_tinycc_r_function <- function(graph) {
  ns <- getNamespace("anvl")
  if (!ns$is_graph(graph)) {
    stop("`graph` must be an <AnvlGraph> from anvl::trace_fn().", call. = FALSE)
  }
  if (!(inherits(graph$out_tree, "LeafNode") && length(graph$outputs) == 1L)) {
    stop(
      "anvltcc currently lowers graphs with exactly one output leaf.",
      call. = FALSE
    )
  }
  if (isTRUE(any(graph$is_static_flat))) {
    stop("anvltcc does not lower graphs with static arguments yet.", call. = FALSE)
  }

  user_arg_names <- ns$quickr_user_arg_names(length(graph$inputs))
  n_const <- length(graph$constants)
  const_arg_names <- if (n_const) paste0("anvl_const", seq_len(n_const)) else character()
  all_arg_names <- c(user_arg_names, const_arg_names)

  arg_avals <- c(
    lapply(graph$inputs, function(node) node$aval),
    lapply(graph$constants, function(node) node$aval)
  )
  declare_stmt <- ns$quickr_declare_stmt(all_arg_names, arg_avals)

  node_expr <- utils::hashtab()
  for (i in seq_along(graph$inputs)) {
    node_expr[[graph$inputs[[i]]]] <- as.name(user_arg_names[[i]])
  }
  const_values <- vector("list", n_const)
  for (i in seq_along(graph$constants)) {
    const_node <- graph$constants[[i]]
    if (!ns$is_graph_value(const_node) || !ns$is_concrete_tensor(const_node$aval)) {
      stop("anvltcc: graph constants must be concrete arrays.", call. = FALSE)
    }
    node_expr[[const_node]] <- as.name(const_arg_names[[i]])
    const_values[[i]] <- anvl::as_array(const_node$aval$data)
  }
  names(const_values) <- const_arg_names

  tmp_i <- 0L
  new_tmp_sym <- function() {
    tmp_i <<- tmp_i + 1L
    as.name(sprintf("anvl_tcc_v%d", tmp_i))
  }
  ctx <- list(node_expr = node_expr, new_tmp_sym = new_tmp_sym)

  stmts <- list(declare_stmt)
  for (graph_call in graph$calls) {
    input_exprs <- lapply(
      graph_call$inputs,
      ns$quickr_expr_of_node,
      node_expr = node_expr
    )
    out_syms <- vector("list", length(graph_call$outputs))
    out_avals <- vector("list", length(graph_call$outputs))
    for (i in seq_along(graph_call$outputs)) {
      out_node <- graph_call$outputs[[i]]
      if (!ns$is_graph_value(out_node)) {
        stop("anvltcc: non-GraphValue primitive outputs are unsupported.", call. = FALSE)
      }
      sym <- new_tmp_sym()
      node_expr[[out_node]] <- sym
      out_syms[[i]] <- sym
      out_avals[[i]] <- out_node$aval
    }

    lower_call <- function(rule) {
      rule(
        graph_call$primitive$name,
        input_exprs,
        graph_call$params,
        out_syms,
        graph_call$inputs,
        out_avals,
        ctx = ctx
      )
    }
    call_stmts <- NULL
    tinycc_rule <- graph_call$primitive[["tinycc"]]
    if (!is.null(tinycc_rule)) {
      call_stmts <- lower_call(tinycc_rule)
    }
    if (is.null(call_stmts)) {
      quickr_rule <- graph_call$primitive[["quickr"]]
      if (is.null(quickr_rule)) {
        stop(
          sprintf(
            "anvltcc: primitive `%s` has neither a tinycc nor a quickr lowering rule.",
            graph_call$primitive$name
          ),
          call. = FALSE
        )
      }
      call_stmts <- lower_call(quickr_rule)
    }
    stmts <- c(stmts, call_stmts)
  }

  out_expr <- ns$quickr_expr_of_node(graph$outputs[[1L]], node_expr = node_expr)
  result_sym <- as.name("anvl_tcc_out")
  stmts <- c(stmts, list(as.call(list(as.name("<-"), result_sym, out_expr)), result_sym))

  lowered <- function() NULL
  formals(lowered) <- as.pairlist(stats::setNames(
    rep(list(quote(expr = )), length(all_arg_names)),
    all_arg_names
  ))
  body(lowered) <- as.call(c(list(as.name("{")), stmts))
  lowered_env <- new.env(parent = baseenv())
  lowered_env$declare <- function(...) invisible(NULL)
  environment(lowered) <- lowered_env
  attr(lowered, "anvltcc_constants") <- const_values
  lowered
}
