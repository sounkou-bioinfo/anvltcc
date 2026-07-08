library(tinytest)
library(anvltcc)

if (!requireNamespace("anvl", quietly = TRUE)) {
  exit_file("anvl is not installed")
}
if (!requireNamespace("Rtinycc", quietly = TRUE)) {
  exit_file("Rtinycc is not installed")
}

# TinyCC's in-memory JIT can crash the whole process on platforms with
# strict W^X code-page policies; probe in a subprocess as tccquickr does.
can_jit_with_rtinycc <- function() {
  probe_path <- tempfile("anvltcc_jit_probe_", fileext = ".R")
  writeLines(c(
    "ffi <- Rtinycc::tcc_ffi()",
    "ffi <- Rtinycc::tcc_bind(",
    "  ffi,",
    "  anvltcc_probe_add = list(args = list(\"f64\", \"f64\"), returns = \"f64\")",
    ")",
    "ffi <- Rtinycc::tcc_source(",
    "  ffi,",
    "  \"double anvltcc_probe_add(double a, double b) { return a + b; }\"",
    ")",
    "compiled <- Rtinycc::tcc_compile(ffi)",
    "stopifnot(identical(compiled[[\"anvltcc_probe_add\"]](1, 2), 3))",
    "cat(\"anvltcc-jit-ok\")"
  ), probe_path)
  on.exit(unlink(probe_path), add = TRUE)
  output <- tryCatch(
    suppressWarnings(system2(
      file.path(R.home("bin"), "Rscript"),
      c("--vanilla", probe_path),
      stdout = TRUE,
      stderr = TRUE
    )),
    error = function(e) character()
  )
  status <- attr(output, "status")
  (is.null(status) || identical(as.integer(status), 0L)) &&
    any(grepl("anvltcc-jit-ok", output, fixed = TRUE))
}

if (!can_jit_with_rtinycc()) {
  exit_file("TinyCC in-memory JIT is not viable on this platform")
}

# Elementwise with scalar broadcast: the lowered graph passes through the
# bridge's broadcast_in_dim rule and compiles as one map nest.
f <- function(a, b, x) a * x + b
k1 <- tinycc_jit(f, a = 2, b = 1, x = c(1, 2, 3))
expect_equal(as.numeric(k1(2, 1, c(1, 2, 3))), f(2, 1, c(1, 2, 3)))

# Reduction over an elementwise subtree.
g <- function(x) sum(x * x)
k2 <- tinycc_jit(g, x = c(1, 2, 3, 4))
expect_equal(k2(c(1, 2, 3, 4)), sum(c(1, 2, 3, 4)^2))

# Matrix product through the bridge's dot_general rule (%*% contraction).
h <- function(x, w) anvl::nv_matmul(x, w)
xm <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), nrow = 3)
wm <- matrix(c(0.5, -1, 2, 1, 0, 1, -0.5, 0.25), nrow = 4)
k3 <- tinycc_jit(h, x = xm, w = wm)
expect_equal(as.numeric(k3(xm, wm)), as.numeric(xm %*% wm))

# A composed chain: matmul buffer feeding an elementwise sigmoid.
sigmoid <- function(x, w) 1 / (1 + anvl::nv_exp(-anvl::nv_matmul(x, w)))
k4 <- tinycc_jit(sigmoid, x = xm, w = wm)
expect_equal(as.numeric(k4(xm, wm)), as.numeric(1 / (1 + exp(-(xm %*% wm)))))

# The source path emits a C kernel without compiling it.
c_source <- tinycc_source(h, x = xm, w = wm)
expect_true(is.character(c_source) && length(c_source) == 1L)
expect_true(grepl("double", c_source, fixed = TRUE))

# The lowered function is also plain R.
lowered <- graph_to_tinycc_r_function(
  anvl::trace_fn(h, args = list(
    x = anvl::nv_array(xm, backend = "plain"),
    w = anvl::nv_array(wm, backend = "plain")
  ))
)
expect_equal(as.numeric(lowered(xm, wm)), as.numeric(xm %*% wm))
